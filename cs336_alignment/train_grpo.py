"""
GRPO training script for OLMo-2-0425-1B on GSM8K dataset.
Uses two-GPU setup: one for vLLM rollout generation, one for training.
All hyperparameters are hardcoded as specified in the assignment.
"""

import argparse
import json
import torch
import wandb
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams

from grpo import grpo_train_step
from checkpoint import get_model_and_tokenizer
from drgrpo_grader import r1_zero_reward_fn

# Hardcoded configuration
RANDOM_SEED = 45
MODEL_NAME = "allenai/OLMo-2-0425-1B"
DATASET_PATH = "../data/gsm8k"
PROMPT_PATH = "prompts/r1_zero.prompt"

# Training hyperparameters
N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024
NUM_ROLLOUT_STEPS = 250  # override with --num-rollout-steps
LEARNING_RATE = 1e-5  # default; override with --lr (see sweep_lr.sh)
ROLLOUT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 256
GROUP_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 256  # microbatch of 1 seq; 8 OOMs on a 23GB L4
SAMPLING_TEMPERATURE = 1.0
SAMPLING_MAX_TOKENS = 512
MAX_GRAD_NORM = 1.0

# GRPO settings
BASELINE = "mean"
ADVANTAGE_NORMALIZER = "std"
LOSS_NORMALIZATION = "sequence"

# Logging
EVAL_EVERY = 10
LOG_ROLLOUTS_EVERY = 40
SAVE_EVERY = 100
CHECKPOINT_DIR = "checkpoints/grpo"

# GPU assignment
TRAINING_GPU = 0  # GPU for training model
VLLM_GPU = 1      # GPU for vLLM rollout generation

# WandB config
WANDB_PROJECT = "cs336-grpo"


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K.")
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--num-rollout-steps",
        type=int,
        default=NUM_ROLLOUT_STEPS,
        help="Number of rollout/train steps.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="WandB run name. Defaults to grpo_lr=<lr>.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Defaults to <CHECKPOINT_DIR>/lr_<lr>.",
    )
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"grpo_lr={args.lr:g}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(CHECKPOINT_DIR) / f"lr_{args.lr:g}")
    return args


def extract_gsm8k_answer(answer: str) -> str:
    """Extract the final answer from a GSM8K worked solution."""
    if "####" not in answer:
        raise ValueError(f"Malformed GSM8K answer: {answer!r}")
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def load_dataset(split: str) -> list[dict]:
    """Load GSM8K dataset."""
    file_path = Path(DATASET_PATH) / f"{split}.jsonl"
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            item["answer"] = extract_gsm8k_answer(item["answer"])
            data.append(item)
    return data


def load_prompt_template() -> str:
    """Load prompt template."""
    with open(PROMPT_PATH, 'r') as f:
        return f.read()


def format_prompt(question: str, template: str) -> str:
    """Format a question with the prompt template."""
    return template.replace("{question}", question)


def sync_weights_to_vllm(training_model, vllm_engine, temp_dir):
    """
    Sync weights from training model to vLLM engine.

    Strategy: Save the training model's weights to a temporary checkpoint,
    then have each vLLM worker reload its weights from that checkpoint.

    The vLLM engine runs in its own process (and on a different GPU), so the
    training model's tensors can't be handed to it directly. `reload_weights`
    only needs the checkpoint path, which crosses the process boundary cheaply.
    """
    training_model.save_pretrained(temp_dir, safe_serialization=True)
    vllm_engine.collective_rpc("reload_weights", kwargs={"weights_path": temp_dir})


def generate_rollouts(
    vllm_engine: LLM,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
    group_size: int,
) -> tuple[list[str], list[str]]:
    """Generate rollouts using vLLM on dedicated GPU."""
    # Repeat each prompt group_size times
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * group_size)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    outputs = vllm_engine.generate(repeated_prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    return repeated_prompts, responses


def evaluate(
    vllm_engine: LLM,
    val_data: list[dict],
    prompt_template: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, float]:
    """Evaluate on validation set using vLLM."""
    prompts = [format_prompt(item["question"], prompt_template) for item in val_data]
    ground_truths = [item["answer"] for item in val_data]

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    outputs = vllm_engine.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    total_reward = 0.0
    format_reward = 0.0
    answer_reward = 0.0

    for response, gt in zip(responses, ground_truths):
        result = r1_zero_reward_fn(response, gt)
        total_reward += result["reward"]
        format_reward += result["format_reward"]
        answer_reward += result["answer_reward"]

    n = len(val_data)
    return {
        "val_reward": total_reward / n,
        "val_format_reward": format_reward / n,
        "val_answer_reward": answer_reward / n,
    }


def main():
    args = parse_args()
    learning_rate = args.lr
    num_rollout_steps = args.num_rollout_steps
    checkpoint_dir = args.checkpoint_dir

    # Set random seed
    torch.manual_seed(RANDOM_SEED)

    # Create temp directory for weight syncing
    temp_dir = tempfile.mkdtemp()

    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=args.run_name,
        config={
            "random_seed": RANDOM_SEED,
            "model_name": MODEL_NAME,
            "n_train_examples": N_TRAIN_EXAMPLES,
            "n_val_examples": N_VAL_EXAMPLES,
            "num_rollout_steps": num_rollout_steps,
            "learning_rate": learning_rate,
            "rollout_batch_size": ROLLOUT_BATCH_SIZE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "sampling_max_tokens": SAMPLING_MAX_TOKENS,
            "max_grad_norm": MAX_GRAD_NORM,
            "baseline": BASELINE,
            "advantage_normalizer": ADVANTAGE_NORMALIZER,
            "loss_normalization": LOSS_NORMALIZATION,
            "save_every": SAVE_EVERY,
            "checkpoint_dir": checkpoint_dir,
            "training_gpu": TRAINING_GPU,
            "vllm_gpu": VLLM_GPU,
        }
    )

    print("Loading dataset...")
    train_data = load_dataset("train")[:N_TRAIN_EXAMPLES]
    val_data = load_dataset("test")[:N_VAL_EXAMPLES]
    prompt_template = load_prompt_template()

    print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")

    # Initialize training model on GPU 0
    print(f"Loading training model on GPU {TRAINING_GPU}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, device=f"cuda:{TRAINING_GPU}")
    device = torch.device(f"cuda:{TRAINING_GPU}")
    model = model.to(device)
    model.config.use_cache = False  # no KV cache needed for training forwards

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # Initialize vLLM engine on GPU 1
    print(f"Initializing vLLM engine on GPU {VLLM_GPU}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(VLLM_GPU)
    vllm_engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    # Reset CUDA_VISIBLE_DEVICES to allow training on GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{TRAINING_GPU},{VLLM_GPU}"

    # Training loop
    print("Starting training...")
    global_step = 0

    for step in range(num_rollout_steps):
        print(f"STEP: {step}")
        # Sample a batch of training examples
        batch_indices = torch.randint(0, len(train_data), (ROLLOUT_BATCH_SIZE // GROUP_SIZE,)).tolist()
        batch_data = [train_data[i] for i in batch_indices]

        batch_prompts = [format_prompt(item["question"], prompt_template) for item in batch_data]
        batch_ground_truths = [item["answer"] for item in batch_data]

        # Sync weights from training model to vLLM engine
        sync_weights_to_vllm(model, vllm_engine, temp_dir)

        # Generate rollouts on GPU 1 (vLLM)
        repeated_prompts, rollout_responses = generate_rollouts(
            vllm_engine,
            batch_prompts,
            SAMPLING_TEMPERATURE,
            SAMPLING_MAX_TOKENS,
            GROUP_SIZE,
        )

        repeated_ground_truths = []
        for gt in batch_ground_truths:
            repeated_ground_truths.extend([gt] * GROUP_SIZE)

        # Training step on GPU 0
        loss, metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=GROUP_SIZE,
            baseline=BASELINE,
            advantage_eps=1e-6,
            advantage_normalizer=ADVANTAGE_NORMALIZER,
            loss_normalization=LOSS_NORMALIZATION,
        )

        # Log training metrics
        wandb.log({
            "step": step,
            **metadata,
        })

        # Save a persistent checkpoint after every SAVE_EVERY completed steps.
        completed_steps = step + 1
        if completed_steps % SAVE_EVERY == 0:
            checkpoint_path = Path(checkpoint_dir) / f"step_{completed_steps}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_path, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Log rollouts periodically
        if step % LOG_ROLLOUTS_EVERY == 0:
            rollout_table = wandb.Table(
                columns=["step", "prompt", "response", "ground_truth"],
                data=[
                    [step, repeated_prompts[i], rollout_responses[i], repeated_ground_truths[i]]
                    for i in range(min(5, len(repeated_prompts)))
                ]
            )
            wandb.log({"rollouts": rollout_table})

        # Evaluate periodically
        if (step + 1) % EVAL_EVERY == 0:
            print(f"\nEvaluating at step {step + 1}...")

            # Sync weights before evaluation
            sync_weights_to_vllm(model, vllm_engine, temp_dir)

            eval_metrics = evaluate(
                vllm_engine,
                val_data,
                prompt_template,
                SAMPLING_TEMPERATURE,
                SAMPLING_MAX_TOKENS,
            )
            wandb.log({
                "step": step,
                **eval_metrics,
            })
            print(f"Validation reward: {eval_metrics['val_reward']:.4f}")

        global_step += 1

    # Final evaluation
    print("\nFinal evaluation...")
    sync_weights_to_vllm(model, vllm_engine, temp_dir)
    final_metrics = evaluate(
        vllm_engine,
        val_data,
        prompt_template,
        SAMPLING_TEMPERATURE,
        SAMPLING_MAX_TOKENS,
    )
    wandb.log({
        "step": num_rollout_steps,
        **{"final_" + k: v for k, v in final_metrics.items()}
    })
    print(f"Final validation reward: {final_metrics['val_reward']:.4f}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
