from typing import Literal
from collections.abc import Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from checkpoint import get_model_and_tokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    seqs = [p + o for p, o in zip(prompt_ids, output_ids)]
    max_len = max(len(s) for s in seqs)

    pad_id = tokenizer.pad_token_id

    batch_size = len(seqs)
    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    is_response = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        seq = p + o
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        is_response[i, len(p): len(seq)] = True
    return {
        "input_ids": padded[:, :-1], 
        "labels": padded[:, 1:], 
        "response_mask": is_response[:, 1:]}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs * log_probs.exp()).sum(dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    
    out_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    out = {"log_probs": out_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    reward, format_reward, answer_reward = 0.0, 0.0, 0.0
    rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(resp, gt)
        rewards.append(result["reward"])
        reward += result["reward"]
        format_reward += result["format_reward"]
        answer_reward += result["answer_reward"]

    metadata = {
            "mean_total": reward / len(rollout_responses), 
            "mean_format": format_reward / len(rollout_responses)}
    return torch.tensor(rewards), metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    group_wise_rewards = raw_rewards.reshape(len(raw_rewards) // group_size, group_size)
    mean_rewards = group_wise_rewards.mean(dim=-1, keepdim=True)
    std_rewards = group_wise_rewards.std(dim=-1, keepdim=True)

    if baseline == "mean":
        group_wise_rewards -= mean_rewards
    if advantage_normalizer == "std":
        advantages = group_wise_rewards / (std_rewards + torch.tensor(advantage_eps))
    elif advantage_normalizer == "mean":
        advantages = group_wise_rewards / mean_rewards
    else:
        advantages = group_wise_rewards
    return advantages.flatten(), {}

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError

    x = raw_rewards_or_advantages
    z = -policy_log_probs * x
    return z, {}

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    mask_f = mask.to(per_token_policy_gradient_loss.dtype)
    counts = mask_f.sum(dim=-1)
    row_sums = torch.where(mask.bool(), 
        per_token_policy_gradient_loss, 
        torch.zeros_like(per_token_policy_gradient_loss)).sum(dim=-1)
    if loss_normalization == "sequence":
        row_means = row_sums / counts.clamp(min=1)
        valid = (counts > 0).to(per_token_policy_gradient_loss.dtype)
        result = (row_means * valid).sum() / valid.sum().clamp(min=1)
    else:
        sum_of_sums = row_sums.sum()
        result = sum_of_sums / torch.tensor(normalization_constant)

    return result

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    return torch.tensor(0), {}
