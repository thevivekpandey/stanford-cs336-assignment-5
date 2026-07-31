from pathlib import Path
from vllm_utils import VLLMServer, wait_for_server
from drgrpo_grader import extract_answer, question_only_reward_fn, r1_zero_reward_fn
import json

SAMPLING_PARAMS = {"temperature": 1.0,
    "max_tokens": 1024,
    "stop": "</answer>",
    "seed": 1,
    "n": 1,
    "include_stop_str_in_output": True
}

# Prompts sent to vLLM per HTTP request. vLLM schedules the whole batch
# concurrently on the GPU, so this is where the speedup comes from.
BATCH_SIZE = 512

LOG_SEPARATOR = "-" * 43

def get_answers(vllm_server, prompts):
    """Generate one completion per prompt, batched. Returns text in prompt order."""
    completions = vllm_server.generate_completions(prompts, SAMPLING_PARAMS, batch_size=BATCH_SIZE)
    assert len(completions) == len(prompts), (
        f"expected {len(prompts)} completions, got {len(completions)}; "
        "SAMPLING_PARAMS['n'] must be 1 for the 1:1 zip below"
    )
    return [completion.text for completion in completions]

def build_prompt(question_template, question):
    return question_template.replace("{question}", question)

def extract_model_answer(response):
    """Display only -- mirrors how the reward fns locate the answer.

    Scoring comes from reward_fn, never from this. r1_zero puts the answer in
    <answer>...</answer>; question_only puts it in \\boxed{...}.
    """
    if "<answer>" in response:
        response = response.split("<answer>")[-1].replace("</answer>", "").strip()
        if not response:
            return None
        if "\\boxed" not in response:
            return response
    return extract_answer(response)

def extract_gold_answer(answer):
    """GSM8K gold answers are a reasoning trace ending in '#### <final answer>'."""
    return answer.split("####")[-1].strip()

def load_question_bank(question_bank):
    """Returns a list of (question, gold_answer) pairs."""
    examples = []
    with open(question_bank) as f:
        for line in f:
            j = json.loads(line.strip())
            examples.append((j['question'], extract_gold_answer(j['answer'])))
    return examples

def prompting_baselines(question_bank, prompt_path, reward_fn, log_path):
    vllm_server = VLLMServer("allenai/OLMo-2-0425-1B", gpu=0)
    vllm_server.start()

    question_template = Path(prompt_path).read_text(encoding="utf-8")
    print(question_template)

    examples = load_question_bank(question_bank)
    print(f"Loaded {len(examples)} questions, generating in batches of {BATCH_SIZE}...")

    prompts = [build_prompt(question_template, question) for question, _ in examples]
    responses = get_answers(vllm_server, prompts)

    format_score = 0
    correctness_score = 0

    log = open(log_path, "w", encoding="utf-8")
    for (question, gold_answer), response in zip(examples, responses):
        rewards = reward_fn(response, gold_answer)
        is_formatted = rewards["format_reward"]
        is_correct = rewards["answer_reward"]

        format_score += is_formatted
        correctness_score += is_correct

        model_answer = extract_model_answer(response)

        print(f"{question[:10]!r} | {'RIGHT' if is_correct else 'WRONG'} | "
              f"gold: {gold_answer} | "
              f"model: {'NONE' if model_answer is None else model_answer} | "
              f"format: {int(is_formatted)}")

        log.write(f"QUESTION\n{question}\nANSWER\n{response}\n{LOG_SEPARATOR}\n")
    log.close()
    print(f"\nWrote full model outputs to {log_path}")

    n_total = len(examples)
    print(f"\nFormat score:      {int(format_score)}/{n_total} = {format_score / n_total:.3f}")
    print(f"Correctness score: {int(correctness_score)}/{n_total} = {correctness_score / n_total:.3f}")

if __name__ == "__main__":
    question_bank = "../data/gsm8k/test.jsonl"
    #prompting_baselines(question_bank, "prompts/question_only.prompt",
    #                    question_only_reward_fn, "question_only_outputs.log")
    #prompting_baselines(question_bank, "prompts/r1_zero.prompt",
    #                    r1_zero_reward_fn, "r1_zero_outputs.log")
    prompting_baselines(question_bank, "prompts/r1_zero_three_shot_gsm8k.prompt",
                        r1_zero_reward_fn, "r1_zero_three_shot_gsm8k_outputs.log")
