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

def compute_entropy_old(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs * log_probs.exp()).sum(dim=-1)

def compute_entropy(logits: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    # Same result as compute_entropy_old, chunked over the sequence dimension.
    # log_softmax reduces over the vocab dimension only, so each position is
    # computed exactly as it would be in one shot, but we never hold more than
    # `chunk_size` positions worth of vocab-sized intermediates at once. With a
    # ~100k vocab those intermediates are what blow up GPU memory.
    entropies = []
    for start in range(0, logits.shape[-2], chunk_size):
        chunk_log_probs = torch.log_softmax(logits[..., start:start + chunk_size, :], dim=-1)
        entropies.append(-(chunk_log_probs * chunk_log_probs.exp()).sum(dim=-1))
    return torch.cat(entropies, dim=-1)

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
        # Entropy is only logged as a metric, never backpropagated through, so
        # keeping it out of the autograd graph saves a vocab-sized tensor.
        with torch.no_grad():
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

    # raw_rewards_or_advantages: (batch_size,) or (batch_size, 1)
    # policy_log_probs: (batch_size, sequence_length)
    # Need to broadcast advantages to match policy_log_probs shape

    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)  # (batch_size, 1)

    # The loss is -A * log_prob for each token
    # We return the negative of the objective so gradient descent performs gradient ascent
    per_token_loss = -raw_rewards_or_advantages * policy_log_probs

    return per_token_loss, {}

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

    # Only support standard on-policy GRPO for now
    if importance_reweighting_method != "none":
        raise NotImplementedError("Only on-policy GRPO is supported")
    if baseline != "mean":
        raise NotImplementedError("Only baseline='mean' is supported")
    if advantage_normalizer != "std":
        raise NotImplementedError("Only advantage_normalizer='std' is supported")
    if loss_normalization != "sequence":
        raise NotImplementedError("Only loss_normalization='sequence' is supported")

    # Tokenize prompts and outputs
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)

    # Compute rewards
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    # Normalize rewards to get advantages
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    # Rewards are computed on CPU from the rollout strings; the loss needs them
    # on the same device as the model's log probs.
    advantages = advantages.to(device)

    # Split into microbatches for gradient accumulation
    batch_size = len(input_ids)
    microbatch_size = batch_size // gradient_accumulation_steps

    # Initialize metrics
    total_loss = 0.0
    all_entropy = []

    optimizer.zero_grad()

    for i in range(gradient_accumulation_steps):
        start_idx = i * microbatch_size
        end_idx = start_idx + microbatch_size

        # Get microbatch
        mb_input_ids = input_ids[start_idx:end_idx]
        mb_labels = labels[start_idx:end_idx]
        mb_response_mask = response_mask[start_idx:end_idx]
        mb_advantages = advantages[start_idx:end_idx]

        # Forward pass with gradients
        model.train()
        result = get_response_log_probs(
            model, mb_input_ids, mb_labels, return_token_entropy=True
        )
        mb_log_probs = result["log_probs"]
        mb_entropy = result["token_entropy"]

        # Compute per-token loss
        per_token_loss, _ = compute_policy_gradient_loss(
            mb_advantages,
            mb_log_probs,
            importance_reweighting_method,
            old_log_probs,
            cliprange,
            mb_response_mask,
        )

        # Aggregate loss across sequence and batch
        # For sequence normalization, we need to scale by number of sequences in microbatch
        loss = aggregate_loss_across_microbatch(
            per_token_loss, mb_response_mask, loss_normalization, normalization_constant
        )

        # Scale loss for gradient accumulation (so the gradient is averaged correctly)
        scaled_loss = loss * (len(mb_input_ids) / batch_size)

        # Backward pass
        scaled_loss.backward()

        # Track metrics
        total_loss += loss.item() * (len(mb_input_ids) / batch_size)
        all_entropy.append(mb_entropy[mb_response_mask].detach())

    # Clip gradients
    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        ).item()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    # Prepare metadata
    metadata = {
        "loss": total_loss,
        "train_reward": reward_metadata["mean_total"],
        "train_format_reward": reward_metadata["mean_format"],
    }

    if grad_norm is not None:
        metadata["grad_norm"] = grad_norm

    if len(all_entropy) > 0:
        metadata["token_entropy"] = torch.cat(all_entropy).mean().item()

    return torch.tensor(total_loss), metadata
