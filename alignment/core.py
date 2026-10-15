"""Core SFT + DPO math shared by the training scripts and submission_adapter.py.

Everything the private grader checks lives here (PA2 Sec. 4.1):
  * building response-masked LM batches        -> build_lm_batch / build_sft_batch / build_dpo_batch
  * summed response-token log-probabilities    -> response_logprobs
  * the SFT loss  L_SFT = -log pi(y|x)          -> sft_loss            (PA2 Eq. 2)
  * the DPO loss  -log sigma(beta * (D_theta - D_ref))  -> dpo_loss_from_logps (PA2 Eq. 5 / RA2 Eq. 7)

No Hugging Face Trainer / TRL is used here (PA2 Sec. 3, "Library Boundary").
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100  # label value ignored by the loss (prompt + padding positions)


# =============================================================================
# 1. Tokenization helpers
# =============================================================================
def pad_token_id(tokenizer) -> int:
    """Pad id with fallbacks for tokenizers that define no pad token."""
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is None:
        pid = getattr(tokenizer, "eos_token_id", None)
    return 0 if pid is None else int(pid)


def encode_many(tokenizer, texts: Sequence[str]) -> List[List[int]]:
    """Tokenize without special tokens; uses the fast batched path when available."""
    texts = list(texts)
    try:
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [list(x) for x in ids]
    except Exception:
        return [list(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def tokenize_prompt_response(
    prompt_ids: List[int], response_ids: List[int], eos_id, max_length: int
) -> Tuple[List[int], List[int]]:
    """Concatenate prompt + response(+EOS), truncate to max_length, build labels.

    EOS is appended BEFORE truncation, so a long response can lose its EOS
    (PA2 Sec. 4.1, expected behaviour bullet 3). Prompt tokens get IGNORE_INDEX
    so only response tokens are trained / scored.
    """
    response_ids = list(response_ids) + ([int(eos_id)] if eos_id is not None else [])
    input_ids = (list(prompt_ids) + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate_lm(
    rows: Sequence[Tuple[List[int], List[int]]], pad_id: int, pad_to: int | None = None
) -> Dict[str, torch.Tensor]:
    """Right-pad (input_ids, labels) rows to a common width.

    Dynamic padding (to the longest row) is used by default: on a T4 this avoids
    wasting compute on pad tokens. Row order is preserved (PA2 Sec. 4.1).
    """
    width = pad_to if pad_to is not None else max(1, max(len(ids) for ids, _ in rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# =============================================================================
# 2. Batch builders (SFT and DPO)
# =============================================================================
def build_lm_batch(tokenizer, prompts, responses, max_length, pad_to=None):
    """Response-masked causal-LM batch: input_ids, attention_mask, labels."""
    assert len(prompts) == len(responses), "prompts and responses must align"
    p_ids = encode_many(tokenizer, prompts)
    r_ids = encode_many(tokenizer, responses)
    eos = getattr(tokenizer, "eos_token_id", None)
    rows = [tokenize_prompt_response(p, r, eos, max_length) for p, r in zip(p_ids, r_ids)]
    return collate_lm(rows, pad_token_id(tokenizer), pad_to=pad_to)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return build_lm_batch(tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Two LM batches sharing the SAME prompt formatting (PA2 Sec. 2.4.3, steps 1-3)."""
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, max_length),
    }


# =============================================================================
# 3. Response-only log-probabilities
# =============================================================================
def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    """Sum of log p(token) over response positions.

    logits[:, t] predicts token t+1, so logits and labels are shifted by one.
    Returns (summed logp per row [B], number of scored tokens per row [B]).
    """
    logits = logits[:, :-1].float()  # fp32 for a numerically stable log-softmax
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe = targets.masked_fill(~mask, 0)
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1), mask.sum(-1)


def _supports_fast_path(model) -> bool:
    """HF decoder-only models expose .model (backbone) and .lm_head."""
    cfg = getattr(model, "config", None)
    return (
        cfg is not None
        and isinstance(getattr(model, "model", None), nn.Module)
        and isinstance(getattr(model, "lm_head", None), nn.Module)
        and getattr(cfg, "final_logit_softcapping", None) is None
    )


def response_logprobs(model, input_ids, attention_mask, labels, fast: bool = True):
    """Summed response-token log-probs per row, plus token counts.

    T4 optimisation ("fast" path): Qwen2.5 has a 151,936-token vocabulary, so the
    full [B, T, V] logits tensor is the largest activation in the network. We run
    the transformer backbone, keep ONLY the hidden states at response positions,
    and apply lm_head to those. Prompt/padding positions never get logits. The
    result is mathematically identical to logprobs_from_logits(model(...).logits).
    Falls back to the plain path for non-HF models (e.g. the toy test LM).
    """
    if fast and _supports_fast_path(model):
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        targets = labels[:, 1:]
        mask = targets.ne(IGNORE_INDEX)
        h = hidden[:, :-1][mask]  # [N_response_tokens, H]
        tgt = targets[mask]
        # Without gradients (reference scoring / validation) we can process the
        # tokens in chunks, so a big eval batch never materialises N x 151,936 logits.
        chunk = max(1, h.size(0)) if torch.is_grad_enabled() else 1024
        parts = []
        for s in range(0, h.size(0), chunk):
            logits = model.lm_head(h[s : s + chunk]).float()  # [n, V] in fp32
            parts.append(logits.gather(-1, tgt[s : s + chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1))
        tok = torch.cat(parts) if parts else h.new_zeros(0, dtype=torch.float32)
        rows = mask.nonzero(as_tuple=True)[0]
        sums = torch.zeros(input_ids.size(0), device=tok.device, dtype=tok.dtype).index_add(0, rows, tok)
        return sums, mask.sum(-1)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logprobs_from_logits(logits, labels)


class ResponseScorer(nn.Module):
    """Thin nn.Module wrapper so DistributedDataParallel sees the fast path as forward().

    (Calling model.model / model.lm_head directly on a DDP-wrapped model would
    bypass DDP's gradient hooks and silently break gradient synchronisation.)
    """

    def __init__(self, model: nn.Module, fast: bool = True):
        super().__init__()
        self.model = model
        self.fast = fast

    def forward(self, input_ids, attention_mask, labels):
        return response_logprobs(self.model, input_ids, attention_mask, labels, fast=self.fast)


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def batch_to(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# =============================================================================
# 4. Losses
# =============================================================================
def sft_loss_from_logps(logps: torch.Tensor, n_tokens: torch.Tensor) -> torch.Tensor:
    """Token-mean reduction: -(sum of response logp) / (number of response tokens).

    This is reduction #2 accepted by PA2 Sec. 4.1 ("mean over all unmasked
    response tokens in the batch"). Document this choice in the report.
    """
    return -logps.sum() / n_tokens.sum().clamp_min(1)


def dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float,
):
    """DPO objective (PA2 Eq. 5; RA2 Eq. 7 and RA2 Appendix B code).

    Returns (scalar mean loss, chosen_rewards, rejected_rewards) where the
    implicit rewards r = beta * (log pi_theta - log pi_ref) are PA2 Eq. 7-8.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps  # Delta_theta (PA2 Eq. 3)
    ref_logratios = reference_chosen_logps - reference_rejected_logps  # Delta_ref (PA2 Eq. 4)
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def dpo_metrics(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> Dict[str, float]:
    """Reward accuracy (strict r_w > r_l, PA2 Eq. 9) and reward margin (PA2 Eq. 10)."""
    return {
        "reward_acc": (chosen_rewards > rejected_rewards).float().mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    }
