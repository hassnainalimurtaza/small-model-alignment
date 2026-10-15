"""Stable grading interface for the SFT and DPO implementation (PA2 Sec. 4.1).

Each function is a thin wrapper around alignment/core.py. Signatures are unchanged.

Documented choices:
  * Padding: right padding to the longest row in the batch (never beyond max_length).
  * SFT reduction: mean over all unmasked response tokens in the batch.
  * Reference model computations run under torch.no_grad().
"""

from __future__ import annotations

import torch

from alignment import core


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return core.build_sft_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = core.batch_to(core.build_sft_batch(tokenizer, prompts, responses, max_length), core.model_device(model))
    logps, _ = core.response_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    batch = core.batch_to(core.build_sft_batch(tokenizer, prompts, responses, max_length), core.model_device(model))
    logps, n_tokens = core.response_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    return core.sft_loss_from_logps(logps, n_tokens)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return core.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    loss, _, _ = core.dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )
    return loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    batch = core.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    p_dev, r_dev = core.model_device(policy_model), core.model_device(reference_model)

    def score(model, side, device):
        b = core.batch_to(batch[side], device)
        return core.response_logprobs(model, b["input_ids"], b["attention_mask"], b["labels"])[0]

    pc, pr = score(policy_model, "chosen", p_dev), score(policy_model, "rejected", p_dev)
    with torch.no_grad():  # the reference model is frozen (PA2 Sec. 2.4.3 step 5)
        rc = score(reference_model, "chosen", r_dev).to(p_dev)
        rr = score(reference_model, "rejected", r_dev).to(p_dev)
    loss, _, _ = core.dpo_loss_from_logps(pc, pr, rc, rr, beta)
    return loss
