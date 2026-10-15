"""Extra checks suggested by PA2 Sec. 4.2: truncation boundaries, unusual
tokenizer configs, mixed lengths, row order, DPO numerics, and that the fast
response-only logit path matches the plain full-logits path on a real HF model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_eos_is_dropped_when_response_is_truncated():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: "], ["hello world"], max_length=6)
    assert batch["input_ids"].shape == (1, 6)
    assert tok.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0, :4].eq(-100).all() and batch["labels"][0, 4:].ne(-100).all()


def test_prompt_longer_than_max_length_gives_no_response_tokens():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["a very long prompt"], ["x"], max_length=5)
    assert batch["labels"].eq(-100).all()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    logps = adapter.compute_response_logprobs(model, tok, ["a very long prompt"], ["x"], max_length=5)
    assert logps.item() == 0.0


def test_tokenizer_without_pad_token_falls_back_to_eos():
    tok = ToyTokenizer()
    # Only the pad ID is removed: ToyTokenizer.encode() scans for the pad *string*, so it must stay a str.
    tok.pad_token_id = None
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["x", "y"], max_length=10)
    assert batch["input_ids"][0, -1].item() == tok.eos_token_id
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts, responses = ["p: ", "q: ", "r: "], ["aaaa", "b", "cc"]
    together = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=16)
    single = torch.stack([adapter.compute_response_logprobs(model, tok, [p], [r], 16)[0] for p, r in zip(prompts, responses)])
    torch.testing.assert_close(together, single)


def test_dpo_loss_matches_closed_form():
    pc, pr = torch.tensor([-1.0, -3.0]), torch.tensor([-2.0, -1.0])
    rc, rr = torch.tensor([-1.5, -2.0]), torch.tensor([-1.5, -2.0])
    beta = 0.1
    expected = -F.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()
    torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), expected)
    _, cr, rj = core.dpo_loss_from_logps(pc, pr, rc, rr, beta)
    m = core.dpo_metrics(cr, rj)
    assert m["reward_acc"] == 0.5 and math.isclose(m["reward_margin"], -0.05, rel_tol=1e-5)  # mean of [0.1, -0.2]


def test_identical_policy_and_reference_give_zero_accuracy_and_log2():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    loss = adapter.compute_dpo_loss(model, model, tok, ["P: "], ["ab"], ["cd"], beta=0.1, max_length=12)
    torch.testing.assert_close(loss.detach(), torch.log(torch.tensor(2.0)))


def test_fast_path_matches_full_logits_on_tiny_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(2, 64, (3, 10))
    labels = ids.clone()
    labels[:, :4] = -100
    labels[2, 7:] = -100
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0
    fast, n_fast = core.response_logprobs(model, ids, mask, labels, fast=True)
    slow, n_slow = core.response_logprobs(model, ids, mask, labels, fast=False)
    torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)
    assert torch.equal(n_fast, n_slow)
