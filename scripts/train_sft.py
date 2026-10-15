"""Task 3: from-scratch supervised fine-tuning of Qwen2.5-0.5B (PA2 Sec. 9).

Launch on Kaggle T4 x2 (data-parallel, one process per GPU):
  torchrun --nproc_per_node 2 scripts/train_sft.py --output_dir outputs/sft

Implements every item of PA2 Sec. 9.1:
  1 prompt formatting          -> alignment.data.format_prompt (ChatML)
  2 response-only loss masking -> labels = -100 on prompt + pad
  3 padding + attention masks  -> alignment.core.collate_lm
  4 gradient accumulation      -> DDP no_sync() on non-boundary micro-steps
  5 fp16 autocast              -> torch.autocast(float16)   (T4 has no bf16 tensor cores)
  6 fp32 trainable weights     -> model loaded in float32 + GradScaler
  7 gradient clipping          -> clip_grad_norm_(1.0) after unscale_
  8 validation loss            -> evaluate()
  9 best + final checkpoints   -> outputs/sft/{best,final}
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import nullcontext
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from alignment import core, utils  # noqa: E402
from alignment.data import format_prompt, read_jsonl  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_path", default="data/processed/sft_train.jsonl")
    ap.add_argument("--val_path", default="data/processed/sft_val.jsonl")
    ap.add_argument("--output_dir", default="outputs/sft")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--effective_batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_examples", type=int, default=None, help="debug: subsample")
    ap.add_argument("--max_val_examples", type=int, default=None)
    # ON by default: without it, fp32 weights + grads + AdamW (~8 GB) plus activations and
    # 152k-wide logits overflow a 15 GB T4 (observed OOM on Kaggle). Disable with --no_gradient_checkpointing.
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


# =============================================================================
# Tokenize every example once, up front (fast batched tokenizer)
# =============================================================================
def tokenize_rows(tokenizer, rows, max_length):
    p_ids = core.encode_many(tokenizer, [format_prompt(r["instruction"]) for r in rows])
    r_ids = core.encode_many(tokenizer, [r["response"] for r in rows])
    out = []
    for p, r in zip(p_ids, r_ids):
        ids, labels = core.tokenize_prompt_response(p, r, tokenizer.eos_token_id, max_length)
        if any(l != core.IGNORE_INDEX for l in labels[1:]):  # drop rows whose response was fully truncated
            out.append((ids, labels))
    return out


# =============================================================================
# Validation: token-mean response NLL, summed across both GPUs
# =============================================================================
@torch.no_grad()
def evaluate(scorer, val_rows, pad_id, device, batch_size, rank, world):
    scorer.eval()
    shard = sorted(val_rows[rank::world], key=lambda x: len(x[0]))  # length-sorted -> little padding
    tot = torch.zeros(2, device=device, dtype=torch.float64)
    for i in range(0, len(shard), batch_size):
        b = core.batch_to(core.collate_lm(shard[i : i + batch_size], pad_id), device)
        with torch.autocast("cuda", dtype=torch.float16):
            logps, ntok = scorer(b["input_ids"], b["attention_mask"], b["labels"])
        tot[0] += -logps.sum().double()
        tot[1] += ntok.sum().double()
    utils.all_reduce_sum(tot)
    scorer.train()
    return (tot[0] / tot[1].clamp_min(1)).item()


@record  # torchrun then shows the real Python traceback of a crashed rank
def main():
    args = parse_args()
    rank, world, local_rank, device = utils.setup_distributed()
    utils.set_seed(args.seed + rank)
    is_main = rank == 0
    logger = utils.MetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"), enabled=is_main)

    # ---- Gradient accumulation so global batch = 32 -------------------------
    # Guide (single GPU): 2 x 16 = 32. On 2 GPUs: 2 x 8 x 2 = 32 (same effective batch).
    accum = max(1, args.effective_batch_size // (args.per_device_batch_size * world))

    # ---- Tokenizer + data ---------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = core.pad_token_id(tokenizer)
    train_raw, val_raw = read_jsonl(args.train_path), read_jsonl(args.val_path)
    train_raw = train_raw[: args.max_train_examples] if args.max_train_examples else train_raw
    val_raw = val_raw[: args.max_val_examples] if args.max_val_examples else val_raw
    train_rows = tokenize_rows(tokenizer, train_raw, args.max_length)
    val_rows = tokenize_rows(tokenizer, val_raw, args.max_length)

    sampler = DistributedSampler(train_rows, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        train_rows,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        collate_fn=partial(core.collate_lm, pad_id=pad_id),
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    total_steps = math.ceil(len(loader) / accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    # ---- Model: FP32 master weights + SDPA attention ------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = core.ResponseScorer(model)
    ddp = DDP(scorer, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else None
    fwd = ddp if ddp is not None else scorer

    # ---- Optimizer / schedule / AMP scaler ----------------------------------
    opt = torch.optim.AdamW(
        utils.param_groups(model, args.weight_decay), lr=args.lr, fused=torch.cuda.is_available()
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, utils.lr_lambda_factory(warmup_steps, total_steps, "cosine"))
    scaler = torch.amp.GradScaler("cuda")

    if is_main:
        print(f"world={world} accum={accum} micro/rank={len(loader)} steps={total_steps} "
              f"warmup={warmup_steps} train={len(train_rows)} val={len(val_rows)}", flush=True)
        logger.log(step=0, val_loss=evaluate(scorer, val_rows, pad_id, device, args.eval_batch_size, rank, world))
    else:
        evaluate(scorer, val_rows, pad_id, device, args.eval_batch_size, rank, world)

    # ---- Training loop ------------------------------------------------------
    best_val, step = float("inf"), 0
    run_loss = torch.zeros((), device=device)
    run_n = 0
    fwd.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            batch = core.batch_to(batch, device)
            boundary = (i + 1) % accum == 0 or (i + 1) == len(loader)
            # Skip the gradient all-reduce on non-boundary micro-steps (saves PCIe traffic)
            sync_ctx = ddp.no_sync() if (ddp is not None and not boundary) else nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    logps, ntok = fwd(batch["input_ids"], batch["attention_mask"], batch["labels"])
                    loss = core.sft_loss_from_logps(logps, ntok)
                scaler.scale(loss / accum).backward()
            run_loss += loss.detach()
            run_n += 1
            if not boundary:
                continue

            # ---- Optimizer step: unscale -> clip -> step -> update scaler ----
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0:
                avg = utils.all_reduce_sum(run_loss.clone()) / (run_n * world)
                logger.log(step=step, train_loss=avg.item(), lr=sched.get_last_lr()[0],
                           grad_norm=float(grad_norm), mem_gb=round(utils.gpu_mem_gb(device), 2))
                run_loss.zero_()
                run_n = 0

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate(scorer, val_rows, pad_id, device, args.eval_batch_size, rank, world)
                logger.log(step=step, val_loss=val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    if is_main:
                        utils.save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best"),
                                              {"step": step, "val_loss": val_loss, **vars(args)})
                utils.barrier()

    # ---- Final checkpoint ---------------------------------------------------
    if is_main:
        utils.save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"),
                              {"step": step, "best_val_loss": best_val, **vars(args)})
        print(f"done. best val loss {best_val:.4f}", flush=True)
    utils.barrier()
    utils.cleanup_distributed()


if __name__ == "__main__":
    main()
