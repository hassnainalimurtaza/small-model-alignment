"""Task 4: from-scratch DPO (PA2 Sec. 10; RA2 Sec. 4 + Appendix B).

Same script for both required runs:
  torchrun --nproc_per_node 2 scripts/train_dpo.py --sft_path <sft/best> --data hh --output_dir outputs/dpo_hh
  torchrun --nproc_per_node 2 scripts/train_dpo.py --sft_path <sft/best> --data uf --output_dir outputs/dpo_uf

T4 x2 design:
  Phase A - a frozen copy of the SFT checkpoint (the reference pi_ref) scores every
            chosen/rejected response ONCE under no_grad, sharded over both GPUs.
            pi_ref never changes, so log pi_ref(y|x) is a constant per example;
            RA2 Appendix B's dpo_loss likewise takes ref_logps as an input tensor.
            The reference model is then freed.
  Phase B - the policy pi_theta (fp32 weights, fp16 autocast) trains data-parallel
            on both GPUs: 1 pair per device x 16 accumulation x 2 GPUs = 32 pairs
            per optimizer step (PA2 Sec. 10.3 item 7).
  Compared with the manual's "policy on cuda:0 / reference on cuda:1" tip (PA2
  Sec. 10.1), this keeps the maths identical and roughly halves wall-clock time.
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

DATA = {
    "hh": ("data/processed/hh_train.jsonl", "data/processed/hh_val.jsonl"),
    "uf": ("data/processed/uf_train.jsonl", "data/processed/uf_val.jsonl"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_path", required=True, help="SFT best checkpoint = initial policy")
    ap.add_argument("--reference_path", default=None, help="frozen reference (default: same as --sft_path)")
    ap.add_argument("--data", choices=list(DATA), required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    ap.add_argument("--warmup_steps", type=int, default=0, help="RA2 App. B used 150; PA2 default is none")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--per_device_batch_size", type=int, default=1, help="pairs per device")
    ap.add_argument("--effective_batch_size", type=int, default=32)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=50, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_batch_size", type=int, default=8, help="pairs per eval forward")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_val_examples", type=int, default=None)
    # ON by default: without it, fp32 weights + grads + AdamW (~8 GB) plus activations and
    # 152k-wide logits overflow a 15 GB T4 (observed OOM on Kaggle). Disable with --no_gradient_checkpointing.
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


# =============================================================================
# 1. Tokenize pairs: same prompt formatting for chosen and rejected
#    (PA2 Sec. 2.4.3 steps 1-3). Prompts longer than max_prompt_length (256)
#    are dropped rather than cut, so no example loses its instruction.
# =============================================================================
def tokenize_pairs(tokenizer, rows, max_length, max_prompt_length):
    p = core.encode_many(tokenizer, [format_prompt(r["instruction"]) for r in rows])
    c = core.encode_many(tokenizer, [r["chosen"] for r in rows])
    r_ = core.encode_many(tokenizer, [r["rejected"] for r in rows])
    eos, pairs, dropped = tokenizer.eos_token_id, [], 0
    for pi, ci, ri in zip(p, c, r_):
        if len(pi) > max_prompt_length:
            dropped += 1
            continue
        pairs.append((core.tokenize_prompt_response(pi, ci, eos, max_length),
                      core.tokenize_prompt_response(pi, ri, eos, max_length)))
    return pairs, dropped


def pair_batch(pairs, idxs, pad_id):
    """Chosen rows then rejected rows in ONE padded batch -> one forward pass per micro-step."""
    rows = [pairs[i][0] for i in idxs] + [pairs[i][1] for i in idxs]
    return core.collate_lm(rows, pad_id)


def collate_indices(idxs):
    return list(idxs)


# =============================================================================
# 2. Score every pair (used for the reference pass AND for policy validation,
#    so both go through identical batching / numerics)
# =============================================================================
@torch.no_grad()
def score_all_pairs(scorer, pairs, pad_id, device, batch_size, rank, world):
    """Return a [N, 2] tensor of (chosen, rejected) summed response log-probs on every rank."""
    scorer.eval()
    n = len(pairs)
    out = torch.zeros(n, 2, device=device, dtype=torch.float32)
    mine = sorted(range(rank, n, world), key=lambda i: len(pairs[i][0][0]) + len(pairs[i][1][0]))
    for s in range(0, len(mine), batch_size):
        idxs = mine[s : s + batch_size]
        b = core.batch_to(pair_batch(pairs, idxs, pad_id), device)
        with torch.autocast("cuda", dtype=torch.float16):
            logps, _ = scorer(b["input_ids"], b["attention_mask"], b["labels"])
        k = len(idxs)
        out[idxs, 0], out[idxs, 1] = logps[:k].float(), logps[k:].float()
    return utils.all_reduce_sum(out)  # every index is filled by exactly one rank


def load_model(path, device):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    m.config.use_cache = False
    return m


def reference_logps(args, pairs, split, pad_id, device, rank, world):
    """Phase A: frozen reference scores, cached to disk so a restart can skip it."""
    cache = os.path.join(args.output_dir, f"ref_logps_{split}.pt")
    if os.path.exists(cache):
        t = torch.load(cache, map_location=device)
        if t.shape[0] == len(pairs):
            return t
    ref = load_model(args.reference_path, device)
    ref.requires_grad_(False)  # frozen: no gradients, no optimizer state
    t = score_all_pairs(core.ResponseScorer(ref), pairs, pad_id, device, args.eval_batch_size, rank, world)
    del ref
    torch.cuda.empty_cache()
    if rank == 0:
        torch.save(t.cpu(), cache)
    return t


def dpo_val_metrics(policy_logps, ref_logps, beta):
    loss, cr, rr = core.dpo_loss_from_logps(policy_logps[:, 0], policy_logps[:, 1], ref_logps[:, 0], ref_logps[:, 1], beta)
    return {
        "val_loss": loss.item(),
        **{f"val_{k}": v for k, v in core.dpo_metrics(cr, rr).items()},
        "val_chosen_reward": cr.mean().item(),
        "val_rejected_reward": rr.mean().item(),
        "val_policy_chosen_logp": policy_logps[:, 0].mean().item(),
    }


@record  # torchrun then shows the real Python traceback of a crashed rank
def main():
    args = parse_args()
    args.reference_path = args.reference_path or args.sft_path
    rank, world, local_rank, device = utils.setup_distributed()
    utils.set_seed(args.seed + rank)
    is_main = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)
    logger = utils.MetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"), enabled=is_main)
    accum = max(1, args.effective_batch_size // (args.per_device_batch_size * world))

    # ---- Data ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = core.pad_token_id(tokenizer)
    train_path, val_path = DATA[args.data]
    train_raw, val_raw = read_jsonl(train_path), read_jsonl(val_path)
    train_raw = train_raw[: args.max_train_examples] if args.max_train_examples else train_raw
    val_raw = val_raw[: args.max_val_examples] if args.max_val_examples else val_raw
    train_pairs, d1 = tokenize_pairs(tokenizer, train_raw, args.max_length, args.max_prompt_length)
    val_pairs, d2 = tokenize_pairs(tokenizer, val_raw, args.max_length, args.max_prompt_length)
    if is_main:
        print(f"pairs train={len(train_pairs)} (dropped {d1} long prompts) val={len(val_pairs)} (dropped {d2})", flush=True)

    # ---- Phase A: frozen reference log-probs (no_grad) -----------------------
    ref_train = reference_logps(args, train_pairs, "train", pad_id, device, rank, world)
    ref_val = reference_logps(args, val_pairs, "val", pad_id, device, rank, world)
    utils.barrier()

    # ---- Phase B: trainable policy, initialised from the SFT checkpoint ------
    policy = load_model(args.sft_path, device)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = core.ResponseScorer(policy)
    ddp = DDP(scorer, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else None
    fwd = ddp if ddp is not None else scorer

    if args.optimizer == "rmsprop":  # RA2 Appendix B: RMSprop, lr 1e-6
        opt = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
    sampler = DistributedSampler(range(len(train_pairs)), num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(range(len(train_pairs)), batch_size=args.per_device_batch_size, sampler=sampler,
                        collate_fn=collate_indices)
    total_steps = math.ceil(len(loader) / accum) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, utils.lr_lambda_factory(args.warmup_steps, total_steps, "constant"))
    scaler = torch.amp.GradScaler("cuda")
    if is_main:
        print(f"world={world} accum={accum} steps={total_steps} optimizer={args.optimizer} lr={args.lr} beta={args.beta}", flush=True)

    def run_eval(step):
        pol = score_all_pairs(scorer, val_pairs, pad_id, device, args.eval_batch_size, rank, world)
        fwd.train()
        m = dpo_val_metrics(pol, ref_val, args.beta)
        logger.log(step=step, **m)
        return m

    # Step 0: policy == reference -> every r_w = r_l = 0, so accuracy is exactly 0 (PA2 Sec. 10.5)
    run_eval(0)

    # ---- Training loop ------------------------------------------------------
    best_acc, step = -1.0, 0
    acc_loss = torch.zeros(3, device=device)  # sum loss, sum correct, sum margin
    acc_n = 0
    fwd.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, idxs in enumerate(loader):
            b = core.batch_to(pair_batch(train_pairs, idxs, pad_id), device)
            k = len(idxs)
            ref = ref_train[idxs]
            boundary = (i + 1) % accum == 0 or (i + 1) == len(loader)
            sync_ctx = ddp.no_sync() if (ddp is not None and not boundary) else nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    logps, _ = fwd(b["input_ids"], b["attention_mask"], b["labels"])
                # loss in fp32: Eq. 5 of PA2 / Eq. 7 of RA2
                loss, cr, rr = core.dpo_loss_from_logps(logps[:k].float(), logps[k:].float(), ref[:, 0], ref[:, 1], args.beta)
                scaler.scale(loss / accum).backward()
            acc_loss += torch.stack([loss.detach() * k, (cr > rr).float().sum(), (cr - rr).sum()])
            acc_n += k
            if not boundary:
                continue

            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0:
                tot = utils.all_reduce_sum(torch.cat([acc_loss, torch.tensor([acc_n], device=device, dtype=acc_loss.dtype)]))
                logger.log(step=step, train_loss=(tot[0] / tot[3]).item(), train_reward_acc=(tot[1] / tot[3]).item(),
                           train_reward_margin=(tot[2] / tot[3]).item(), grad_norm=float(grad_norm),
                           lr=sched.get_last_lr()[0], mem_gb=round(utils.gpu_mem_gb(device), 2))
                acc_loss.zero_()
                acc_n = 0

            if step % args.eval_every == 0 or step == total_steps:
                m = run_eval(step)
                # Checkpoint selection = best validation reward accuracy (PA2 Sec. 10.1 item 6)
                if m["val_reward_acc"] > best_acc:
                    best_acc = m["val_reward_acc"]
                    if is_main:
                        utils.save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "best"),
                                              {"step": step, **m, **vars(args)})
                utils.barrier()

    if is_main:
        utils.save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "final"),
                              {"step": step, "best_val_reward_acc": best_acc, **vars(args)})
        print(f"done. best val reward acc {best_acc:.4f}", flush=True)
    utils.barrier()
    utils.cleanup_distributed()


if __name__ == "__main__":
    main()
