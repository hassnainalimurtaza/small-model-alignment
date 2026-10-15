"""Training utilities: seeding, 2xT4 DDP setup, logging, LR schedules, checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist


# =============================================================================
# 1. Reproducibility
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. Distributed setup (torchrun --nproc_per_node 2 -> one process per T4)
# =============================================================================
def setup_distributed():
    """Return (rank, world_size, local_rank, device). Works with or without torchrun."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), local_rank, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


# =============================================================================
# 3. Logging: metrics.jsonl is appended after every log/eval step so partial
#    progress survives a Kaggle disconnect (PA2 Sec. 5.5, item 3)
# =============================================================================
class MetricsLogger:
    def __init__(self, path: str, enabled: bool = True):
        self.path, self.enabled, self.t0 = path, enabled, time.time()
        if enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, **kv) -> None:
        if not self.enabled:
            return
        kv["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv) + "\n")
        # .4g keeps small values readable: lr 2e-05 used to print as 0.0000
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()), flush=True)


# =============================================================================
# 4. Learning-rate schedule: linear warmup, then cosine decay or constant
# =============================================================================
def lr_lambda_factory(warmup_steps: int, total_steps: int, schedule: str):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


# =============================================================================
# 5. Parameter groups: no weight decay on biases and norm weights
# =============================================================================
def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# =============================================================================
# 6. Checkpoints: only "best" and "final" (PA2 Sec. 9.1 item 9, Sec. 5.5 item 2)
# =============================================================================
def save_checkpoint(model, tokenizer, out_dir: str, extra: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # FP32 weights are saved as-is (~2 GB), so a DPO run restarts from exactly
    # the SFT weights. save_pretrained overwrites, so disk use stays bounded.
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    if extra:
        with open(os.path.join(out_dir, "checkpoint_info.json"), "w") as f:
            json.dump(extra, f, indent=2)


def gpu_mem_gb(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
