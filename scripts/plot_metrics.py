"""Training plots for the report (PA2 Sec. 14.4).

  python scripts/plot_metrics.py --run outputs/sft --kind sft
  python scripts/plot_metrics.py --run outputs/dpo_hh --kind dpo
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def ema(ys, a=0.9):
    out, m = [], None
    for y in ys:
        m = y if m is None else a * m + (1 - a) * y
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", choices=["sft", "dpo"], required=True)
    args = ap.parse_args()
    rows = load(os.path.join(args.run, "metrics.jsonl"))
    name = os.path.basename(os.path.normpath(args.run))

    if args.kind == "sft":
        fig, ax = plt.subplots(figsize=(6, 4))
        x, y = series(rows, "train_loss")
        ax.plot(x, y, alpha=0.3, label="train (raw)")
        ax.plot(x, ema(y), label="train (EMA 0.9)")
        x, y = series(rows, "val_loss")
        ax.plot(x, y, "o-", label="validation")
        ax.set(xlabel="optimizer step", ylabel="response-token NLL", title=f"{name}: SFT loss")
        ax.legend()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (tk, vk, label) in zip(axes, [("train_loss", "val_loss", "DPO loss"),
                                              ("train_reward_acc", "val_reward_acc", "reward accuracy"),
                                              ("train_reward_margin", "val_reward_margin", "reward margin")]):
            x, y = series(rows, tk)
            ax.plot(x, ema(y), label="train (EMA 0.9)")
            x, y = series(rows, vk)
            ax.plot(x, y, "o-", label="validation")
            ax.set(xlabel="optimizer step", ylabel=label, title=f"{name}: {label}")
            ax.legend()
    fig.tight_layout()
    out = os.path.join(args.run, f"{name}_curves.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
