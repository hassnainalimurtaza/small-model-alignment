"""Download + format all datasets and fixed eval sets (PA2 Sec. 6).

Deterministic (seed 42), so every notebook can regenerate exactly the same
files instead of passing them around.

  python scripts/prepare_data.py --only sft hh uf alpaca persona
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment import data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--eval_dir", default="eval_sets")
    ap.add_argument("--only", nargs="+", default=["sft", "hh", "uf", "alpaca", "persona"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sft", type=int, default=30_000)
    ap.add_argument("--n_pref", type=int, default=15_000)
    ap.add_argument("--n_val", type=int, default=1_000)
    args = ap.parse_args()

    # ---- Training / validation splits --------------------------------------
    jobs = {
        "sft": lambda: D.prepare_sft(args.n_sft, args.n_val, args.seed),
        "hh": lambda: D.prepare_hh(args.n_pref, args.n_val, args.seed),
        "uf": lambda: D.prepare_ultrafeedback(args.n_pref, args.n_val, args.seed),
    }
    for name in [j for j in args.only if j in jobs]:
        splits = jobs[name]()
        for split, rows in splits.items():
            path = os.path.join(args.out_dir, f"{name}_{split}.jsonl")
            D.write_jsonl(path, rows)
            print(f"[{name}] {split}: {len(rows):,} rows -> {path}")

    # ---- Fixed evaluation sets (save exact subsets, PA2 Sec. 6.4) -----------
    if "alpaca" in args.only:
        rows = D.prepare_alpaca_eval(300, args.seed)
        path = os.path.join(args.eval_dir, "alpaca_eval_300_seed42.json")
        D.write_json(path, rows)
        print(f"[alpaca] {len(rows)} prompts -> {path}")
    if "persona" in args.only:
        path = os.path.join(args.eval_dir, "persona_style_eval.json")
        D.write_json(path, D.persona_eval_set())
        print(f"[persona] {len(D.PERSONA_EVAL_PROMPTS)} prompts -> {path}")


if __name__ == "__main__":
    main()
