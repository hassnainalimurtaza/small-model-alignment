"""Task 6: build blinded pairwise comparison files (PA2 Sec. 12).

Response order is randomised per pair; the position -> model mapping is kept
only in hidden metadata that is never shown to the judge. (RA2 Appendix C.2 does
the same: "the order of summaries or responses are randomly chosen for every
evaluation".)

  python scripts/make_pairs.py --a outputs/generations/dpo_uf_alpaca.json --a_name dpo_uf \
      --b outputs/generations/sft_alpaca.json --b_name sft --out outputs/eval/dpo_uf_vs_sft_pairs.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment.data import read_json, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a_name", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = {r["id"]: r for r in read_json(args.a)}
    b = {r["id"]: r for r in read_json(args.b)}
    rng = random.Random(args.seed)
    rows = []
    for pid in sorted(a.keys() & b.keys()):
        flip = rng.random() < 0.5
        first, second = (b[pid], a[pid]) if flip else (a[pid], b[pid])
        rows.append({
            "pair_id": pid,
            "instruction": a[pid]["instruction"],
            "category": a[pid].get("category", a[pid].get("dataset", "")),
            "response_a": first["output"],
            "response_b": second["output"],
            # hidden metadata: NOT included in the judge prompt
            "meta": {"model_a": args.b_name if flip else args.a_name,
                     "model_b": args.a_name if flip else args.b_name},
        })
    write_jsonl(args.out, rows)
    n_a_first = sum(r["meta"]["model_a"] == args.a_name for r in rows)
    print(f"{len(rows)} pairs -> {args.out} ({args.a_name} shown as A in {n_a_first})")


if __name__ == "__main__":
    main()
