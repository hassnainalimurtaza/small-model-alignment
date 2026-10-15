"""Tasks 2 + 6: batched greedy generation on a fixed eval set (PA2 Sec. 8 and 12).

  python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \
      --eval_set eval_sets/alpaca_eval_300_seed42.json --output outputs/generations/base_alpaca.json

Run two of these at once (CUDA_VISIBLE_DEVICES=0 and =1) to use both T4s.
The SAME decoding settings must be used for every model in a comparison (PA2 Sec. 12 tip).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from alignment.data import format_prompt, read_json, write_json  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--generator", required=True, help="name stored in the output JSON")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt_format", choices=["chatml", "chat_template"], default="chatml",
                    help="chatml = our SFT/DPO template; chat_template = tokenizer's own (SmolLM2 persona task)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="sanity check on a few prompts (PA2 Sec. 12)")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = read_json(args.eval_set)[: args.limit] if args.limit else read_json(args.eval_set)

    # ---- Resume: skip if this model's file is already complete (PA2 Sec. 5.5) ----
    if os.path.exists(args.output) and not args.limit:
        done = read_json(args.output)
        if len(done) == len(rows):
            print(f"{args.output} already complete, skipping")
            return

    # ---- Model in fp16 for inference; left padding for batched decoding ------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    model.config.use_cache = True  # training checkpoints are saved with use_cache=False; the KV cache makes decoding fast

    # Stop on either <|im_end|> (end of ChatML turn) or the tokenizer EOS.
    stop_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id:
        stop_ids.add(im_end)

    def build(instr: str) -> str:
        if args.prompt_format == "chat_template":
            return tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)
        return format_prompt(instr)

    prompts = [build(r["instruction"]) for r in rows]
    # Longest prompts first: batches have similar lengths -> less padding, and an OOM shows up immediately.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs = [None] * len(prompts)

    for s in range(0, len(order), args.batch_size):
        idxs = order[s : s + args.batch_size]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                do_sample=False,  # greedy decoding (PA2 Sec. 8)
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=list(stop_ids),
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        for i, t in zip(idxs, new_tokens):
            outputs[i] = tok.decode(t, skip_special_tokens=True).strip()
        print(f"{min(s + args.batch_size, len(order))}/{len(order)}", flush=True)

    # ---- Save immediately after this model finishes (PA2 Sec. 5.5) -----------
    result = []
    for r, o in zip(rows, outputs):
        item = {"id": r["id"], "instruction": r["instruction"], "output": o,
                "generator": args.generator, "dataset": r.get("dataset", "")}
        if "category" in r:
            item["category"] = r["category"]
        result.append(item)
    write_json(args.output, result)
    write_json(args.output.replace(".json", "_decoding.json"), vars(args))
    print(f"saved {len(result)} generations -> {args.output}")
    if args.limit:
        for item in result:
            print("=" * 80, "\nPROMPT:", item["instruction"], "\n---\nOUTPUT:", item["output"][:800])


if __name__ == "__main__":
    main()
