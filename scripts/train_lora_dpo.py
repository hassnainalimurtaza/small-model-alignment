"""Task 5: "Idiot Sandwich" persona tuning with TRL DPOTrainer + PEFT/LoRA (PA2 Sec. 11).

This is the ONLY task where TRL/PEFT are allowed (PA2 Sec. 3, Library Boundary).

  CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo

LoRA choices (justify these in the report, PA2 Sec. 11):
  * r=16, alpha=32, dropout=0.05: a style shift is a low-rank change; r=16 is
    enough capacity for 200 examples while keeping ~1% of params trainable.
  * target = all attention + MLP projections: style lives in both the attention
    (what to attend to) and MLP (what words to produce) blocks.
  * lr 5e-5: LoRA adapters start at zero (B=0) and need a larger lr than the
    1e-6 used for full-parameter DPO; 3 epochs because the dataset is tiny.
  * The reference model is the same network with the adapter disabled, so no
    second copy of the model is needed (TRL does this when peft_config is given).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    ap.add_argument("--output_dir", default="outputs/lora_dpo")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# =============================================================================
# Conversational format expected by TRL: lists of {"role", "content"} messages
# =============================================================================
def to_conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def filter_kwargs(cls, kwargs):
    """Keep only arguments this installed TRL version understands (TRL renames args often)."""
    params = inspect.signature(cls.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return {k: v for k, v in kwargs.items() if k in params}


def main():
    args = parse_args()
    with open(args.data_path, encoding="utf-8") as f:
        rows = json.load(f)
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    train_ds = Dataset.from_list([to_conversational(r) for r in rows[n_val:]])
    val_ds = Dataset.from_list([to_conversational(r) for r in rows[:n_val]])
    print(f"train={len(train_ds)} val={len(val_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # FP32 weights + fp16=True autocast: required for the fp16 GradScaler on T4.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, attn_implementation="sdpa")

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = DPOConfig(**filter_kwargs(DPOConfig, dict(
        output_dir=args.output_dir, beta=args.beta, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=5, eval_strategy="epoch", save_strategy="no", fp16=torch.cuda.is_available(),
        max_length=512, max_prompt_length=256, report_to="none", seed=args.seed,
    )))

    trainer_kwargs = dict(model=model, ref_model=None, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, peft_config=lora)
    tparams = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in tparams else "tokenizer"] = tok
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters()
    trainer.train()

    # ---- Save adapter, a merged model for generation, and the logs ----------
    trainer.model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(os.path.join(args.output_dir, "merged"), safe_serialization=True)
    tok.save_pretrained(os.path.join(args.output_dir, "merged"))
    with open(os.path.join(args.output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    with open(os.path.join(args.output_dir, "lora_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("saved adapter + merged model to", args.output_dir)


if __name__ == "__main__":
    main()
