# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

**PA2 submission, Hussnain Ali Murtaza (25280069).** From-scratch supervised fine-tuning and Direct Preference Optimization (HH-RLHF and UltraFeedback) on `Qwen/Qwen2.5-0.5B`, plus TRL DPOTrainer + LoRA persona tuning on `SmolLM2-360M-Instruct`. Everything ran on Kaggle (2× Tesla T4).

- **Report:** [`25280069_report.pdf`](25280069_report.pdf) (LaTeX source: [`report/main.tex`](report/main.tex))
- **Executed Kaggle notebooks, with outputs:** [`notebooks/kaggle_runs/`](notebooks/kaggle_runs/)
- **Concepts and design notes:** [`GUIDE.md`](GUIDE.md), [`EXPLAINED.md`](EXPLAINED.md)

## Results

| Comparison (judge: `google/gemini-3.6-flash`) | Wins | Ties | Losses | Unknown | Win rate excl. ties |
|---|---|---|---|---|---|
| DPO-HH vs SFT, HH synthetic set (300) | 159 | 27 | 100 | 14 | **61.4%** |
| DPO-UF vs SFT, AlpacaEval-300 seed 42 | 126 | 17 | 144 | 13 | 46.7% |
| LoRA-DPO vs SmolLM2 base, style (20) | 15 | 3 | 0 | 2 | 100% |
| LoRA-DPO vs SmolLM2 base, quality (20) | 11 | 0 | 7 | 2 | 61.1% |

| Run | Best validation metric |
|---|---|
| SFT (843 steps) | response NLL 1.418 → **1.385** |
| DPO-HH (467 steps) | reward accuracy **0.647**, margin 0.28 |
| DPO-UF (380 steps) | reward accuracy **0.691**, margin 0.53 |
| LoRA-DPO (75 steps) | eval loss 0.132, reward accuracy 1.0 |

## Layout

```text
submission_adapter.py        grading interface (PA2 Sec. 4.1): thin wrappers over alignment/core.py
alignment/core.py            batching, response-only log-probs, SFT + DPO losses
alignment/data.py            ChatML template, dataset preparation, fixed eval sets
alignment/utils.py           DDP setup, logging, LR schedule, checkpoints
scripts/prepare_data.py      UltraChat / HH-RLHF / UltraFeedback / AlpacaEval-300-seed42
scripts/train_sft.py         Task 3 (torchrun, fp16 AMP, FP32 weights, gradient checkpointing)
scripts/train_dpo.py         Task 4 (same code for --data hh and --data uf)
scripts/train_lora_dpo.py    Task 5 (TRL DPOTrainer + PEFT LoRA)
scripts/generate.py          Tasks 2 + 6 (greedy, batched)
scripts/make_pairs.py        Task 6 (blinded, randomised A/B pairs)
scripts/judge_pairwise_kaggle.py  Task 7 (resumable kaggle_benchmarks judge)
scripts/plot_metrics.py      training curves
notebooks/01..05_*.ipynb     clean notebooks, in run order
notebooks/kaggle_runs/       the same notebooks as executed on Kaggle, with outputs
outputs/                     metrics, generations, pairs, judgments (model weights not included)
report/                      final report (PDF + LaTeX + figures)
tests/                       public smoke tests + extra edge cases
```

Model checkpoints (~2 GB each) are not in this repository because of GitHub's file-size limit. They are kept as Kaggle notebook outputs.

## Reproduce

Run `notebooks/01` → `05` on Kaggle (GPU T4 x2, Internet on). Notebook 05 must be a Kaggle Benchmarks notebook. Tests:

```bash
PYTHONPATH=. python -m pytest -q tests/
```

## Documented choices
* Prompt template: `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n` + response + `<|endoftext|>`, with no system prompt.
* SFT loss reduction: mean over all unmasked response tokens in the batch.
* Padding: right padding to the longest row (never beyond `max_length`); EOS is appended before truncation.
* DPO: β = 0.1, lr 1e-6 constant, RMSprop, effective batch 32; best checkpoint = highest validation reward accuracy.
