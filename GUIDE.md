# PA2 Solution Guide: SFT + DPO on Qwen2.5-0.5B (Kaggle T4 x2)

References use **PA2 §x** for the assignment manual (`docs/assignment_manual.pdf`) and **RA2 §x / Eq. x / App. x** for the DPO paper (Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv 2305.18290v3).

---

## 1. Concepts you need for the assignment

| # | Concept | What you must understand | Where it is used | Manual reference |
|---|---|---|---|---|
| 1 | **Causal LM pretraining objective** | next-token prediction, `Σ log p(x_t \| x_<t)`; why a base model just continues text | baseline generation (Task 2) | PA2 §2.1, Eq. 1 |
| 2 | **Teacher forcing and logit shifting** | `logits[:, t]` predicts token `t+1`, so logits and labels are shifted by one | `alignment/core.py::logprobs_from_logits` | PA2 §2.2 |
| 3 | **Response-only loss masking** | prompt and padding labels = `-100`; loss is `-log π(y\|x)` over response tokens only | `tokenize_prompt_response`, `collate_lm` | PA2 §2.2 Eq. 2, §4.1, §9.1 item 2 |
| 4 | **Prompt templates and EOS** | one fixed template for SFT, DPO and evaluation; EOS appended *before* truncation | `alignment/data.py::CHATML_TEMPLATE` | PA2 §4.1, §10.3 item 4 |
| 5 | **Padding, attention masks, truncation** | right-pad for training, left-pad for batched generation; truncate to `max_length` | `collate_lm`, `generate.py` | PA2 §4.1, §9.1 item 3 |
| 6 | **SFT loss reductions** | per-example sum mean vs token mean; pick one and document it (this code uses the **token mean**) | `sft_loss_from_logps` | PA2 §4.1 |
| 7 | **Mixed precision (fp16 AMP)** | autocast fp16 + `GradScaler`, FP32 master weights, `unscale_` before clipping | both training scripts | PA2 §9.1 items 5-6 |
| 8 | **Gradient accumulation and clipping** | effective batch = per-device × accumulation × GPUs; clip at 1.0 | both training scripts | PA2 §9.1 items 4, 7; §10.3 item 7 |
| 9 | **LR schedules and optimizers** | warmup + cosine (SFT), constant 1e-6 (DPO); AdamW vs RMSprop | `utils.lr_lambda_factory` | PA2 §9.2, §10.4 |
| 10 | **RLHF pipeline** | SFT → reward model (Bradley-Terry) → PPO with a KL penalty; why it is complex | background for the report | PA2 §2.3; RA2 §3 |
| 11 | **DPO loss** | four log-probs, Δθ − Δref, `-log σ(β·…)` | `dpo_loss_from_logps` | PA2 §2.4, Eq. 3-6 |
| 12 | **Reference model and β** | frozen SFT copy as the anchor, which replaces the explicit KL penalty; β sets the strength | `train_dpo.py` Phase A | PA2 §2.4.1-2.4.2 |
| 13 | **Implicit reward, reward accuracy, reward margin** | `r = β(log π_θ − log π_ref)`; strict `r_w > r_l`; accuracy is exactly 0 at step 0 | `dpo_metrics`, `dpo_val_metrics` | PA2 §10.5, Eq. 7-10 |
| 14 | **Checkpoint selection** | SFT: lowest validation loss. DPO: highest validation reward accuracy. Save only best and final | both training scripts | PA2 §9.1 item 9, §10.1 item 6 |
| 15 | **LoRA / PEFT** | low-rank adapters `W + BA`; rank, alpha, dropout, target modules | `train_lora_dpo.py` | PA2 §11 |
| 16 | **TRL DPOTrainer** | conversational dataset format, `peft_config`, reference = adapter disabled | `train_lora_dpo.py` | PA2 §11 |
| 17 | **Decoding** | greedy vs sampling, repetition penalty / no-repeat n-gram, same settings for every compared model | `generate.py` | PA2 §8, §12 |
| 18 | **LLM-as-judge evaluation** | blinded A/B pairs, randomised order, win/tie/loss, position and length bias | `make_pairs.py`, `judge_pairwise_kaggle.py` | PA2 §12-13 |
| 19 | **Reliable batch jobs** | small batches, sleep, retry, failure logs, resume | `judge_pairwise_kaggle.py` | PA2 §13.2 |
| 20 | **Multi-GPU basics** | DDP (`torchrun`, `no_sync`, `DistributedSampler`) vs placing models on different GPUs | training scripts | PA2 §5.1, §10.1 memory tip |
| 21 | **Alignment tax and over-optimisation** | DPO can reduce helpfulness or verbosity, or over-refuse | report analysis | PA2 §1, §14.6; RA2 §7 |

---

## 2. Which RA2 (DPO paper) concepts you can use in the solution

| RA2 concept | Where in RA2 | How it is used in this solution |
|---|---|---|
| **KL-constrained RLHF objective** `max E[r] − β·KL(π‖π_ref)` | §3, Eq. 3 | Explains in the report *why* a reference model and β exist. DPO optimises the same objective without RL (PA2 §2.4.1 says the reference "replaces the explicit KL penalty"). |
| **Closed-form optimal policy** `π* ∝ π_ref·exp(r/β)` | §4, Eq. 4; App. A.1 | Theory behind the loss. Cite it when you explain where DPO comes from. |
| **Reward reparameterisation** `r = β log(π/π_ref) + β log Z(x)`; Z cancels under Bradley-Terry | §4, Eq. 5-6; App. A.2 | This is why the implicit reward in PA2 Eq. 7-8 is `β(log π_θ − log π_ref)`. Used directly in `dpo_loss_from_logps` and in the reward accuracy/margin metrics. |
| **The DPO loss** | §4, **Eq. 7** | Same as PA2 Eq. 5-6. Implemented in `alignment/core.py::dpo_loss_from_logps`. |
| **Reference implementation** (`pi_logratios − ref_logratios`, `-F.logsigmoid`, `rewards = β(π−ref).detach()`) | **App. B** code | `dpo_loss_from_logps` follows it line for line. It also motivates the **precomputed reference log-probs** in `train_dpo.py`, because RA2's `dpo_loss` takes `ref_logps` as an input tensor. |
| **Hyperparameters**: β = 0.1, RMSprop, lr 1e-6, batch 64, 150-step warmup | **App. B** | PA2 adopts β = 0.1 and lr 1e-6 (§10.3 item 6). This code defaults to **RMSprop** (allowed by PA2 §10.4). `--warmup_steps 150` reproduces RA2's warmup if you want to try it; report it as a deviation (PA2 §10.3 last paragraph). |
| **Gradient analysis**: the weight `σ(r̂(y_l) − r̂(y_w))` is larger when the implicit reward orders the pair wrongly. Without it the model degenerates | §4 "What does the DPO update do?"; App. A.4 | Use it to explain the training curves: loss starts at log 2, margins grow, and easy pairs stop contributing. It also explains why chosen log-probs can *fall* while accuracy rises (logged as `val_policy_chosen_logp`). |
| **π_ref = π_SFT, otherwise Preferred-FT** | §4 "DPO outline" | Justifies starting each DPO run from the SFT best checkpoint and using a frozen copy of it as the reference (PA2 §10.3 items 2-3). The distribution-shift argument explains why HH and UltraFeedback pairs, which were not sampled from *your* SFT model, give a weaker signal. |
| **Theory: every reward class can be represented** | §5.1, Lemma 1-2, Thm 1 | Optional theory paragraph for the report. Not needed in the code. |
| **Instability of actor-critic / PPO** | §5.2 | Explains why the assignment uses DPO instead of PPO (PA2 §2.3). |
| **Reward-KL frontier** | §6.1, Fig. 2 (left) | Analysis idea: plot reward margin against the policy's drift from the reference. β trades one against the other. |
| **Anthropic-HH single-turn dialogue experiment** | §6.2, Fig. 3 | The direct analogue of your HH-RLHF DPO run. RA2 also filters to **single-turn** dialogue, matching PA2 §6.3. |
| **Win-rate evaluation with an LLM judge; robustness to sampling temperature** | §6.2, Fig. 2-3 | Same structure as PA2 Task 7. Use greedy decoding (temperature 0) for the main comparison, as RA2 reports results at temperature 0. |
| **Judge prompt format**: one-sentence comparison, then the verdict; **order randomised** | **App. C.2** | `judge_pairwise_kaggle.py::build_prompt` uses this structure. `make_pairs.py` randomises A/B order as RA2 does. |
| **The judge prefers longer answers; the "concise" prompt agrees more with humans** | **§6.4**, Table 2 | The judge prompt says "do not prefer a response just because it is longer". Notebook 04 reports mean response length per model so you can discuss length bias (see also PA2 ref. 3, Length-Controlled AlpacaEval). |
| **Limitations: over-optimisation, OOD generalisation, judge-prompt sensitivity** | §7 | Material for PA2 §14.6-14.7 (interpretation, limitations). |

---

## 3. How the code is optimised for Kaggle T4 x2

| Optimisation | Why it helps on a T4 | Location |
|---|---|---|
| **fp16 autocast + GradScaler, FP32 master weights** | T4 tensor cores support fp16. bf16 runs emulated and slowly. Required by PA2 §9.1 items 5-6 | train scripts |
| **PyTorch SDPA attention** (`attn_implementation="sdpa"`) | FlashAttention-2 does not support Turing (sm_75). SDPA uses the memory-efficient kernel | all model loads |
| **Data-parallel on both GPUs (`torchrun --nproc_per_node 2`)** | about 2× throughput. Effective batch kept at 32 (2×8×2 for SFT, 1×16×2 for DPO) | `train_sft.py`, `train_dpo.py` |
| **`no_sync()` on accumulation micro-steps** | gradients are all-reduced once per optimiser step, not 8-16 times | train scripts |
| **Response-only logits** ("fast path") | Qwen's vocabulary has 151,936 tokens, so `[B,T,V]` logits are the largest tensor. This code computes logits **only at response positions**. The result is identical, verified by `test_fast_path_matches_full_logits_on_tiny_qwen2` | `core.response_logprobs` |
| **Chunked lm_head when no gradients are needed** | evaluation and reference scoring never materialise N×151,936 logits at once | `core.response_logprobs` |
| **Reference log-probs computed once** | the frozen π_ref is used for one no-grad pass on both GPUs and then freed. The training step runs only the policy, so both GPUs are free for DDP. Results are cached for restarts | `train_dpo.py` Phase A |
| **Chosen + rejected in one forward pass** | one kernel launch sequence instead of two | `train_dpo.py::pair_batch` |
| **Dynamic padding + length-sorted evaluation** | no compute is spent on pad tokens | `collate_lm`, `evaluate`, `score_all_pairs` |
| **Fused AdamW** | fewer kernel launches per step | `train_sft.py` |
| **`expandable_segments:True`** | less CUDA memory fragmentation with variable sequence lengths | notebook setup cell |
| **Two generation jobs at once (one per GPU)** + left padding + length-sorted batches + KV cache | generation throughput | `generate.py`, notebooks 01/04 |
| **Best + final checkpoints only, logs zipped without weights** | stays within Kaggle's ~20 GB `/kaggle/working` limit | `utils.save_checkpoint`, notebooks |
| **TRL notebook pinned to one GPU** | stops HF Trainer from wrapping a 360M model in slow `DataParallel` | notebook 03 |

Rough runtime on T4 x2 (estimates, not measured): SFT ~1 h · each DPO ~1-1.5 h · LoRA ~15 min · generation ~30 min · judging ~30 min.

---

## 4. Running it on Kaggle, step by step

### One-time setup
1. **Kaggle account**: phone-verify it (Settings → Phone verification). Without verification you get no GPU and no internet.
2. **GitHub repo** (PA2 §5.4 step 1, §16 item 1): create a new GitHub repository, e.g. `small-model-alignment`, and push this folder to it:
   ```bash
   cd small-model-alignment
   git init && git add . && git commit -m "SFT + DPO implementation"
   git branch -M main
   git remote add origin https://github.com/<you>/small-model-alignment.git
   git push -u origin main
   ```
   *Alternative:* zip the folder and upload it as a Kaggle Dataset (**Datasets → New Dataset**), then set `REPO_DATASET` in the first cell of each notebook.
3. In each notebook, set `REPO_URL` in **Step 0** to your repo URL.

### Importing a notebook
**Create → New Notebook → File → Import Notebook →** choose a `.ipynb` from `notebooks/`. Then open the right-hand **Settings** panel:
* **Accelerator: GPU T4 x2** (notebooks 01, 02, 04), **GPU T4 x2 or one T4** (03), **None** (05)
* **Internet: On**
* **Persistence: Files only**, so files survive restarts of an interactive session

### Order of execution (PA2 §5.4: separate notebooks so you never re-train while debugging evaluation)

| Notebook | Inputs to attach (**Add Input → Your Work → Notebooks**) | Produces |
|---|---|---|
| `01_setup_baseline_sft` | none | `outputs/sft/{best,final}`, base generations, SFT plot |
| `02_dpo_hh_and_ultrafeedback` | saved version of **01** | `outputs/dpo_hh/*`, `outputs/dpo_uf/*`, DPO plots |
| `03_lora_persona_dpo` | none | `outputs/lora_dpo/*`, persona pairs |
| `04_generate_eval_outputs` | **01** and **02** | generations and `outputs/eval/*_pairs.jsonl` |
| `05_judge_with_kaggle_models` | **03** and **04** | `*_judged.jsonl`, `*_summary.json`, `results_table.csv` |

Notebook 03 is independent, so you can run it while 01 is running. Kaggle allows several sessions at once within your weekly GPU quota (about 30 h).

### For each GPU notebook
1. Run cells interactively up to and including the **smoke test**. It takes a few minutes and catches path, memory or data errors.
2. Then click **Save Version → Save & Run All (Commit)**. The run continues on Kaggle's servers even if you close the browser, for up to 12 h. Interactive sessions can stop when idle.
3. When the version finishes, open it and check the **Output** tab: `outputs/...` should be there. This saved version is what you attach to the next notebook.
4. Every long stage writes `metrics.jsonl` as it goes, and every notebook can be restarted. Finished DPO runs, generations and judgments are skipped on re-run (PA2 §5.5).

### Notebook 05 (judge)
* Create it as a **Kaggle Benchmarks** notebook, or enable benchmark/model access, so `kaggle_benchmarks` can call `google/gemini-3.6-flash` for free (PA2 §13.1).
* **Step 2** prints the available models and sends one test prompt. If the API shape differs from `kbench.llms[name].prompt(text)`, change the 4-line `make_judge()` function in `scripts/judge_pairwise_kaggle.py`. This call is the only part of the code that could not be checked against Kaggle's environment beforehand.
* Re-running the judging cell resumes where it stopped (`--reuse_judgments`).

### Troubleshooting
| Symptom | Fix |
|---|---|
| `assert torch.cuda.device_count() == 2` fails | Settings → Accelerator → **GPU T4 x2** |
| `torchrun` hangs at start | add `os.environ["NCCL_P2P_DISABLE"] = "1"` in Step 0 and re-run |
| CUDA OOM | gradient checkpointing is already on by default; next try `--per_device_batch_size 1` (accumulation is recomputed automatically). Report it as a resource-driven change (PA2 §14.3) |
| HF download errors | Internet must be **On**. Re-run the cell, because downloads resume |
| `find_ckpt` cannot find `outputs/sft/best` | attach the **saved version** of notebook 01 as input (not the draft) |
| TRL argument errors | `train_lora_dpo.py` filters arguments by TRL version. If one still fails, pin `trl==0.19.1` |

---

## 5. Deviations from the guide config (copy into report §14.3)

| Setting | Guide (PA2) | This code | Type | Reason |
|---|---|---|---|---|
| SFT per-device batch × accumulation | 2 × 16 (one GPU) | 2 × 8 × **2 GPUs** | resource | uses both T4s; effective batch is still 32 |
| DPO per-device batch × accumulation | 1 × 32 | 1 × 16 × **2 GPUs** | resource | same effective batch of 32 |
| DPO reference model placement | reference on `cuda:1`, run every step | frozen reference run **once** under `no_grad`, log-probs cached | resource | mathematically identical (π_ref is frozen; RA2 App. B passes `ref_logps` as an input); frees both GPUs for the policy |
| DPO optimizer | RMSprop or AdamW | RMSprop | (choice) | matches RA2 App. B |
| Gradient checkpointing | not mentioned | **on** for SFT and DPO | resource | without it, SFT hit CUDA OOM on a 15 GB T4 (14.36 GiB used at the first optimizer step); costs ~25-30% extra compute |
| DPO prompts > 256 tokens | max prompt length 256 | such pairs are **dropped**, not truncated | data | keeps every instruction intact; the count is printed in the log |
| SFT loss reduction | either | token mean (per micro-batch) | (choice) | documented, as PA2 §4.1 requires |
| Decoding | greedy | greedy + repetition penalty 1.1, 512 new tokens | evaluation | allowed by PA2 §8; identical for all models |
| Prompt template | student choice | ChatML `<\|im_start\|>user … <\|im_end\|>\n<\|im_start\|>assistant\n`, no system prompt, EOS `<\|endoftext\|>` | (choice) | Qwen-native format, used in SFT, DPO and generation |

Report the values from `outputs/*/best/checkpoint_info.json`, not the guide values (PA2 §14.2).
