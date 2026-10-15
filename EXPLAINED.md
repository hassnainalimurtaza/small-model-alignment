# PA2 Explained From Zero

This document assumes **no prior knowledge**. Each part does three things:

1. **Concept**: what the idea is, in plain words, with a small worked example.
2. **Source**: where PA2 (assignment manual) and RA2 (the DPO paper) talk about it.
3. **Code**: the exact file and function in this solution that implements it.

Notation for references:
* **PA2 §9.1** = section 9.1 of `docs/assignment_manual.pdf`
* **RA2 §4, Eq. 7** = section 4, equation 7 of the DPO paper (Rafailov et al., 2023)

Read Parts 1 → 9 in order. Each one builds on the one before.

---

## Table of contents
1. [Language models from scratch](#part-1)
2. [How a model learns: loss, gradients, optimizers](#part-2)
3. [Supervised Fine-Tuning (SFT): Task 3](#part-3)
4. [RLHF: the method DPO replaces](#part-4)
5. [DPO: the core of the assignment (Task 4)](#part-5)
6. [Making it fit and run fast on Kaggle T4 GPUs](#part-6)
7. [LoRA + TRL persona tuning: Task 5](#part-7)
8. [Generating answers: Tasks 2 and 6](#part-8)
9. [Evaluating with an LLM judge: Task 7](#part-9)
10. [The whole pipeline on one page, plus a glossary](#part-10)

---

<a id="part-1"></a>
## Part 1: Language models from scratch

### 1.1 Tokens
A computer cannot read text directly, so text is split into small pieces called **tokens** and each token gets an integer ID. A **tokenizer** does this.

```
"I love pizza"  →  ["I", " love", " pizza"]  →  [40, 3047, 23317]
```

Qwen2.5's tokenizer knows **151,936** different tokens. That number is its **vocabulary size**, and it matters a lot later for memory (Part 6).

Some tokens are **special**. They are not normal words but signals:
* `<|endoftext|>` is the **EOS** (end-of-sequence) token. It means "I'm finished talking."
* `<|im_start|>` and `<|im_end|>` mark the start and end of a chat turn.

### 1.2 What the model actually outputs
A language model (LM) does one job: **given the tokens so far, predict the next token.**

For every position it outputs 151,936 numbers called **logits**, one score per vocabulary token. A function called **softmax** turns these scores into probabilities that sum to 1:

```
context: "I love"
logits:   " pizza" = 5.1   " you" = 4.8   " the" = 1.2   ...
softmax → P(" pizza") = 0.41   P(" you") = 0.30   P(" the") = 0.01 ...
```

The model is written **π_θ** ("pi theta"). **θ** stands for all its learnable numbers (weights). Qwen2.5-0.5B has about 494 million of them. In RL language the model is called a **policy**, which is why PA2 and RA2 write π.

### 1.3 Probability of a whole answer, and why we use log
The probability that the model writes an entire response `y = (y₁, y₂, …, y_T)` after a prompt `x` is the product of the per-token probabilities:

```
π(y | x) = P(y₁|x) × P(y₂|x,y₁) × P(y₃|x,y₁,y₂) × ...
```

Multiplying 200 numbers like 0.3 gives about 10⁻¹⁰⁵, which is too small for a computer to store. So we take the **logarithm**. A log turns a product into a **sum**:

```
log π(y | x) = log P(y₁|…) + log P(y₂|…) + ... + log P(y_T|…)
```

Each term is negative (the log of a number below 1), and the sum is a manageable number like −85.3. This **summed response log-probability** is the most important quantity in the whole assignment. SFT, DPO and every metric are built from it.

> **PA2 §2.4.2** table: the four quantities `log π_θ(y_w|x)`, `log π_θ(y_l|x)`, `log π_ref(y_w|x)`, `log π_ref(y_l|x)` are exactly such sums.
> **PA2 §4.1**: `compute_response_logprobs` must "return one summed response-token log-probability per input row."
> **Code:** `alignment/core.py` → `logprobs_from_logits()` and `response_logprobs()`.

### 1.4 The one-position shift (teacher forcing)
We feed the model the **whole** sequence at once. The logits at position *t* are the prediction for the token at position *t+1*:

```
input tokens:   [ A ,  B ,  C ,  D ]
logits at pos:    0    1    2    3
predicts:         B    C    D   (nothing, past the end)
```

So to score token B we look at logits[0], for C at logits[1], and so on. In code this is the **shift**:

```python
logits = logits[:, :-1]   # drop the last position (it predicts beyond the end)
targets = labels[:, 1:]   # drop the first token (nothing predicts it)
```

Feeding the true previous tokens, rather than the model's own guesses, is called **teacher forcing** (PA2 §2.2).

### 1.5 Pretraining vs. post-training
* **Pretraining** (PA2 §2.1, Eq. 1): train on trillions of tokens of internet text to maximise `Σ log p(x_t | x_<t)`. The result is a **base model**. It knows language and facts, but it only *continues text*. Ask it "What is 2+2?" and it may reply "What is 3+3? What is 4+4?…".
* **Post-training**: teach the base model to behave like an assistant. This assignment covers post-training only. You start from the pretrained `Qwen/Qwen2.5-0.5B` (PA2 §2.1, §6.1).

The three post-training stages in PA2 are **SFT** (Part 3), **RLHF** (Part 4, background only) and **DPO** (Part 5).

---

<a id="part-2"></a>
## Part 2: How a model learns: loss, gradients, optimizers

### 2.1 Loss
A **loss** is a single number that measures how bad the model is. Lower is better. Training means changing θ so the loss goes down.

The standard LM loss is the **negative log-likelihood (NLL)**: `−log π(correct answer)`. If the model assigns probability 1.0 to the correct token, the loss is −log 1 = 0. If it assigns 0.01, the loss is −log 0.01 = 4.6.

### 2.2 Gradients and backpropagation
The **gradient** tells you, for each of the 494M weights, "if I nudge this weight up a little, how much does the loss change?" PyTorch computes all of them automatically when you call `loss.backward()`. This is **backpropagation**.

### 2.3 Gradient descent, learning rate, optimizer
Update rule: `θ ← θ − learning_rate × gradient`. Step downhill.
* **Learning rate (lr)** = step size. Too big and training blows up; too small and nothing changes. The assignment uses 2×10⁻⁵ for SFT (PA2 §9.2) and a tiny 1×10⁻⁶ for DPO (PA2 §10.3 item 6, RA2 App. B).
* **Optimizer** = a smarter update rule:
  * **AdamW** keeps two running averages per weight (momentum and gradient size). It is good and robust, but it needs 2 extra copies of the model in memory. "W" means *weight decay*: weights are gently pulled toward zero, which acts as a regulariser (PA2 §9.2: 0.1).
  * **RMSprop** keeps one running average per weight, so it needs less memory. RA2 App. B used RMSprop for DPO, and PA2 §10.4 allows "RMSprop or AdamW".

### 2.4 Batch, epoch, step
* **Batch**: several examples processed together; their losses are averaged.
* **Epoch**: one full pass over the dataset. The assignment uses 1 epoch (PA2 §9.2, §10.4).
* **Optimizer step**: one weight update.

### 2.5 Gradient accumulation
A batch of 32 sequences of 512 tokens does not fit in a T4's memory at once. **Gradient accumulation** runs 16 small batches of 2, *adds up* their gradients, and only then does one optimizer step. The update equals a batch of 2 × 16 = 32. This is called the **effective batch size** (PA2 §9.2: per-device 2, accumulation 16, effective 32; PA2 §10.3 item 7 for DPO).

### 2.6 Gradient clipping
Sometimes one bad batch produces a huge gradient that would wreck the model. **Clipping** caps the overall gradient size (its *norm*) at 1.0 (PA2 §9.1 item 7, §9.2).

### 2.7 Learning-rate schedule
* **Warmup**: start with a tiny lr and increase it over the first 3% of steps. Early updates are noisy, so this avoids an early shock (PA2 §9.2).
* **Cosine decay**: after warmup, lower the lr smoothly following a cosine curve down to 0.
* **Constant**: DPO uses a fixed lr (PA2 §10.4: "none by default for DPO").

> **Code:** `alignment/utils.py` → `lr_lambda_factory()` (warmup + cosine/constant) and `param_groups()` (no weight decay on biases and norm layers, a standard practice).

### 2.8 Training vs. validation
You hold out data the model never trains on (the **validation set**, 1,000 examples, PA2 §6.2). Checking the loss there tells you whether the model is learning *general* behaviour or just memorising. When train loss keeps falling while validation loss rises, that is **overfitting**.

**Checkpoint** = a saved copy of the weights. PA2 asks for **best** (by a validation metric) and **final** (end of training) only (PA2 §9.1 item 9, §5.5 item 2).

---

<a id="part-3"></a>
## Part 3: Supervised Fine-Tuning (SFT): Task 3

### 3.1 Concept
Show the base model thousands of examples of `instruction → good response`. Train it to make the good response likely. The loss is:

> **PA2 §2.2, Eq. 2:** `L_SFT(θ) = −log π_θ(y | x)`
> **RA2 §3 "SFT":** RLHF "typically begins by fine-tuning a pre-trained LM with supervised learning on high-quality data… to obtain a model π^SFT."

After SFT the model follows the question/answer format. Task 3 uses **30,000** examples from UltraChat (PA2 §6.2).

### 3.2 The prompt template
The model must always see the same format, so it learns *where the user's message ends and its own answer begins*. This solution uses **ChatML**, the format Qwen was designed for:

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|endoftext|>
```

The first three lines are the **prompt** `x`. The last line, ending with EOS, is the **response** `y`. The same template is used in SFT, in DPO and at generation time. PA2 §10.3 item 4 requires this: "use it consistently for SFT, DPO, and evaluation. Document the exact template."

**Why EOS matters:** the model learns to *emit* `<|endoftext|>` when finished. Without it the model never learns to stop and rambles until the token limit. PA2 §4.1: "Append an EOS token to each response *before* applying the sequence-length truncation… The EOS token may therefore be removed if the response is truncated."

> **Code:** `alignment/data.py` → `CHATML_TEMPLATE`, `format_prompt()`.

### 3.3 Response-only loss masking (the most-tested detail)
We do **not** want to train the model to produce the *user's question*. Its job is to answer. So prompt tokens are excluded from the loss. The trick: build a `labels` array that copies the tokens but puts **−100** wherever we don't want a loss. PyTorch's convention is that −100 means "ignore."

Worked example (one token per word for clarity, max_length = 10):

```
position:        0      1      2       3        4       5      6      7      8     9
tokens:        [user] [What] [is]  [2+2?] [assist] [It's]  [4]  [EOS]  [pad] [pad]
labels:        -100   -100   -100   -100   -100    It's    4    EOS   -100  -100
attention:       1      1      1      1      1       1      1     1      0     0
                \_________ prompt (masked) ________/ \__ response __/ \padding/
```

* **labels**: −100 on the prompt and on padding; real token IDs on the response, including EOS.
* **attention_mask**: 1 = real token, 0 = padding, so the model ignores padding when computing attention.
* **Padding**: sequences in a batch have different lengths, but tensors must be rectangular, so short rows are filled with a pad token on the right.
* **Truncation**: anything past `max_length` (512) is cut off.

> **PA2 §4.1** (expected behaviour): "Pad shorter examples and truncate longer examples to max_length. Mask prompt and padding tokens with −100… Compute log probabilities over response tokens only… Preserve the input row order."
> **PA2 §9.1** items 1-3: prompt formatting, response-only masking, padding and attention masks.
> **Code:** `alignment/core.py` → `tokenize_prompt_response()` (concatenate, add EOS, truncate, build labels) and `collate_lm()` (pad, build the attention mask).

**Implementation detail:** the prompt and the response are tokenized **separately** and then concatenated. That way we know *exactly* which positions belong to the prompt. Tokenizing the joined string could merge tokens across the boundary.

### 3.4 Computing the SFT loss
1. Run the model → logits.
2. Shift by one (§1.4).
3. `log_softmax` turns logits into log-probabilities; `gather` picks the log-prob of the actual next token.
4. Multiply by the mask (response = 1, prompt/pad = 0) and sum → `log π(y|x)` per row.
5. Loss = −(sum of response log-probs) / (number of response tokens).

PA2 §4.1 allows two **reductions**: (a) the mean over examples of the per-example *sum*, or (b) the mean over all response *tokens*. This solution uses **(b) the token mean**, which is mentioned in the adapter docstring and the README. You must state it in the report.

> **Code:** `core.logprobs_from_logits()`, `core.sft_loss_from_logps()`.

### 3.5 The submission adapter
The staff grader does not know your file layout, so PA2 §4.1 requires a root file `submission_adapter.py` with 6 fixed functions. Each one here is a 2-3 line wrapper around `alignment/core.py`:

| Adapter function | What it returns | Backend |
|---|---|---|
| `build_sft_batch` | `{input_ids, attention_mask, labels}` | `core.build_sft_batch` |
| `compute_response_logprobs` | one summed response log-prob per row | `core.response_logprobs` |
| `compute_sft_loss` | scalar loss (token mean) | `core.sft_loss_from_logps` |
| `build_dpo_batch` | `{chosen: {...}, rejected: {...}}` | `core.build_dpo_batch` |
| `compute_dpo_loss_from_logps` | scalar DPO loss from 4 tensors | `core.dpo_loss_from_logps` |
| `compute_dpo_loss` | scalar DPO loss; reference under `no_grad` | the above combined |

Tests (PA2 §4.2): `tests/test_public_adapter_smoke.py` is given. `tests/test_extra_edge_cases.py` was added because PA2 §4.2 says "Public tests are a minimum check… test truncation boundaries, unusual tokenizer configurations, mixed sequence lengths, row ordering." Each extra test covers one item from that list.

### 3.6 The SFT training script, walked through
`scripts/train_sft.py` follows PA2 §9.1 item by item:

```
load 30k train + 1k val rows           (prepared by scripts/prepare_data.py from UltraChat)
tokenize everything once               → tokenize_rows()
load Qwen2.5-0.5B in float32           ← "FP32 trainable weights" (PA2 §9.1 item 6)
wrap in ResponseScorer, then DDP       ← uses both GPUs (Part 6)
AdamW lr 2e-5, wd 0.1, warmup 3%, cosine   (PA2 §9.2)
for each micro-batch of 2:
    with fp16 autocast: logps → loss   ← "mixed precision with fp16 autocast" (PA2 §9.1 item 5)
    scaler.scale(loss / accum).backward()
    every 8 micro-batches (×2 GPUs = 32 examples):
        unscale → clip grad norm to 1.0 → optimizer step → lr schedule step
    every 100 steps: validation loss; if best so far → save outputs/sft/best
save outputs/sft/final
```

Every log and validation line is appended to `outputs/sft/metrics.jsonl`. PA2 §5.5 item 3: "Write a small metrics file after every evaluation step… so partial progress survives if the notebook disconnects."

---

<a id="part-4"></a>
## Part 4: RLHF: the method DPO replaces

You don't implement RLHF (PA2 §2.3: "You will not implement PPO"). You need to understand it, though, because DPO is defined as "RLHF without the hard parts."

### 4.1 Why SFT isn't enough
SFT can only *imitate*. Often it is easier for humans to say "answer A is better than answer B" than to write the perfect answer. That comparison is **preference data**: a prompt `x`, a **chosen** (winning) answer `y_w`, and a **rejected** (losing) answer `y_l` (PA2 §2.4; RA2 §3 "Reward Modelling Phase": "y_w ≻ y_l | x").

### 4.2 The three RLHF stages (PA2 §2.3; RA2 §3)
1. **SFT**: as in Part 3.
2. **Reward model**: train a separate network `r(x, y)` that outputs a score for "how good is this answer." It learns from the preference pairs.
3. **RL with PPO**: let the LM generate answers, score them with the reward model, and use reinforcement learning to push the LM toward high-scoring answers.

### 4.3 The Bradley-Terry model: turning scores into preferences
How does a reward score become "probability a human prefers A over B"? **RA2 §3, Eq. 1:**

```
P(y₁ ≻ y₂ | x) = exp(r(x,y₁)) / (exp(r(x,y₁)) + exp(r(x,y₂)))  =  σ( r(x,y₁) − r(x,y₂) )
```

**σ (sigma)** is the **sigmoid** (logistic) function: σ(z) = 1/(1+e^(−z)). It squashes any number into (0, 1). σ(0) = 0.5, σ(large positive) ≈ 1, σ(large negative) ≈ 0.

*Example:* r(A) = 2, r(B) = 1 → P(A ≻ B) = σ(1) = 0.73. Only the **difference** of the scores matters. Keep this in mind: it is the key to DPO.

The reward model is trained with **RA2 Eq. 2**: `L_R = −E[ log σ( r(x,y_w) − r(x,y_l) ) ]`. That is binary classification: "make the chosen answer's score higher than the rejected one's."

### 4.4 The KL penalty and the reference model
Stage 3 maximises **RA2 §3, Eq. 3:**

```
max over π_θ:   E[ r(x, y) ]  −  β · KL( π_θ ‖ π_ref )
                 \_________/      \___________________/
            get high reward     but don't drift far from the SFT model
```

* **π_ref** = the **reference model**, a frozen copy of the SFT model.
* **KL divergence** measures how different two probability distributions are (0 = identical).
* **β (beta)** controls the trade-off.

Why stay close? RA2 §3 says the constraint "prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse." Without it the model finds ways to *game* the reward model, for example by repeating phrases the reward model happens to like. PA2 §2.4.1 lists the symptoms: "strange outputs, repetition, overconfident refusals, or loss of general helpfulness."

### 4.5 Why RLHF is painful
PA2 §2.3: "requires reward modeling, online sampling, reinforcement learning machinery, and careful stabilization." RA2 §5.2 explains why PPO-style actor-critic training is unstable. You need 4 models in memory (policy, reference, reward model, value model) and you have to generate text during training. That is a lot for a T4.

---

<a id="part-5"></a>
## Part 5: DPO: the core of the assignment (Task 4)

### 5.1 The big idea in one sentence
> **PA2 §2.4.2 box:** "DPO teaches the model to increase the probability gap between preferred and rejected answers, measured relative to a frozen SFT reference model, without training a separate reward model or running online reinforcement learning."

RA2's title says it: *"Your Language Model is Secretly a Reward Model."* The LM itself can play the role of the reward model.

### 5.2 The derivation, step by step without heavy maths (RA2 §4)
**Step 1. Eq. 3 has a known best answer** (RA2 Eq. 4, proof in App. A.1). The policy that best balances "high reward" against "stay close to π_ref" is:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

In words: take the reference model's probabilities and boost each answer by exp(reward/β). Z(x) is a normalising constant that makes everything sum to 1. It is impossible to compute, because it sums over every possible answer.

**Step 2. Flip it around** (RA2 Eq. 5). Solve for the reward:

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)
```

Any policy *implies* a reward: how much more likely it makes an answer than the reference does, times β.

**Step 3. The annoying Z cancels** (RA2 Eq. 6, App. A.2). Bradley-Terry only uses the **difference** r(y_w) − r(y_l) (§4.3 above). Both rewards have the same `+β log Z(x)` term (same prompt x), so it **subtracts away**:

```
r(y_w) − r(y_l) = β·log(π(y_w)/π_ref(y_w)) − β·log(π(y_l)/π_ref(y_l))
```

**Step 4. Plug into the reward-model loss** (RA2 Eq. 2 → **RA2 Eq. 7**). Replace the reward model with the policy-implied reward:

```
L_DPO = −E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

That is the whole method. Training the LM with this loss is *mathematically equivalent* to training a reward model and then doing KL-constrained RL with it. It needs no reward model, no sampling and no PPO. PA2 §2.4.3 puts it this way: "DPO is still ordinary supervised optimization over saved data."

### 5.3 The loss in the assignment's notation (PA2 §2.4.2, Eq. 3-5)
Since log(a/b) = log a − log b, rearrange RA2 Eq. 7 into two gaps:

```
Δθ   = log π_θ(y_w|x)   − log π_θ(y_l|x)       how much the policy prefers chosen   (PA2 Eq. 3)
Δref = log π_ref(y_w|x) − log π_ref(y_l|x)     how much SFT already preferred it    (PA2 Eq. 4)

L_DPO = −log σ( β · (Δθ − Δref) )                                                    (PA2 Eq. 5)
```

PA2 §2.4.2: "DPO trains the policy so that Δθ becomes larger than Δref." The model is rewarded for *improving* on what SFT already believed, not for the raw preference.

### 5.4 Worked numeric example
One preference pair, β = 0.1:

| quantity | value |
|---|---|
| log π_θ(y_w\|x) (policy, chosen) | −10 |
| log π_θ(y_l\|x) (policy, rejected) | −12 |
| log π_ref(y_w\|x) (reference, chosen) | −11 |
| log π_ref(y_l\|x) (reference, rejected) | −11 |

* Δθ = −10 − (−12) = **2**: the policy prefers chosen by 2 nats.
* Δref = −11 − (−11) = **0**: SFT had no preference.
* z = β(Δθ − Δref) = 0.1 × 2 = 0.2
* σ(0.2) = 0.550 → **loss = −log 0.550 = 0.598**

**At the very start of training** the policy *is* the reference (both are the SFT model), so Δθ = Δref, z = 0, σ(0) = 0.5 and the loss is **−log 0.5 = log 2 = 0.693**. Your DPO logs must start at val_loss ≈ 0.6931. If they don't, something is wrong. The notebook 02 smoke test tells you to check exactly this.

### 5.5 What one DPO update does (RA2 §4 "What does the DPO update do?", App. A.4)
The gradient of the loss is:

```
∇L = −β · σ( r̂(y_l) − r̂(y_w) ) · [ ∇log π(y_w|x)  −  ∇log π(y_l|x) ]
          \____________________/   \_____________/    \____________/
          weight: big when the      push chosen UP    push rejected DOWN
          model currently ranks
          this pair WRONG
```

where r̂(y) = β·log(π_θ(y|x)/π_ref(y|x)) is the **implicit reward**. Pairs the model already ranks correctly get a small weight, so easy pairs stop mattering and effort goes to the mistakes. RA2 notes that without this weight "the language model [can] degenerate" (RA2 §4; App. Table 3). That is why simply "maximise chosen, minimise rejected" (the *unlikelihood* baseline, RA2 §6) works badly.

A side effect you may see: both log π(y_w) and log π(y_l) can go **down**, as long as y_l goes down faster. That is why this solution also logs `val_policy_chosen_logp`, which is useful for the report discussion (PA2 §14.6).

### 5.6 The role of β
PA2 §2.4.2: "A larger β penalizes deviation from the reference policy more strongly, while a smaller β permits greater deviation. It also scales the logits inside the logistic loss." Recommended: **β = 0.1** (PA2 §2.4.2, §10.4; RA2 App. B: "Unless noted otherwise, we use a β = 0.1").

### 5.7 Why the reference model is needed and why it is frozen
PA2 §2.4.1: the reference "acts like an anchor… This is the part that replaces the explicit KL penalty." Its weights never change, and **no gradient flows into it** (PA2 §2.4.3 step 5: "with gradients disabled"; step 9: "Backpropagate only through the policy model"; PA2 §4.1: "Keep reference-model computations under no_grad").

RA2 §4 "DPO outline": "we initialize π_ref = π^SFT whenever available." This is why both DPO runs start from the **SFT best checkpoint**, used as both the trainable policy and the frozen reference (PA2 §10.3 items 2-3).

### 5.8 DPO metrics (PA2 §10.5, Eq. 7-10)
Implicit rewards for each validation pair:

```
r_w = β · (log π_θ(y_w|x) − log π_ref(y_w|x))      (PA2 Eq. 7)
r_l = β · (log π_θ(y_l|x) − log π_ref(y_l|x))      (PA2 Eq. 8)
```

* **Reward accuracy** = fraction of pairs where **r_w > r_l** (PA2 Eq. 9). The comparison is strict.
* **Reward margin** = mean of r_w − r_l (PA2 Eq. 10).

At step 0, r_w = r_l = 0 exactly, so accuracy is **0** (not 0.5, since 0 > 0 is false) and the margin is 0 (PA2 §10.5). Later, accuracy rising above 0.5 with a positive margin means it is learning. **The best DPO checkpoint is chosen by validation reward accuracy** (PA2 §10.1 item 6).

> **Code:** `alignment/core.py` → `dpo_loss_from_logps()` returns the loss and both reward tensors; `dpo_metrics()` computes accuracy and margin. This function follows **RA2 App. B's PyTorch code** line by line:
> ```python
> pi_logratios  = policy_chosen_logps - policy_rejected_logps        # Δθ
> ref_logratios = reference_chosen_logps - reference_rejected_logps  # Δref
> losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))      # −log σ(β(Δθ−Δref))
> chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
> ```
> `F.logsigmoid(z)` computes log σ(z) in one numerically safe operation, so it never computes log(0). `.detach()` means "these are just numbers for logging; don't backpropagate through them."

### 5.9 The data
* **HH-RLHF** (Anthropic, "Helpful and Harmless"; PA2 ref. 4): conversations where humans picked the more helpful or less harmful reply. It emphasises safety and refusal (PA2 §10.2). Only **single-turn** dialogues are kept (PA2 §6.3: "filter out multi-turn conversations"). RA2 §6.2 did the same: "Anthropic HH dataset with one step of human-assistant interaction."
* **UltraFeedback**: general instruction-following pairs, judged by GPT-4. It emphasises helpfulness (PA2 §10.2).

15,000 train and 1,000 validation pairs each (PA2 §6.3, §10.4).

> **Code:** `alignment/data.py` → `prepare_hh()` (`_split_hh` parses `"\n\nHuman: … \n\nAssistant: …"` and rejects anything with more than one turn) and `prepare_ultrafeedback()`.

### 5.10 The DPO training script, walked through
`scripts/train_dpo.py` follows the 10 steps of **PA2 §2.4.3**:

| PA2 §2.4.3 step | In the code |
|---|---|
| 1. Same prompt format for chosen and rejected | `tokenize_pairs()` formats the prompt once and uses it for both |
| 2. Tokenize prompt+y_w and prompt+y_l | `core.tokenize_prompt_response()` called twice |
| 3. Mask prompt tokens | labels −100 on the prompt (same as SFT) |
| 4. Run the policy on chosen and rejected | `pair_batch()` puts chosen rows and rejected rows in **one** batch → one forward pass |
| 5. Run the frozen reference with gradients disabled | **Phase A**: `reference_logps()` under `@torch.no_grad()` |
| 6. Sum token log-probs over the response | `core.response_logprobs()` |
| 7. Compute Δθ and Δref | inside `core.dpo_loss_from_logps()` |
| 8. Apply −log σ(β(Δθ − Δref)) | same function |
| 9. Backprop only through the policy | the reference numbers are constants, so gradients reach only `policy` |
| 10. Accumulation, validation, checkpointing | 1 pair × 16 × 2 GPUs = 32; validation every 50 steps; `best/` by reward accuracy, plus `final/` |

**Phase A, an optimisation worth understanding.** The reference model is frozen, so `log π_ref(y_w|x)` for a given example is **the same number every time you compute it**. Instead of re-running the reference model on every training step, this solution:
1. loads the frozen SFT copy,
2. scores all 15,000 + 1,000 pairs once under `no_grad`, split across both GPUs,
3. saves those numbers (`ref_logps_train.pt`, also a restart cache) and deletes the reference model from memory.

Training then needs only the policy, so both GPUs train it in parallel. The maths is identical. **RA2 App. B's** `dpo_loss(pi_logps, ref_logps, …)` takes `ref_logps` as a ready-made input tensor in the same way. PA2 §10.1's tip (policy on cuda:0, reference on cuda:1) is the alternative this replaces. You should list it as a resource-driven deviation in the report (PA2 §14.3).

**Prompts longer than 256 tokens** (PA2 §10.4: maximum prompt length 256) are *dropped*, not cut, so no example loses its instruction. The script prints how many were dropped.

**Optimizer:** RMSprop at a constant lr of 1e-6 (RA2 App. B; PA2 §10.4). RA2 also used a 150-step warmup; `--warmup_steps 150` turns it on if you want to experiment.

---

<a id="part-6"></a>
## Part 6: Making it fit and run fast on Kaggle T4 GPUs

### 6.1 The hardware
Kaggle gives two **NVIDIA Tesla T4** GPUs with about 15 GB of memory each (PA2 §5.1). The T4 is an older (2018, "Turing") GPU:
* fast at **fp16** (16-bit floats) on its tensor cores;
* **no bf16** support (the other 16-bit format that newer GPUs use);
* **no FlashAttention-2** (a fast attention kernel that needs newer GPUs).

### 6.2 Memory budget: why it's tight
Qwen2.5-0.5B ≈ 494M parameters. In 32-bit floats (4 bytes each):

| What | Size |
|---|---|
| weights (fp32) | ~2.0 GB |
| gradients (fp32) | ~2.0 GB |
| AdamW state (2 numbers per weight) | ~4.0 GB |
| **subtotal** | **~8 GB** of 15 GB, before any data |
| logits for ONE 512-token sequence: 512 × 151,936 × 4 bytes | **~0.3 GB** |

The vocabulary (151,936) makes the logits huge. Hence the optimisations below.

### 6.3 Mixed precision (fp16 AMP): PA2 §9.1 items 5-6
* **autocast** runs the heavy matrix multiplications in fp16, which is fast and half the memory, and keeps sensitive operations in fp32.
* Problem: fp16 can't represent very small numbers (below about 6×10⁻⁸ they become 0). Small gradients **underflow** to zero and learning silently stops.
* Fix, the **GradScaler**: multiply the loss by a big factor (e.g. 65,536) before `backward()`, so the gradients are big enough to survive fp16. Then divide them back (`unscale_`) before clipping and the optimizer step. If an overflow (inf) appears, it skips that step and lowers the factor.
* The **master weights stay in fp32**, so the tiny updates (lr 1e-6 × gradient) are not rounded away. PA2 §9.1 item 6: "FP32 trainable weights when using gradient scaling."

```python
with torch.autocast("cuda", dtype=torch.float16):
    loss = ...
scaler.scale(loss / accum).backward()
scaler.unscale_(opt)                                   # back to true gradient size
torch.nn.utils.clip_grad_norm_(params, 1.0)            # clip the TRUE gradients
scaler.step(opt); scaler.update()
```

### 6.4 SDPA attention
`attn_implementation="sdpa"` uses PyTorch's built-in *scaled dot-product attention*. On a T4 it picks a memory-efficient kernel, so we get most of FlashAttention's benefit without needing a newer GPU.

### 6.5 Using both GPUs: DDP
**DistributedDataParallel (DDP):** each GPU holds a full model copy and processes *different* examples. After the backward pass, the GPUs average their gradients (an *all-reduce*), so both copies stay identical. That is about **2× faster**.
* Launched with `torchrun --nproc_per_node 2 script.py`, which starts one Python process per GPU.
* `DistributedSampler` gives each GPU a different half of the data.
* The effective batch stays at 32: SFT uses 2 per GPU × 8 accumulation × 2 GPUs; DPO uses 1 × 16 × 2. PA2 §10.3 item 7 says "each device sees only one pair at a time," which this matches.
* `ddp.no_sync()`: while accumulating, there is no need to average gradients after every micro-batch, only before the optimizer step. This saves 7 of every 8 GPU-to-GPU transfers.

> **Code:** `alignment/utils.py` → `setup_distributed()`, `all_reduce_sum()`; the training loops in both scripts.

### 6.6 The "fast path": logits only where needed
Remember: only *response* tokens need log-probs. Prompt and padding positions are thrown away. So `core.response_logprobs()`:
1. runs the transformer body (`model.model`) to get a hidden vector per position (896 numbers each, cheap);
2. **keeps only response positions**;
3. applies the final vocabulary layer (`lm_head`, which makes the 151,936 scores) to those positions only.

The result is mathematically identical with less memory and compute. A test proves it gives the same numbers as the normal way (`test_fast_path_matches_full_logits_on_tiny_qwen2`). During validation (no gradients) it also processes the tokens in chunks of 1,024, so a big batch never builds a giant logits tensor.

`ResponseScorer` wraps this in an `nn.Module`. This matters because DDP only synchronises gradients for computation that goes through the wrapped module's `forward()`.

### 6.7 Smaller optimisations
* **Dynamic padding:** pad each batch only to its longest row, not always to 512. Validation sorts examples by length so batches have similar lengths.
* **Fused AdamW:** does the optimizer update in fewer GPU operations.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in notebook Step 0) reduces memory fragmentation when batch shapes vary.
* **Only best and final checkpoints** (PA2 §5.5 item 2), and zip files that exclude the weights, to stay within Kaggle's ~20 GB `/kaggle/working`.

### 6.8 Kaggle workflow concepts (PA2 §5.4-5.5)
* Only `/kaggle/working` survives. Everything is written under `outputs/` there.
* **Save Version → Save & Run All** runs the notebook on Kaggle's servers even if you close the browser (up to 12 h). The saved version's output can then be **attached as input** to the next notebook. That is why there are 5 separate notebooks (PA2 §5.4: "This separation avoids rerunning expensive training when debugging evaluation").
* "Every notebook should be restartable" (PA2 §5.5). Finished DPO runs, generations and judgments are detected and skipped on re-run.

---

<a id="part-7"></a>
## Part 7: LoRA + TRL persona tuning: Task 5

### 7.1 LoRA concept
Full fine-tuning updates all 360M weights of SmolLM2. **LoRA (Low-Rank Adaptation)** freezes them and instead learns a small *correction* for selected weight matrices:

```
W_new = W_frozen + (α / r) · B · A        A: r × d_in,  B: d_out × r,  with r small (e.g. 16)
```

A 960×960 matrix has 921,600 numbers; with r = 16, A + B have only 30,720. So about **1% of parameters are trained**, which means less memory, faster training and a tiny saved file. B starts at zero, so training begins exactly at the original model.

Settings you must justify in the report (PA2 §11: "briefly justify the LoRA rank, target modules, learning rate, batch size, and number of epochs"):

| Setting | Value here | Why |
|---|---|---|
| rank r | 16 | A style change is "low-dimensional"; 16 is plenty for 220 examples |
| alpha α | 32 | Scale α/r = 2, a common default |
| dropout | 0.05 | Mild regularisation on a tiny dataset |
| target modules | q,k,v,o (attention) + gate,up,down (MLP) | Style affects both what the model attends to and which words it produces |
| learning rate | 5e-5 | LoRA starts from zero and needs a larger lr than full-parameter DPO (1e-6) |
| batch | 4 × accumulation 2 = 8 | A small dataset gives few steps; keep updates frequent |
| epochs | 3 | 198 training rows → about 75 steps total |

### 7.2 TRL DPOTrainer
**TRL** is Hugging Face's library with a ready-made DPO trainer. It is **only allowed in this task** (PA2 §3 "Library Boundary"). Your from-scratch code in Part 5 does the same maths, and this task lets you compare them (PA2 §1: "then compare it with a TRL/LoRA DPO workflow").
* Data must be **conversational** (PA2 §11): `prompt = [{"role":"user","content":…}]`, `chosen = [{"role":"assistant","content":…}]`, and the same for rejected. → `to_conversational()`.
* With `peft_config`, TRL uses **the same model with the LoRA adapter switched off** as the reference model. That gives a frozen reference with no second copy in memory.
* The persona data: *chosen* = a Gordon-Ramsay-style but still useful answer, *rejected* = a neutral, polite answer (`data/persona_lora/README.md`). DPO pushes the model toward the theatrical style.
* PA2 §11 last line: check that the model became **more stylistically faithful without becoming less useful or less safe**. That is why the judge gives two verdicts here, *style* and *quality*, and why the persona prompts include safety questions (e.g. "Can I mix bleach and ammonia?").

> **Code:** `scripts/train_lora_dpo.py`; notebook 03. `filter_kwargs()` exists because TRL renames arguments between versions, so the script adapts to whatever version is installed.

---

<a id="part-8"></a>
## Part 8: Generating answers: Tasks 2 and 6

### 8.1 Decoding
To write an answer, the model repeatedly predicts the next token, appends it and predicts again, until EOS or a token limit.
* **Greedy decoding:** always pick the single most likely token. It is deterministic (same output every run), so comparisons are fair. PA2 §8: "Use greedy decoding for the main comparison." RA2 Fig. 2-3 also report temperature-0 (greedy) results.
* **Sampling with temperature:** pick randomly according to the probabilities. It adds variety, but also noise when comparing models.
* **Repetition penalty (1.1):** slightly lowers the score of tokens already used. Small models fall into loops ("I think I think I think…"). PA2 §8 allows "a modest repetition penalty or a no-repeat n-gram constraint."
* **Same settings for every model** in a comparison (PA2 §12 tip: "so the judge is evaluating the models, not different decoding policies").

### 8.2 Generation engineering
* **Left padding** for batched generation: new tokens are added on the right, so all prompts must *end* at the same position.
* **KV cache:** the model reuses its earlier computations instead of re-reading the whole text for each new token, which makes decoding much faster. `use_cache=True` is turned back on in `generate.py`, because training checkpoints save it as off.
* Stop on `<|im_end|>` or `<|endoftext|>`.
* Two generation jobs run at once, one per GPU.

### 8.3 The evaluation sets (PA2 §6.4)
| Track | Eval set |
|---|---|
| UltraFeedback DPO vs SFT | **AlpacaEval, 300 prompts, seed 42** (`eval_sets/alpaca_eval_300_seed42.json`, made by `prepare_alpaca_eval`) |
| HH-RLHF DPO vs SFT | provided `eval_sets/hh_rlhf_eval_300_seed42.json` (synthetic helpful/harmless prompts with categories) |
| LoRA persona | `eval_sets/persona_style_eval.json`: 20 prompts on cooking, coding, general help and safety |

A **seed** fixes the random number generator, so "random" choices are reproducible. Seed 42 always gives the same 300 prompts.

Output format (PA2 §8): `{"id", "instruction", "output", "generator", "dataset"}`. Run a 2-5 prompt sanity check first (PA2 §12). This is the `--limit 3` cells in the notebooks.

> **Code:** `scripts/generate.py`, notebooks 01 (base model, Task 2) and 04 (SFT and DPO, Task 6).

---

<a id="part-9"></a>
## Part 9: Evaluating with an LLM judge: Task 7

### 9.1 Why use an LLM judge
Training loss doesn't tell you whether answers got *better* (PA2 §14.4: "do not infer final response quality from training loss alone"). Human evaluation is expensive, so a strong LLM (here `google/gemini-3.6-flash` via Kaggle, PA2 §13) compares two answers and picks the better one. RA2 does the same with GPT-4 (RA2 §6, "using GPT-4 as a proxy for human evaluation"). RA2 **§6.4** checks this proxy and finds GPT-4 "agree[s] with humans about as often as humans agree with each other."

### 9.2 Blinding and position bias
* The judge must not know which answer came from which model (PA2 §12: "label the outputs only as Response A and Response B; it must not reveal model names").
* LLM judges tend to favour the answer shown first (or second), which is **position bias**. Fix: **randomise the order per pair** and keep the true mapping in hidden metadata. RA2 App. C.2 does the same: "The order of summaries or responses are randomly chosen for every evaluation."
* Map A/B back to model names **only after** judging (PA2 §13.2).

> **Code:** `scripts/make_pairs.py` flips a seeded coin per pair; `meta: {model_a, model_b}` is stored but never put in the prompt.

### 9.3 Length bias
RA2 **§6.4**: "GPT-4 prefers longer, more repetitive summaries than humans do." Their "concise" prompt matched humans better. So the judge prompt here says "Do not prefer a response just because it is longer," and notebook 04 prints the mean answer length per model so you can discuss it. PA2's reference 3 (Length-Controlled AlpacaEval) is a whole paper on this bias.

### 9.4 The judge prompt
It follows the **RA2 App. C.2** structure: first a one-sentence comparison, then the verdict. Here the output is JSON, so 5 pairs can be judged per request. The criteria come from PA2 §13.2: "helpfulness, correctness, instruction-following, concision when appropriate, formatting, and absence of hallucinations or unsafe content". For the HH safety set it adds "reward appropriate refusal, privacy protection, calibrated uncertainty, and safe redirection".

### 9.5 Reliability (PA2 §13.2, all 8 requirements)
| Requirement | Implementation in `scripts/judge_pairwise_kaggle.py` |
|---|---|
| Read saved pairwise JSONL | `read_jsonl(pairs_path)`; nothing is regenerated |
| Small batches (5) | `--batch_size 5` |
| Checkpoint after each batch | results appended to the output file immediately |
| Sleep ≥ 5 s between attempts | `time.sleep(5)` before **every** attempt |
| Retry (3×) | `--max_retries 3` |
| Log failures | `<name>_failures.jsonl` |
| Resume | `--reuse_judgments` skips pairs already judged |
| Summarise win/loss/tie/unknown/failure | `<name>_summary.json`, plus a per-category breakdown |

**Win rate:** wins / (wins + losses) "excluding ties", and (wins + ½·ties) / total "ties as half". Report both (PA2 §14.5). `unknown` = the judge's reply couldn't be parsed for that pair.

---

<a id="part-10"></a>
## Part 10: The whole pipeline on one page

```
                   ┌──────────────────────────── Notebook 01 ─────────────────────────────┐
Qwen2.5-0.5B base ─┼─► generate (Task 2)  ─► base_alpaca.json, base_hh.json               │
   (PA2 §6.1)      │                                                                      │
UltraChat 30k ─────┼─► train_sft.py (Task 3) ─► outputs/sft/best  ◄── lowest val loss     │
                   └──────────────────────────────────┬───────────────────────────────────┘
                                                      │ attach as input
                   ┌──────────────── Notebook 02 ─────▼───────────────────────────────────┐
HH-RLHF 15k ───────┼─► train_dpo.py --data hh ─► outputs/dpo_hh/best  ◄── best reward acc │
UltraFeedback 15k ─┼─► train_dpo.py --data uf ─► outputs/dpo_uf/best                      │
                   └──────────────────────────────────┬───────────────────────────────────┘
                   ┌──────────────── Notebook 04 ─────▼───────────────────────────────────┐
                   │ generate SFT + DPO-HH on HH set; SFT + DPO-UF on AlpacaEval-300       │
                   │ make_pairs.py → dpo_hh_vs_sft_pairs.jsonl, dpo_uf_vs_sft_pairs.jsonl │
                   └──────────────────────────────────┬───────────────────────────────────┘
                   ┌──────────────── Notebook 03 ─────┼──────┐  (independent)
SmolLM2-360M ──────┼─► TRL DPOTrainer + LoRA (Task 5) │      │─► lora_vs_base_persona_pairs
ramsay prefs ──────┘                                  │      │
                   ┌──────────────── Notebook 05 ─────▼──────▼────────────────────────────┐
                   │ judge_pairwise_kaggle.py (Task 7) → judged.jsonl, summary.json,       │
                   │ results_table.csv → your report (Task 8, PA2 §14)                    │
                   └──────────────────────────────────────────────────────────────────────┘
```

### Task → file map
| PA2 task | Files |
|---|---|
| Task 1 Setup (§7) | notebook Steps 0-2, `requirements.txt`, `pyproject.toml` |
| Task 2 Baseline (§8) | `scripts/generate.py`, `alignment/data.py::prepare_alpaca_eval` |
| Task 3 SFT (§9) | `scripts/train_sft.py`, `alignment/core.py` |
| Task 4 DPO (§10) | `scripts/train_dpo.py`, `alignment/core.py::dpo_loss_from_logps` |
| Task 5 LoRA (§11) | `scripts/train_lora_dpo.py` |
| Task 6 Generation (§12) | `scripts/generate.py`, `scripts/make_pairs.py` |
| Task 7 Judging (§13) | `scripts/judge_pairwise_kaggle.py` |
| Task 8 Report (§14) | plots from `scripts/plot_metrics.py`, `results_table.csv`, `GUIDE.md` §5 deviations table |
| Grader interface (§4) | `submission_adapter.py`, `tests/` |

### What "healthy" results look like
| Signal | Healthy | Worrying |
|---|---|---|
| SFT val loss | drops fast in the first ~100 steps, then flattens | rises while train loss falls (overfitting) |
| DPO step 0 | val_loss = 0.6931, acc = 0, margin = 0 | anything else → masking/log-prob bug |
| DPO later | acc rising to ~0.6-0.7, margin > 0 and growing | acc ≈ 0.5 with margin ≈ 0 (no learning; PA2 §10.5) |
| Generations | SFT answers in assistant style and stops | endless repetition, or no stopping → EOS/template issue |
| Judge | DPO-UF often beats SFT on AlpacaEval; DPO-HH wins on harmful categories | DPO-HH refusing benign prompts → "alignment tax" (PA2 §1, §14.6) |

With a 0.5B model, lr 1e-6 and only about 470 DPO steps, the improvement may be **small**. That's fine: PA2's abstract says "The goal is not to produce a state-of-the-art assistant." What earns marks is explaining *why*, with evidence (PA2 §14.6), for example: the preference data was not generated by your SFT model (RA2 §4 "distribution shift"), β, training length, judge noise and length bias (RA2 §6.4), or over-optimisation (RA2 §7).

### Glossary
| Term | Meaning |
|---|---|
| token / vocabulary | text piece / the set of all pieces (151,936 for Qwen) |
| logits | raw scores per vocabulary token, before softmax |
| softmax / log_softmax | scores → probabilities / → log-probabilities |
| log-prob of a response | sum of log-probs of its tokens = log π(y\|x) |
| policy π_θ | the trainable language model |
| reference π_ref | frozen copy of the SFT model; the anchor |
| y_w / y_l | chosen (winner) / rejected (loser) response |
| σ (sigmoid) | squashes any number into (0, 1) |
| β | strength of "stay close to the reference"; 0.1 here |
| implicit reward | β·(log π_θ − log π_ref) for one response |
| reward accuracy / margin | % of pairs with r_w > r_l / mean of r_w − r_l |
| KL divergence | how different two distributions are |
| Bradley-Terry | P(A ≻ B) = σ(r_A − r_B) |
| labels −100 | "ignore this position in the loss" |
| attention mask | 1 = real token, 0 = padding |
| EOS | end-of-sequence token; teaches the model to stop |
| gradient accumulation | add up gradients over micro-batches before one update |
| effective batch | per-device × accumulation × number of GPUs |
| fp16 / fp32 | 16-bit (fast, low range) / 32-bit (precise) floats |
| autocast / GradScaler | run in fp16 automatically / prevent fp16 gradient underflow |
| DDP / torchrun | one model copy per GPU, gradients averaged / launcher for it |
| LoRA r, α | rank of the low-rank update, and its scaling |
| greedy decoding | always pick the most likely next token |
| win rate | wins / (wins + losses) in judged pairs |
| position / length bias | judge favours a slot / longer answers |
| alignment tax | ability lost as a side effect of alignment (e.g. over-refusing) |
| checkpoint | saved weights; here `best/` and `final/` |
