# A curriculum of experiments

The pipeline is the instrument; these are the things worth measuring with it.
Each one is a config change plus a re-run, and each isolates one idea. Record
`macro_f1`, `parse_rate` and `best_val_loss` for every run — `runs/*/summary.json`
and `runs/*/eval_report.json` already hold them.

Run the **base model** first and keep that row at the top of your table. Every
number below is only meaningful as a delta from it.

---

## 1. Prompt masking — does loss-on-completion matter?

```bash
geosft train -c configs/default.yaml --name mask-on
# set train.mask_prompt: false
geosft train -c configs/nomask.yaml --name mask-off
```

**Expect:** a large win for masking. With `mask_prompt: false` most of the
gradient teaches the model to reproduce the input passage, which is not the
task. Watch train loss: the unmasked run will show *lower* loss (predicting
prose is easier than predicting exact JSON) while scoring *worse*. This is the
cleanest demonstration in the project that **training loss is not the metric.**

## 2. LoRA rank and depth — how much capacity does the task need?

Sweep `lora_rank` ∈ {4, 8, 16, 32, 64} at fixed `num_layers: 16`, then sweep
`num_layers` ∈ {4, 8, 16, -1} at fixed rank.

**Expect:** a sharp knee at low rank, then a long flat plateau. Format-and-
extraction tasks need far less capacity than people assume. Note where peak
memory and time start growing faster than macro-F1 — that point is the actual
answer to "what rank should I use".

## 3. Dataset size — where is the data ceiling?

Subsample `train.jsonl` to 500 / 1,000 / 4,000 / all rows, iterations held
constant.

**Expect:** a log-shaped curve. The interesting question is where it flattens:
that tells you whether your next hour is better spent collecting data or tuning
hyperparameters. Most people guess wrong here, which is why you measure it.

## 4. Model size — is a bigger base worth it?

Qwen3-1.7B → 4B → 8B, all 4-bit, everything else fixed.

**Expect:** the *gap between base and tuned narrows* as the model grows. Bigger
models already follow the JSON schema, so SFT buys less format compliance and
only the genuine domain extraction remains. Watch `parse_rate` for the base
model specifically — that column tells the story.

## 5. Vocabulary extension — the A/B this repo was built to run

```bash
geosft train --name no-ext                                  # stock base
geosft extend && geosft train --model artifacts/models/...-mlx-4bit --name ext-frozen
geosft train --backend hf --train-embeddings --name ext-trained
```

**Expect:** `ext-frozen` ≈ `no-ext`, or slightly worse. MLX LoRA never touches
embeddings, so the grafted rows keep their mean-init values forever.
`ext-trained` is the only arm that can move — and with a few thousand examples
it probably still won't move much.

**A null result here is a real result.** Vocabulary extension is a
pretraining-scale technique. What you *will* see clearly is the context saving
from `geosft extend`'s verify step (shorter sequences, faster steps, more room
in the window). That is a genuine win even when quality is flat — and knowing
the difference between "cheaper" and "better" is the point of the experiment.

## 6. Overfitting — find it deliberately

Set `iters: 3000` on the default config and watch the live panel.

**Expect:** val loss bottoms out and turns up while train loss keeps falling.
The tracker will fire its patience alarm. Then evaluate the *final* adapter
against the *best* checkpoint and confirm the gap — this is why
`save_every` and checkpoint selection exist, and it is much more convincing
once you have seen your own curve do it.

## 7. Learning rate — see both failure modes

Run `1e-3`, `1e-4`, `1e-5`.

**Expect:** `1e-3` diverges or goes NaN (the tracker will say so), `1e-5` looks
like almost nothing happened. Seeing both failures once makes every future
"my fine-tune didn't work" diagnosable in about thirty seconds.

## 8. Generalisation — the test that actually matters

The rule labeller is high-precision and low-recall by construction. So:

1. Hand-correct the gold set (`geosft review`, set `"reviewed": true`).
2. Score base and tuned against `--gold`.

**Expect:** the tuned model to find things *the rules missed* — lithologies
phrased unusually, relations in sentence forms the regexes don't cover. That is
the moment the project stops being "a model that imitates my regexes" and starts
being a domain model. If it never happens, your labels are too narrow, and the
fix is better data rather than more training.

---

## Recording results

| run | iters | rank | layers | macro F1 | strict JSON | best val | min |
|-----|-------|------|--------|----------|-------------|----------|-----|
| base (untuned) | – | – | – | 0.314 | 1.000 | – | – |
| geo-sft-v1 | 900 | 16 | 16 | **0.831** | 0.000 ¹ | 0.0703 | 117 |
| geo-sft-v2-fixcheck | 300 | 16 | 16 | 0.649 | **1.000** | 0.1007 | 41 |

¹ template-prefix bug, fixed by `PromptCompletionDataset`; see the README.
v2 is undertrained (300 vs 900 iters), not worse — it exists to prove the fix.

Keep one row per run. The table, not any single run, is the deliverable.

Keep one row per run. The table, not any single run, is the deliverable.
