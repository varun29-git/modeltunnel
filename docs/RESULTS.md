# Results and evidence guide

## Scope

The completed experiments evaluate one 146,423,680-parameter, 25-layer decoder with width 640, a tied 65,536-token vocabulary, five query heads and one KV head, 4,096-token context, and four local-512 attention layers per global-attention layer. Both stages use BF16 residuals, FP8 projection operands/activations, FP32 latent projection weights, and Muon with auxiliary AdamW groups. The dense stage uses dense projection weights; QAT switches all 104.448M projection parameters to scaled ternary `{-1,0,+1}` codes while retaining FP8 activations.

Training uses ordinary next-token cross-entropy on packed streams. Reported validation values are mean token negative log-likelihood on the same held-out source shards, with 1,572,864 tokens per source per validation event. They are not task-accuracy measurements.

## Completed runs

| Measurement | Dense FP8 | Ternary QAT continuation |
| --- | ---: | ---: |
| New training tokens | 9,999,998,976 | 1,999,998,976 |
| Optimizer updates | 25,432 | 5,087 |
| Nominal tokens/update | 393,216 | 393,216 |
| Warmup / stable / decay | 1B / 7B / 2B | 0.2B / 1.4B / 0.4B |
| Peak base learning rates | 0.01 | 0.005 |
| Training mixture | 50% English, 10% code, 10% math, 30% Indic | 37% English, 13% code, 13% math, 37% Indic |
| Active training time, one H100 | 13.11 hours | 2.55 hours |
| Recorded non-finite incidents | 0 | 0 |
| Final checkpoint SHA-256 | `3d116e88048c93a0403ab2920b1680b066a9f29cbc07fa8ae1c2c67f2144684b` | `ceee823262d11e914eb63d6a396c3df90fb47e5dd9301f7db0f8add0a468c29f` |

The dense stage was interrupted and resumed from a verified checkpoint. QAT started from the SHA-verified dense final checkpoint, retained compatible optimizer moments and RNG state, reset phase/data counters explicitly, and consumed checked fresh source tails. The final partial update in each stage is included in the token totals.

## Held-out loss

Aggregate comparisons must use the same source weighting for both checkpoints:

| Fixed validation weighting | Dense at 10B | QAT at 10B+2B | QAT minus dense |
| --- | ---: | ---: | ---: |
| Original dense mixture | **2.4194** | 2.6042 | **+0.1848** |
| QAT training mixture | **2.3154** | 2.4889 | **+0.1735** |

Selected per-source values:

| Source | Dense | QAT | Delta |
| --- | ---: | ---: | ---: |
| English | 2.7597 | 2.9857 | +0.2260 |
| Code | 0.6389 | 0.7208 | +0.0819 |
| Math | 1.1989 | 1.3165 | +0.1176 |
| Hindi | 2.8699 | 3.0459 | +0.1761 |
| Marathi | 2.9260 | 3.1042 | +0.1782 |
| Kannada | 2.7501 | 2.9107 | +0.1606 |
| Telugu | 2.7916 | 2.9716 | +0.1800 |
| Tamil | 2.8090 | 2.9862 | +0.1772 |
| Gujarati | 2.9592 | 3.1235 | +0.1643 |
| Bengali | 2.8593 | 3.0338 | +0.1744 |

Immediate ternary conversion caused a large loss increase, followed by substantial recovery during QAT. Nevertheless, final QAT loss remained worse on every tracked source. The dense run also showed an unusually small first attention RMSNorm gain and large sampled residual scales; these are follow-up signals, not proof of a failed run or an identified cause.

## What the evidence does not establish

- **No matched dense continuation:** QAT changed weight mode, data mixture, source position, and LR schedule together. The loss gap cannot be attributed solely to ternarization without a same-parent, same-data, same-schedule dense control.
- **No downstream QAT suite:** code execution, math accuracy, task completion, Indic task quality, and conditional-loss-to-downstream transfer remain unmeasured for the QAT checkpoint. A small dense prompt probe is illustrative only.
- **No recipe optimum or scale claim:** there is one seed and one tested width. The results do not select the best dense/QAT split, LR, mixture, architecture, or 1.5B recipe.
- **No deployment result:** these are QAT training checkpoints, not a packed-ternary serving engine or latency/quality comparison.

## Committed evidence versus external inputs

| Artifact | Location | Contents |
| --- | --- | --- |
| Dense analysis | `experiments/terminus_lean_150m_10b/analysis/FINAL_RUN_ANALYSIS.md` | Methods, validation trajectory, stability audit, throughput, fixed prompt probe and limitations |
| Dense compact metrics and figures | `experiments/terminus_lean_150m_10b/analysis/figures/` | `metrics.json` and generated plots; no checkpoint tensors or raw event streams |
| Combined dense/QAT analysis | `experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md` | Matched-weighting comparisons, conversion shock, quantization and parameter-group diagnostics |
| Combined compact metrics and figures | `experiments/terminus_lean_150m_qat_2b/analysis/` | `combined_metrics.json` and four plots |
| Analysis programs | The two experiment `analysis/` directories | Recompute compact artifacts from external logs and, for combined analysis, SHA-verified final checkpoints |
| Run configurations and audits | `experiments/terminus_lean_150m_10b/`, `experiments/terminus_lean_150m_qat_2b/` | Model/training settings, data-view construction, completion gates and audit code |

The checkpoint files, raw token shards, evaluation records, `events.jsonl`, `stability.jsonl`, initial QAT evaluation, and cloud verification receipts are **not committed**. Their historical existence and verification are reported in the analyses, but the compact committed metrics are not substitutes for those raw inputs. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact regeneration commands and boundaries.
