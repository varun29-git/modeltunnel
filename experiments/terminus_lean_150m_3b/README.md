# Terminus Lean: one dense FP8 run at 150M scale

**Prepared, not launched.** This folder defines one 3B-token dense-weight FP8 run on an Azure H100. It does not start training, allocate a VM, or prepare data. The separate data workstream must deliver a validated manifest before any run.

| Item | Registered value |
| --- | --- |
| Model | 146,423,680 parameters; 25 layers, width 640, FFN 1664, 5 query / 1 KV head |
| Attention | Four 1,024-token-local layers, then one global; 4,096-token context |
| Cuts | FFN SubLN, attention SubLN, and Q/K unit RMS removed; learned pre-block/final RMSNorm retained |
| Tokenizer | `tokenizer/tokenizer-65k.json`, SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29` |
| Precision | Dense FP32 master weights; FP8 E4M3 weight and activation operands; BF16 residuals |
| Production kernels | DeepGEMM, FA3, Triton split FP8 quantization, full FP8 backward, fused QKV/SwiGLU/pointwise, Triton loss |
| Optimizer | Muon for projection matrices, auxiliary AdamW elsewhere |
| Batch | 262,144 tokens/update on one GPU; initial microbatch 32 × 4,096 × accumulation 2 |
| LR | `0.01` base for Muon, AdamW, and norm groups; per-matrix Muon effective rates differ due to shape scaling |
| WSD | 300M warmup, stable until 2.4B, 600M decay, 0.1 final LR fraction |
| Budget | 3,000,000,000 nominal; 2,999,996,416 usable tokens; 11,445 updates including a partial final update |
| Checkpoints | Update 5,722 at 1,499,987,968 tokens; final update 11,445 at 2,999,996,416 tokens |

[MiniCPM4/BitCPM4 Section 3.3.1](../../papers/minicpm4_2506.07900.pdf) reports `1e-2` for its **5M/2B dense FP8 pilot** and `5e-3` after conversion to ternary QAT. This run copies the dense numeric LR only. The paper does not give a 146M Muon optimum; the shape-scaled Muon rate can differ substantially from the paper's optimizer. The 300M warmup and 0.1 LR floor are our provisional choices; the paper specifies an 80% combined warmup/stable phase and 20% decay for that pilot, not those exact values. **No QAT conversion or ternary training is scheduled here.** The removed-normalization architecture and `0.01` LR are jointly being tested, so the run does not isolate their effects.

`plan.py generate` writes only `lean_dense.json` and a synthetic microbatch-sweep config. `plan.py check` verifies the tokenizer, parameter count, dense mode, native stack, batch arithmetic, learning rate and checkpoint boundaries without touching data. `prepare_views.py` is for **later**, after the data owner confirms readiness; it validates a delivered manifest and derives one full dense view over existing shards without retokenizing.

Per-update records in `events.jsonl` include train NLL, token count, total gradient norm before clipping, clip coefficient, actual optimizer-group LR range, FP8 mode strength, throughput, and allocated/reserved/peak GPU memory. Every tenth update, plus the first, midpoint, and final updates, `stability.jsonl` samples all layers' residual/attention/FFN magnitudes, Q/K and attention-logit/entropy behavior, FP8 input reconstruction error and scale ranges, plus sampled weight, gradients before and after clipping, and update-to-weight ratios. Validation runs every 100 updates by language/domain. Nonfinite loss and gradient failures go to `incidents.jsonl`. Sampled probes are diagnostic evidence, not exhaustive scans of every token or exact FA3 attention probabilities. The FP8 reconstruction probe uses the matching reference quantizer on sampled inputs; native-kernel numerical agreement remains a separate H100 preflight requirement.

The planned two checkpoint files are immutable and SHA-receipted. A signal interruption can create an extra recovery checkpoint; that is intentional. The midpoint is a stop/resume boundary, not a second experiment. The runbook is [AZURE_H100.md](AZURE_H100.md). The cuts and their scope are in [CUTS.md](CUTS.md).
