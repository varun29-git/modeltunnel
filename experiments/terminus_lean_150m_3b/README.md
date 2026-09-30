# Terminus Lean: 150M-scale, 3B-token H100 comparison

**Status: prepared, not launched.** No corpus preparation, Azure VM, GPU preflight, benchmark, or training run is performed by this folder's creation. Start only after the owner delivers the prepared data and authorizes the run.

This folder compares the existing normalization policy (`control`) with a cut-down one (`lean`) on the **same production training implementation**. Each variant is a separate 3B-token-total dense-FP8 → ternary-QAT experiment. The control costs another 3B tokens; it is necessary to tell whether the cuts helped. The 60/40 dense/QAT split and learning rates are registered trial settings, not claimed optima.

| Item | Value |
| --- | --- |
| Parameters | 146,423,730 control; 146,423,680 lean |
| Shape | 25 layers, width 640, FFN 1664, 5 query / 1 KV head, head width 128 |
| Attention | 4 local layers (1,024-token window) then 1 global; 4,096-token context |
| Tokenizer | `tokenizer/tokenizer-65k.json`, SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29` |
| Precision | dense FP8 E4M3 weights + FP8 activations; then ternary FP8 weight codes + FP8 activations; FP32 masters |
| H100 path | DeepGEMM, FA3, Triton split quantizer, full FP8 backward, fused QKV/SwiGLU/pointwise, Triton loss, BF16 residual, logical Muon |
| Tokens per update | 262,144 on one GPU, initially microbatch 32 × 4,096 × accumulation 2 |
| Dense → QAT | 6,867 dense updates = 1,800,142,848 tokens; then 1,199,853,568 usable QAT tokens |

The nominal budget is 3,000,000,000 tokens per variant. The last 3,584 tokens cannot form a 4,096-token sequence, so the usable total is 2,999,996,416. The final QAT update is partial; the trainer accounts for it. The 5:1 query/KV ratio differs from the 1.5B model's 3:1 ratio because 640 width and 128-wide heads give five query heads. This proxy preserves 25-layer depth and the 4:1 local/global schedule; it does not prove width transfer.

`plan.py generate` writes the four immutable candidate configs and a synthetic throughput-sweep config. `plan.py check` verifies their source-independent contract and tokenizer hash. `prepare_views.py` is deliberately separate: **do not run it until the data owner says the prepared corpus is ready**. It will verify the delivered manifest and derive two budget views over the same existing token shards, without retokenizing or changing source data.

```bash
.venv/bin/python experiments/terminus_lean_150m_3b/plan.py check --microbatch 32
```

On the Azure H100 host, after data readiness and run approval, follow [AZURE_H100.md](AZURE_H100.md). The first GPU work is native numerical preflight and a bounded throughput sweep. Only then select the microbatch and start the 3B-token jobs. Compare per-language/domain held-out loss, task conditional loss, conversion spike and recovery, nonfinite events, attention-logit/gradient behavior, and **measured** training tokens per GPU-hour. Dense-only quality is not the decision criterion; the final ternary checkpoints are.

The current pipeline's Indic task suite remains gated. If it has not arrived, record that limit rather than calling an English-only result the best eight-language architecture. All data preparation and evaluation-set provenance remain owned by the data workstream.
