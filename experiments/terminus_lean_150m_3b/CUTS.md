# Cuts being tested

| Component | Control | Lean | Why cut it |
| --- | --- | --- | --- |
| FFN SubLN after SwiGLU | scalar RMSNorm | removed | Preserve token-dependent FFN magnitude and eliminate an extra reduction per layer. |
| Attention SubLN before O projection | scalar RMSNorm | removed | Preserve attention-output magnitude and eliminate an extra reduction per layer. |
| Q/K unit-RMS normalization | parameter-free RMSNorm on each | removed | Let learned Q/K magnitudes influence attention selectivity and eliminate Q/K reductions. |

Both candidates retain the two **learned pre-block RMSNorms** and final RMSNorm, all 25 layers, SwiGLU, tokenizer, projections, local/global pattern, optimizer family and FP8 activation arithmetic. That isolates the three cuts. The fused QKV kernel has a no-normalization branch; the packed SwiGLU kernel already supports no SubLN. They require native SM90 numerical verification before use.

If lean loses, the next inexpensive diagnostic is to separate the cuts using `use_attention_subln`, `use_ffn_subln` and `qk_norm_mode`. Do not add a learned temperature, new gains, or wider layers to rescue this first experiment: those are separate hypotheses. The lean candidate may need its own residual initialization or learning-rate bracket if diagnostics show a scale mismatch, but such a retune must be recorded as another trial and compared at equal tokens and GPU-hours.

Gradient clipping is **not** cut in this first pair. Removing it would confound a normalization comparison, and the current trainer still computes the gradient norm when `grad_clip=0`. FP8 activations are a hard requirement for this H100/DeepGEMM comparison. Capacity cuts such as fewer global layers or KV heads are deferred until the normalization result is known.
