# Lean architecture cuts in this dense FP8 run

| Removed component | Why it is cut | Key instability signal |
| --- | --- | --- |
| FFN SubLN after SwiGLU | Eliminates one reduction per block and lets FFN magnitude carry information. | FFN/residual RMS drift by layer and update-to-weight spikes. |
| Attention SubLN before O projection | Eliminates another reduction per block. | Attention-output/residual RMS drift. |
| Q/K unit-RMS normalization | Restores learned Q/K scale and removes two head-wise reductions. | Q/K RMS, sampled logits, attention entropy and gradient spikes. |

Two learned pre-block RMSNorms and the final RMSNorm remain. The 25 layers, 4:1 local/global pattern, tokenizer, FP8 activations, optimizer family and optimized kernels remain as configured. There is **one lean dense run**, with no control or QAT job in this folder. If it is unstable, the records can distinguish likely causes; attribution to a particular cut requires a later controlled trial. Gradient clipping remains at 1.0 and every clip coefficient is recorded. Native SM90 numerical preflight is required before launch.
