# Maintained Terminus 1.5B: archived Muon policy and H100 comparison

The archived 150M run used Muon for separate Q, K, V, gate, up, down and attention-output matrices with auxiliary AdamW groups. The maintained 25-layer model retains fused QKV and gate/up forward projections. Its optimizer now applies the pinned upstream Muon update separately to logical row slices of those fused weights, with the archive's `muon_lr=0.003`, auxiliary `adamw_lr=0.003`, `norm_lr=0.0002`, `eps=1e-10`, momentum 0.95 and weight decay 0.1. The pinned upstream commit is `f98f1cacc0263b04290753e32be8d498c1efc806`. The selected config is `configs/terminus_25l_2304d_64k_h100.json`; `configs/terminus_25l_2304d_64k_h100_adamw_control.json` preserves the AdamW comparison.

One Azure H100 NVL ran the same 1,500,334,898-parameter, 25-layer DeepGEMM/FA3 model at context 4096, no gradient checkpointing and **2,097,152 synthetic tokens per optimizer update**. Each case used one warmup and two timed full forward/backward/optimizer updates. AdamW and Muon runs used identical model, source and runtime fingerprints. The allocator used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

| Optimizer | Microbatch | Accumulation | Throughput, tokens/s | Timed step seconds | Peak allocated |
| --- | ---: | ---: | ---: | --- | ---: |
| AdamW control | 8 | 64 | 47,537 | 43.66, 44.57 | 81.60 GB |
| Muon | 8 | 64 | **46,395** | 45.25, 45.16 | **76.21 GB** |
| AdamW control | 4 | 128 | 46,017 | 45.40, 45.75 | 55.19 GB |
| Muon | 4 | 128 | 45,493 | 46.16, 46.04 | 49.79 GB |

The highest observed Muon rate is batch 8 / accumulation 64. In a separate, untimed batch-8 diagnostic update, the optimizer phase was **271.39 ms for Muon** versus **17.02 ms for fused AdamW**; the additional 254.37 ms is about 0.56% of a 45.2-second Muon update. The observed batch-8 throughput was 2.40% below this short AdamW control, but that whole difference cannot be assigned to Muon: the previous AdamW batch-8 receipt measured 45,395 tokens/s under the same nominal geometry, illustrating run/clock variation. Muon reduced allocated peak by about 5.40 GB at both microbatches. CUDA-event phase times include launch gaps and came from one separate update; they must not be added to the timed steps.

GPU preflight passed for the changed source. Local tests checked exact two-step updates against separate matrices in the pinned upstream Muon optimizer and a saved-state resume. The full-size synthetic benchmark completed with finite loss, gradients and optimizer state. It excludes data loading, communication, validation, checkpoint writing and long-run language-model quality. The optimizer policy and learning rates are copied from the archive; the current 1.5B model still uses its FP8/DeepGEMM path and gradient clipping at 1.0, while the archived 150M run used a different model and had clipping disabled. The archived learning rates are not a validated 1.5B quality recipe. The H100 was deallocated after the comparison.

Receipts: [batch-8 AdamW](adamw.json), [batch-8 Muon](muon.json), [batch-4 AdamW](adamw_b4.json), [batch-4 Muon](muon_b4.json), and [GPU preflight](preflight.json). The generated batch-4 configs are alongside the receipts. Each benchmark JSON contains the exact optimizer/training settings, memory peak, timing samples, and source/runtime hashes.
