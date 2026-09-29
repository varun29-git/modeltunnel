# H100 optimization measurements — September 10, 2026

**67,000 tokens/s remains unmet.** The existing 30-layer, 1.502B-parameter model reaches **42,244 tokens/s** on one Azure H100 NVL. A separate 12-layer, 1.508B-parameter architecture experiment reaches **49,993 tokens/s**. These are measured whole-model training updates, including backward, gradient clipping, fused AdamW and FP8 cache refill. Neither result establishes production throughput, convergence, or a physical optimum.

## Measurements

Each run uses 4,096-token sequences, the full 96,000-token vocabulary, 32,768 tokens per optimizer update, four CPU threads, five warmup updates and twenty timed updates. Microbatch sweeps adjust accumulation to keep update tokens fixed; each update processes eight distinct synthetic sequences. Compilation, preflight and optional profiling are outside the timing window. Throughput is total measured tokens divided by total measured time, rather than tokens divided by the median step time.

| Configuration | Parameters | Microbatch × accumulation | Tokens/s | Median update, s | Peak allocated / reserved, GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original FP8 config, with correctness fixes | 1,502,210,108 | 1 × 8 | 9,464 | 3.4393 | 29.58 / 30.67 |
| Optimized 30-layer | 1,502,210,108 | 4 × 2 | 41,424 | 0.7908 | 63.08 / 64.99 |
| Optimized 30-layer | 1,502,210,108 | 8 × 1 | **42,244** | 0.7758 | 90.63 / 91.36 |
| Experimental wide 12-layer | 1,508,256,792 | 8 × 1 | **49,993** | 0.6554 | 70.03 / 72.80 |

GB means decimal bytes / 1e9. The 30-layer batch-4 setting sacrifices about 2% throughput for substantial memory headroom and is the better starting point for a real-data pilot. Batch 8 is close to the available memory limit once production workspaces and buffers are added.

The original maintained FP8 configuration measured 9.46k after repairing native correctness failures. The user's reported 29k measurement did not come with its command/config, so it was not reproduced as a matched baseline. The 4.46× improvement applies only to the original-config row above. The optimized 30-layer result still needs **1.586×** more throughput to reach 67k; the architecture experiment needs **1.340×**.

All measured updates had finite losses and gradient norms. This benchmark excludes data IO, distributed communication, evaluation, checkpoint writes and restarts. It repeats synthetic input batches across updates and is a performance test, not a language-model quality test. Twenty measured updates are not a sustained training soak.

The [summary](evidence/h100_20260910/summary.json), exact derived configurations and compressed original receipts are in [the evidence directory](evidence/h100_20260910/README.md). Every benchmark/preflight source hash was compared with the retained `terminus/*.py` files before collecting the evidence.

## Comparison against hardware limits

The actual GPU is **one H100 NVL**, not an SXM GPU or an NVL pair: 132 SMs and 99,456,909,312 bytes of memory reported by PyTorch. The power limit was already 400 W; this work did not change power or clocks. [Hardware receipt](evidence/h100_20260910/hardware.csv).

The analytical model uses per-GPU advertised **dense** peaks: 1,671 TFLOP/s FP8, 835.5 TFLOP/s BF16 and 3.9 TB/s HBM for NVL. Multiplication and addition count as two FLOPs. Ternary zeros do not confer NVIDIA's 2:4 structured-sparse throughput. The old planning document used sparsity-assisted figures as dense BF16 throughput and also contains a 1,000× GPU-hour FLOP conversion error; its budget estimates must not drive the run.

For each token the model counts projection forward/dgrad/wgrad, the actual tied-head width, embedding projections, valid causal local/global attention pairs, block-checkpoint replay when enabled, head replay when enabled and FlashAttention's QK recomputation during backward. FP8 work and BF16 work are divided by their respective peaks. A separate HBM lower bound includes optimizer/clipping and cached-weight refill traffic. It omits most activation/operand traffic, non-GEMM math, launch gaps, tile padding and communication, so this is deliberately an **optimistic upper bound**, not an achievable throughput forecast.

| Path | Ideal compute time, µs/token | Optimistic ceiling, tokens/s | Measured useful compute fraction | Compute fraction needed at 67k, including recomputation |
| --- | ---: | ---: | ---: | ---: |
| Original FP8 config | 9.624 | 103,909 | 6.7% | 64.5% |
| Optimized 30-layer | 7.259 | 137,755 | 30.0% | 48.6% |
| Experimental wide 12-layer | 6.429 | 155,556 | 31.6% | 43.1% |

The useful compute fraction excludes checkpoint/attention recomputation; the last column includes it. The optimized 30-layer row performs about 7.833 GFLOPs/token in FP8 and 2.010 GFLOPs/token in useful BF16 operations. Its measured time is about 23.67 µs/token, versus 14.93 µs/token required at 67k and a 7.26 µs/token compute-only floor. These numbers show an unresolved efficiency gap. They do **not** prove that the remaining gap is recoverable, nor that memory bandwidth is saturated.

Run the same calculation for another H100 variant or config:

```bash
python -m terminus.roofline \
  --config reports/evidence/h100_20260910/optimized_30_b8_config.json \
  --h100 nvl --batch-size 8 --target-tokens-s 67000 \
  --measured-tokens-s 42244.19169418488
```

Hardware references: [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [H100 NVL product brief](https://resources.nvidia.com/en-us-tensor-core/nvidia-h100-nvl-gpu). Marketing tables that include structured sparsity must be divided by two for this dense path; pair totals must also be divided by two for a single NVL GPU.

## Retained implementation changes

- **Eliminate avoidable recomputation.** Optimized configs retain block activations. A custom first-order linear cross-entropy autograd function computes each vocabulary chunk and its derivatives once, then frees its logits. It casts the tied head once and accumulates its weight gradient directly into FP32 storage. The entire vocabulary still contributes to the loss; there is no sampled softmax. Triton kernels tile the vocabulary reduction and overwrite logits with derivatives. Ignored targets and upstream loss scaling are supported.
- **Fuse projections and pointwise work.** Q/K/V use one projection, and MLP gate/up use one projection. Triton implements RMSNorm and SwiGLU forward/backward. Another kernel fuses Q/K normalization with RoPE and writes the FA3 sequence-major layout directly. FA3 accepts the supported strided QKV views without redundant whole-tensor copies.
- **Repair and accelerate native quantization.** Real GPU testing found that a combined ternary transpose store left lanes unwritten on Triton 3.4. The transpose is now an exact cached copy of the forward codes/scales. Scale construction follows the reference's reciprocal multiplication; a small rounding difference had crossed FP8 rounding boundaries. Directional activation/gradient quantization uses reciprocal multiplication with an FMA residual correction instead of a full IEEE division for every element. Native reference gates pass for these changes.
- **Fix RoPE dtype promotion.** FP32 residual-derived cosine/sine values previously promoted BF16 Q/K to FP32, causing FA3 to reject the original path. RoPE now uses the projection dtype.
- **Make memory/precision choices explicit.** `residual_dtype="bf16"` lowers activation traffic and permits larger microbatches. `loss_backend`, `fused_pointwise`, `fuse_projections` and optional `embedding_dim` are explicit model policies. Defaults preserve the old choices; optimized configs select the new policies.
- **Add optional single-GPU CUDA graphs.** Fixed-shape microsteps replay forward/backward into persistent FP32 gradient buffers, refresh quantized weights on replay and accept changing inputs, targets and loss scales. Optimizer, clipping and finite checks remain outside capture. Smaller tail shapes run eagerly. Graphs require no block checkpointing/dropout and are rejected with multi-rank training. They did not improve the best larger-microbatch setting, so the measured throughput config leaves them disabled.
- **Make measurements reviewable.** The benchmark records complete optimizer/training config, source/runtime identities, per-update timings, allocated/reserved memory and the dense hardware bound. Profiling runs separately from timed updates. `scripts/h100_sweep.py` preserves update tokens and writes the exact config for every microbatch. `scripts/h100_components.py` supplies isolated CUDA-event measurements when CUPTI is unavailable.

An alternate FP8 scaling policy did not improve throughput and was removed. A structured-sparse probe improved one isolated forward GEMM after tuning, but required a different pruning policy and did not establish full-model throughput or quality. It is not part of the retained model. No per-kernel speedup is substituted for a whole-model measurement.

## Architecture and numerical scope

`configs/1.5b_h100_throughput.json` retains the 30-layer model's parameter count, vocabulary, context, heads, FFN width and local/global attention schedule. It changes residual precision, GEMM grouping and loss accumulation arithmetic. Fusing QKV/gate-up changes state-dictionary names and initialization traversal. This is a new-run configuration; no implicit migration from an existing checkpoint is provided.

`configs/1.5b_h100_wide12_experimental.json` changes depth to 12, width to 3,072, heads to 24 query / 6 KV, FFN width to 10,368, and uses 768-dimensional tied embeddings with input/output projections. It retains the 96k vocabulary, 4,096 context and local-1,024 attention, with a global layer every fourth layer. Fewer sequential layers and a smaller BF16 head reduce overhead while reallocating parameters to the transformer. **The factorized head and shallower architecture may reduce quality or require more training tokens.** Parameter count alone does not establish equivalent capacity.

Kernel comparisons cover the explicit equations, including BF16 rounding boundaries. The reduced complete-model preflight compares checkpointed versus retained activations using the **same selected head arithmetic and chunk shape**. Comparing a differently chunked/full head through many FP8 gradient quantizers can amplify small rounding changes; the preflight does not certify full-model gradient equivalence to the previous arithmetic. Independent head, norm, SwiGLU, QK/RoPE and FP8 linear checks remain in place. Attention cases with a mathematically zero reference gradient use an absolute tolerance only for that zero-reference case.

Neither candidate has an equal-token multilingual quality result. A useful next acceptance run must compare held-out loss by language, clipping/gradient behavior and training progress at equal tokens/global batch. Multi-node scaling, real-data throughput and recovery also remain unmeasured for these new configurations. They cannot be counted as guaranteed production savings.

## Runtime and reproduction

Measurements used Python 3.10.12, PyTorch 2.8.0+cu126, Triton 3.4.0, driver 595.58.03 and DeepGEMM 2.6.1 at commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`. The [pip snapshot](evidence/h100_20260910/runtime-pip-freeze.txt) records the reused environment; preflight receipts also hash the actual loaded native sources/binaries.

FA3 was built separately from tag v2.8.3, commit `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, with CUTLASS `dc4817921edda44a549197ff3a9dcf5df0636e7b`. The build enables BF16 head dimension 128, local attention and backward. Other head dimensions are not enabled in this particular FA3 build. The explicit build script installs into an isolated target directory:

```bash
export RUNTIME_ROOT="$PWD/artifacts/h100-runtime"
export PYTHON_BIN=/path/to/tested/cuda/venv/bin/python
FA3_CUDA_HOME=/usr/local/cuda-12.6 bash scripts/build_h100_fa3.sh
export PYTHONPATH="$RUNTIME_ROOT/fa3-site${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-13.2
export DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
export DG_JIT_CACHE_DIR="$RUNTIME_ROOT/deepgemm-jit"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

FA3 was compiled using CUDA 12.6. The installed DeepGEMM source failed JIT compilation with CUDA 12.6 (`asm` constraint `q`); the successful measurements used the existing CUDA 13.2 compiler for DeepGEMM JIT. Do not infer the JIT compiler version from `torch.version.cuda` alone.

Reproduce the memory-headroom candidate with its **derived batch-4, accumulation-2 config**:

```bash
"$PYTHON_BIN" -m terminus.preflight \
  --config reports/evidence/h100_20260910/optimized_30_b4_config.json \
  --output artifacts/h100_30_preflight.json
"$PYTHON_BIN" -m terminus.benchmark training \
  --config reports/evidence/h100_20260910/optimized_30_b4_config.json \
  --preflight-report artifacts/h100_30_preflight.json \
  --batch-size 4 --warmup-steps 5 --steps 20 --threads 4 \
  --output artifacts/h100_30_b4.json
```

For the measured fastest existing architecture, substitute `optimized_30_b8_config.json` and `--batch-size 8`; for the architecture experiment use `wide12_b8_config.json` and rerun its preflight. Base configs under `configs/` start with accumulation 8 for a batch-1 run/sweep. Passing batch 8 to a base config without reducing accumulation changes the update batch eightfold.

The sweep handles that adjustment automatically:

```bash
"$PYTHON_BIN" -m scripts.h100_sweep \
  --config configs/1.5b_h100_throughput.json --batches 1,2,4,8 \
  --warmup 5 --steps 20 --output-dir artifacts/h100_30_sweep
```

The Azure confidential-computing mode rejects CUPTI with `CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED (41)`. CPU profiler events and CUDA-event component measurements were available; a GPU kernel timeline or achieved HBM-bandwidth measurement was not. No CPU-only trace is presented as GPU profiling evidence.

## Validation evidence

- CPU suite: **95 tests run, 11 CUDA tests skipped, no failures**, including existing recovery/DDP tests and new loss gradcheck, tied-head accumulation, projection fusion, factorized embeddings/caching and roofline accounting. [Log](evidence/h100_20260910/cpu_tests.log).
- H100 selected regression suite: **7 tests passed**, covering the new loss/pointwise/graph kernels and both native quantizer tests. [Log](evidence/h100_20260910/gpu_tests.log).
- Full native preflights passed for original FP8, optimized 30-layer, experimental 12-layer and optional graph configs. These exercise actual configured projection dimensions, full FP8 forward/dgrad/wgrad, attention boundaries/cached offsets, optimized equations and reduced complete models. Graph preflight matched eager results across two optimizer updates and two microsteps per update with changing inputs and scales.
- Four full-size benchmark runs passed, with the measurements above. No long-run multilingual, full GPU-backend matrix, multi-node, or production soak pass is claimed.

The Azure VM was used through an isolated workspace and runtime build directory. The original projects were preserved, and the VM was returned to its initial deallocated state after evidence collection.
