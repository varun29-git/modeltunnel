# Terminus BitNet pretraining

The maintained implementation is `terminus/`. The original 150M run remains in `archive/` and is **not a supported training entrypoint**. Its protection manifest already disagreed with six files when this audit began; the archive has not been edited.

**Launch status: not approved for the new 25-layer run.** The current candidate uses the RWKV-7 World tokenizer's 65,536 ID slots and 25 layers, with four local layers followed by one global layer. It has not been trained on multilingual data. A one-off H100 NVL synthetic full-update benchmark of the new DeepGEMM candidate measured **56.8k tokens/s** at batch 8, context 4096, over eight timed updates. The **67k target remains unmet**; the new result is 84.8% of it. The previous 30-layer, 96,000-token kernel path measured **43.9k tokens/s** versus a matched 41.5k control, with peak allocated memory reduced from 90.6 to 80.1 GB. See the [new benchmark receipt](reports/evidence/rwkv_25l_20260926/README.md), [kernel measurements and power telemetry](reports/H100_KERNEL_OPTIMIZATION_20260915.md), [earlier H100 measurements](reports/H100_OPTIMIZATION.md), and the [pretraining audit](reports/PRETRAINING_AUDIT.md).

The wider **1.500B-parameter** 25-layer configuration now uses the archived hybrid Muon policy with separate logical Q/K/V and gate/up updates. At the fixed 2,097,152-token optimizer batch, it measured **46.4k tokens/s** at microbatch 8 / accumulation 64, versus **47.5k** for the matched AdamW control. These are short synthetic runs, not a Muon quality result; see the [Muon comparison](reports/evidence/muon_25l_20260927/README.md).

The [September 20 scaling validation](reports/H100_SCALE_VALIDATION_20260920.md) adds native H100 coverage for both implementations, fixes distributed configuration drift and short/tail FlexAttention gradients, and launches a bounded full-size synthetic benchmark/recovery campaign. The newer model's 39-test H100 suite passed without skips; multilingual quality and multi-GPU scaling remain separate gates.

## Install and verify

Use a separate virtual environment. The project targets Python 3.10+ and PyTorch 2.8+; the local tested environment is recorded in the audit evidence, not presented as a certified GPU dependency lock.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -t . -v
python -m terminus.preflight --config configs/terminus_25l_64k_candidate.json --count-only
python -m terminus.benchmark loss-memory
python -m terminus.benchmark training --config configs/development.json
python scripts/reproduce_archive_findings.py
```

The CUDA tests skip on CPU. On a GPU host they exercise optional FA3, FlexAttention, DeepGEMM and custom Triton kernels; run the tests for the backend being certified rather than treating missing optional builds as a fallback. GPU implementations must be built and pinned deliberately; no installer silently downloads arbitrary kernel code during training. CI runs the CPU suite; CI execution is separate from the local results in this repository.

## Architecture and configurations

- Decoder-only, bias-free Transformer; pre-RMSNorm, GQA, parameter-free Q/K RMSNorm, RoPE, scalar SubLN, SwiGLU and tied embeddings/output head.
- Ternary projection targets with FP32 latent parameters and straight-through training; full FP8 backward uses explicit operation-specific approximate gradients. This is quantization-aware training, **not 1.58-bit training-state storage**.
- `configs/development.json`: 82,244 parameters, CPU FP32, explicit reference attention. Synthetic correctness configuration, not a multilingual tokenizer configuration.
- `configs/terminus_25l_64k_candidate.json`: **1,222,219,826 parameters**, 25 layers, width 2048, 16 query/4 KV heads, FFN 5376, RWKV-7 World vocabulary capacity 65,536, context 4096, and exactly five cycles of four local-1024 layers followed by one global layer. Token 0 is EOS and is trainable; there is no padding token. This is the current rowwise reference candidate.
- `configs/1.5b_candidate.json`: historical **1,502,210,108-parameter**, 30-layer, 96,000-token control used in the retained measurements.
- `configs/1.5b_24l_candidate.json`: **1,495,316,784 parameters**, unbenchmarked 24-layer alternative with width 2304, 18 query/6 KV heads, FFN 5632, and one global layer every four blocks (six global blocks total).
- The current reference candidate starts with AdamW, FP32 latent/state storage, BF16 autocast, block checkpointing, chunked cross entropy, FA3 attention, rowwise ternary scales, and no FP8 activation quantization. Dimensions are valid; the LR, vocabulary, depth/width, token budget and mixture are **pilot choices, not empirically optimal values**.
- FA3 is explicitly gated to Hopper SM90. FlexAttention is an alternative requiring its own preflight. The reference attention backend is for correctness, not production throughput.
- DeepGEMM is opt-in: SM90, aligned dimensions, activation FP8 and 128x128 weight groups. Changing to it changes the quantization layout and requires a new run/ablation. It is not an automatic accelerator for the rowwise baseline.

## H100 ternary–FP8 path

The custom path now supports DeepGEMM FP8 **forward, input-gradient and weight-gradient** GEMMs, with FP32 returned weight gradients. Fused Triton quantizers write exact ternary FP8 codes, independent activation/gradient orientations and TMA-aligned scales; versioned weight caching avoids repeated work across accumulation and checkpoint recomputation.

- `configs/1.5b_h100_ternary_fp8.json`: full FP8 backward, fused quantization and cached weights.
- `configs/1.5b_h100_ternary_fp8_dgrad.json`: conservative FP8 input-gradient / BF16 weight-gradient control.
- `configs/1.5b_h100_ternary_bf16_reference.json`: matched blockwise ternary / BF16 comparison control.

The full path passed native numerical gates on an Azure H100 NVL. This exposed and fixed ternary-transpose stores, quantizer rounding and RoPE dtype issues that CPU tests had not caught. FP8 changes the approximate backward arithmetic and costs about 2.61 GB extra weight-cache memory per rank. Equal-token multilingual quality is still untested. Packed-ternary inference export/runtime is not included.

The new throughput path adds fused QKV/gate-up projections, Triton RMSNorm/SwiGLU/QK-RoPE kernels, a streaming full-vocabulary loss, retained activations and optional BF16 residuals. These are explicit new-run policies:

- `configs/terminus_25l_64k_h100.json`: current 25-layer RWKV-vocabulary candidate using the selected DeepGEMM, packed MLP/QKV and split FP8 quantization path. A fresh native H100 preflight passed, and a one-off eight-update synthetic benchmark measured **56.8k tokens/s** at batch 8 / accumulation 1. Multilingual quality and sustained training remain untested.
- `configs/terminus_25l_2304d_64k_h100.json`: wider 25-layer experiment with d_model 2304 and **1,500,334,898 parameters**; it preserves the four-local/one-global schedule and RWKV vocabulary. The selected archived-policy Muon setting uses microbatch **8** and accumulation **64** on one GPU: 2,097,152 tokens per update, measured at **46.4k tokens/s**. Microbatch 4 / accumulation 128 measured 45.5k. See the [matched Muon/AdamW comparison](reports/evidence/muon_25l_20260927/README.md).
- `configs/terminus_25l_2304d_64k_h100_adamw_control.json`: the previous AdamW setting retained as a control. The fresh matched batch-8 run measured **47.5k tokens/s**; the earlier fixed-batch sweep measured 45.4k, showing short-run variability.
- `configs/terminus_25l_2304d_64k_h100_checkpointed.json`: whole-block gradient-checkpointing **AdamW** experiment with the same model geometry. Its accumulation is **8** for microbatch 64 on one GPU, also 2,097,152 tokens per update; it measured **35.8k tokens/s**. A separate memory-limit experiment fitted microbatch 90 / accumulation 1 at **34.4k tokens/s**, but that has a different global batch. See the [historical fixed-batch comparison](reports/evidence/rwkv_25l_target_batch_20260926/README.md) and [checkpointed memory sweep](reports/evidence/rwkv_25l_checkpoint_sweep_20260926/README.md).
- `configs/1.5b_h100_kernel_optimized.json`: measured historical 30-layer control with a 96,000-token vocabulary. It reached **43.9k tokens/s** over two 64-update batch-8 runs. See the [measurement report](reports/H100_KERNEL_OPTIMIZATION_20260915.md).
- `configs/1.5b_h100_24l_kernel_experimental.json`: the same 24-layer architecture as the new reference candidate, using the H100 kernel execution path. No throughput or quality result has been measured for this geometry; a fresh preflight and matched benchmark are required before launch.
- `configs/1.5b_h100_optimized.json`: 30 layers, fused operations and streaming loss, FP32 residuals.
- `configs/1.5b_h100_throughput.json`: 30 layers, BF16 residuals and larger loss chunks. **42.2k tokens/s** at microbatch 8 / accumulation 1; **41.4k** at 4 / 2 with much more memory headroom.
- `configs/1.5b_h100_optimized_graphs.json`: optional single-GPU graph capture; did not improve the fastest larger-microbatch setting.
- `configs/1.5b_h100_wide12_experimental.json`: 12 layers, wider transformer and factorized embeddings, **1,508,256,792 parameters, 50.0k tokens/s** at 8 / 1. This is a quality-unvalidated architecture experiment.

The 2048-wide H100 base config starts with accumulation 8; the sweep below adjusts it to preserve 32,768 tokens per update. For the 2304-wide model's 2,097,152-token target, keep `microbatch × data-parallel world size × accumulation = 512` sequences. On one GPU, the selected Muon setting is microbatch 8 / accumulation 64. The largest tested fitting checkpointed divisor is microbatch 64 / accumulation 8; its **AdamW** benchmark was 21.2% slower than the uncheckpointed AdamW run, so it has not been remeasured under Muon.

```bash
python -m scripts.h100_sweep --config configs/terminus_25l_64k_h100.json \
  --batches 1,2,4,8 --warmup 5 --steps 20 --output-dir artifacts/h100_sweep
python -m terminus.roofline --config configs/terminus_25l_64k_h100.json \
  --h100 nvl --target-tokens-s 67000
```

The [September 15 report](reports/H100_KERNEL_OPTIMIZATION_20260915.md) adds sustained power/clock measurements, per-component compute budgets, a matched control, and rejected kernel experiments. `fp8_wgrad_splits` selects an experimental custom FP8 reduction; it was slower than DeepGEMM and is **disabled** in the selected config. The [H100 optimization report](reports/H100_OPTIMIZATION.md) provides measured memory, exact benchmark configs, native build pins, test receipts and a dense-compute upper bound. **67k remains unmet; no physical optimum or production speed is claimed.**

```bash
python -m terminus.preflight --config configs/terminus_25l_64k_h100.json \
  --output artifacts/h100_fp8_preflight.json
python -m terminus.fp8_benchmark --config configs/terminus_25l_64k_h100.json \
  --preflight-report artifacts/h100_fp8_preflight.json \
  --warmup 3 --steps 20 --output artifacts/h100_fp8_projections.json
```

See [the FP8 path report](reports/TERNARY_FP8_PATH.md) for precision contracts, cache tradeoffs and quality gates. [September 10 evidence](reports/evidence/h100_20260910/README.md) records the measured CPU/H100 results; earlier audit and FP8 validation artifacts remain historical snapshots.

## Prepare multilingual data

The supported language codes are `en`, `hi`, `ta`, `te`, `bn`, `mr`, `gu`, `kn`, `ml`, `or`, `pa`: English plus Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Odia, and Punjabi.

Supply local JSONL with one document per record:

```json
{"text":"नमस्ते दुनिया","language":"hi","split":"train","source":"reviewed-corpus","license":"reviewed-license"}
```

Both `train` and held-out `val` must contain all eleven languages. Assign splits and review licensing, privacy, language labels, quality, near-duplicates and benchmark contamination **before** preparing data. Merely writing a license string is not a legal or quality review.

```bash
python scripts/fetch_rwkv_world_tokenizer.py
python -m terminus.tokenizer audit --corpus heldout.jsonl \
  --tokenizer data/tokenizer/rwkv-7-world/rwkv_vocab_v20230424.txt \
  --output artifacts/tokenizer_audit.json
python -m terminus.prepare_data --corpus corpus.jsonl \
  --tokenizer data/tokenizer/rwkv-7-world/rwkv_vocab_v20230424.txt \
  --output-dir data/prepared \
  --source-mix reviewed_language_mix.json --total-train-tokens "$TOKEN_BUDGET" \
  --seq-len 4096 --batch-size 1
```

`reviewed_language_mix.json` maps every language code to a positive token fraction; fractions sum to 1 and use increments of 0.001. Choose these using corpus sizes, measured tokenizer fertility and target quality, not a fabricated equal-weight default. Every curriculum phase must preserve coverage. Sampling consumes available source tokens deterministically; preparation fails if a source cannot meet its allocated budget. It does not silently loop/repeat exhausted data.

The current tokenizer is BlinkDL's `rwkv_vocab_v20230424.txt` from the RWKV-7 source tree, pinned to commit `aeda8a06fc4803655473c7e7359b1d7166e2c604` and SHA-256 `e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89`. The Hugging Face `BlinkDL/rwkv-7-world` model repository contains weights but not the tokenizer file. The byte trie has 65,529 ordinary pieces, token 0 is EOS, and the model reserves 65,536 ID slots. Terminus applies NFC before encoding and audits round-trip decoding. Inspect held-out per-language byte fallback and tokens per codepoint; those metrics are not grapheme counts or quality guarantees. SentencePiece training remains available for a separate tokenizer experiment.

Prepare new shards with the pinned RWKV vocabulary before using the 25-layer configs. Existing 96k token IDs and checkpoints are incompatible with the RWKV vocabulary and changed layer count; the manifest gate checks the tokenizer's exact vocabulary size and file hash.

Preparation packs normalized documents with EOS, removes exact normalized duplicate documents and refuses cross-split duplicates. The RWKV ID range fits `uint16`; the old 96k vocabulary uses `uint32`. IDs are never replaced by EOS. Manifest checks verify the actual local tokenizer, all shard hashes, integer ranges, language coverage and budgets. The whole-shard duplicate check is deliberately conservative: common tiny identical fragments may require larger shards and re-preparation, not disabling the check.

Production packing is a continuous causal stream with EOS, not document-isolated attention. Targets are already shifted by the loader. The model does not shift them a second time. `-100` excludes a target from loss; a reference attention key mask does not implicitly construct a target mask.

## GPU acceptance and launch

The commands in this section use the new 25-layer rowwise reference; use the new H100 config above for the DeepGEMM path. First install the chosen kernel build on the target GPU and freeze the actual environment/container. The commands below do not install FA3 or DeepGEMM for you.

```bash
python -m terminus.preflight --config configs/terminus_25l_64k_candidate.json \
  --output artifacts/h100_preflight.json
python -m terminus.benchmark training --config configs/terminus_25l_64k_candidate.json \
  --preflight-report artifacts/h100_preflight.json \
  --batch-size 1 --warmup-steps 5 --steps 20 \
  --output artifacts/h100_model_benchmark.json
CONFIG=configs/terminus_25l_64k_candidate.json \
MANIFEST=data/prepared/manifest.json DATA_ROOT=data/prepared \
OUTPUT_DIR=runs/indic-pilot PREFLIGHT_REPORT=artifacts/h100_preflight.json \
NPROC_PER_NODE=1 bash scripts/train.sh --stop-after-updates 20
```

A preflight covers backend forward/backward, local/global boundaries, cached offsets, and a reduced complete model with checkpointed/chunked loss. It is bound to model configuration, source hashes, runtime and optional kernel source/binary hashes. It is **not** a full-scale soak certificate. The full-model benchmark includes optimizer updates and accumulation, but excludes data IO, distributed communication, checkpointing and evaluation. Run the real-data pilot as well.

For the 2304-wide model's 2,097,152-token target on one H100, prepare a **separate** manifest with `--seq-len 4096 --batch-size 8` and use the Muon `configs/terminus_25l_2304d_64k_h100.json` (accumulation 64). The generic `--batch-size 1` preparation command above does not reach this target. The checkpointed memory-max alternative is still an AdamW experiment; it needs its own manifest with `--batch-size 64` and `configs/terminus_25l_2304d_64k_h100_checkpointed.json` (accumulation 8). For multi-node DDP, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NPROC_PER_NODE` on every node; data and output paths must identify consistent prepared/shared storage. `batch_size` in the manifest is **per rank**. Global update tokens are `batch_size * world_size * sequence_length * gradient_accumulation_steps`, except a smaller final update. Recalculate accumulation when world size or microbatch changes to keep the same global batch.

## Recovery and observability

```bash
CONFIG=configs/terminus_25l_64k_candidate.json \
MANIFEST=data/prepared/manifest.json DATA_ROOT=data/prepared \
OUTPUT_DIR=runs/indic-pilot PREFLIGHT_REPORT=artifacts/h100_preflight.json \
NPROC_PER_NODE=1 bash scripts/train.sh --resume runs/indic-pilot/checkpoints
```

- Resume is strict: source, model, optimizer, data, world size, schedule and runtime must agree. Old archived checkpoints have no implicit migration path. World-size changes are not elastic resume.
- `--stop-after-updates` is an absolute optimizer-update limit for a rehearsal, **not a new schedule horizon**. The manifest's usable token budget controls the horizon across invocations. The candidate budget must exceed its 300M-token LR warmup; a genuinely smaller pilot horizon needs a separately reviewed configuration. A 20-update rehearsal checks plumbing, not convergence.
- Checkpoints are immutable update-numbered files, fsynced before checksum receipts and an atomic `latest.json` commit. Loading verifies hashes before restricted deserialization. CPU Python/NumPy/Torch and per-rank CUDA RNG states are restored. Checkpoint files and their receipts must be backed up together.
- SIGINT/SIGTERM stop after a complete optimizer update and checkpoint. A hard kill can only recover from the previous committed checkpoint. A failed receipt write can leave an uncommitted file; preserve the previous good checkpoint and use a new output directory rather than overwriting evidence.
- JSONL events record real token counts, token-weighted loss, gradient norm before clipping, LR scale, quantization strength, throughput, memory and per-language validation loss/perplexity. All languages are evaluated on deterministic held-out samples, including valid partial batches.
- Finite-loss/gradient checks guard optimizer updates; nonfinite model or optimizer state cannot be checkpointed. This does not replace long-run collapse detection and native-language quality review.
- Checksums detect corruption, not authenticity. Use trusted storage, filesystem semantics that support atomic rename/fsync, capacity monitoring and independently tested off-host backups. There is no automatic checkpoint retention/deletion policy.

## Inference

`terminus.generate.generate` provides bounded cached generation with top-k/top-p sampling and optional EOS stopping. It restores model mode, rejects invalid/nonfinite logits, and projects only the last prompt state during generation. Local cache backing storage is bounded by the local window; global storage is bounded by trained context. Exceeding trained context is rejected; there is no unvalidated RoPE extension or special packed-ternary serving engine.
