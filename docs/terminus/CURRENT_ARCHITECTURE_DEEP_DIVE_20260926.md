# Current architecture deep dive

_Inspected 2026-09-26. This is a code-and-artifact inventory, not a new benchmark or quality result._

## Executive reading

This checkout is a research/training repository for an approximately 1.5B-parameter multilingual ternary language model, but it is not one monolithic system. It contains three materially different strata:

1. **`terminus/` — the maintained, supported pretraining stack.** This is the root package installed by the root `pyproject.toml`, documented by the top-level README, launched through `scripts/train.sh`, and treated as the current route for a 30-layer decoder-only BitNet-style model. It owns the multilingual tokenizer/packing pipeline, strict data-manifest validation, model, DDP training, preflight, benchmark, checkpoint, and cached Python inference paths.
2. **`terminus_next/` — an isolated architecture challenger and deployment prototype.** It has its own Python package (`ternary_lm`), its own `pyproject.toml`, configurations, data format, trainer, checkpoint format, tests, and a C++17 mmap/native runtime. It deliberately imports nothing from `terminus/`. Its main candidate changes the transformer topology to a lower local self-attention stack plus an upper stack cross-attending to shared memory. It is executable and has CPU/H100 acceptance evidence, but is not declared the supported final training entrypoint.
3. **`archive/150m-30btokens-run-original/` — a protected historical run.** It contains an earlier 150M run's source snapshot, configs, logs, checkpoints/evaluation receipts and a protection lock. The root README explicitly says it is not supported and must not be used as a current training entrypoint. Current code does not extend it.

The checkout is therefore a **controlled comparison platform**, not a single completed foundation-model release. It has substantial correctness/recovery/precision engineering, but no checked-in real multilingual corpus, trained 1.5B checkpoint, multilingual quality result, or production approval. The top-level conclusion remains: **do not start the final 1.5B run yet**. The selected maintained H100 path is still below its 67k tokens/s target; the alternate `terminus_next` path is much slower in its current full-size synthetic implementation. Both need real-data quality and multi-GPU production validation.

## Repository map and ownership

```text
root package: terminus-bitnet (Python >=3.10, Torch 2.8–<3)
|
+-- terminus/                         maintained pretraining system
|   |-- tokenizer.py / prepare_data.py multilingual corpus -> verified packed tokens
|   |-- data.py / integrity.py         manifest, shards, deterministic routing, validation
|   |-- model.py / quantization.py     maintained decoder-only ternary model
|   |-- kernels*.py / pointwise*.py    optional CUDA/Triton/FA3/DeepGEMM acceleration
|   |-- loss*.py / optim.py            full-vocabulary CE and AdamW/Muon contracts
|   |-- train.py / checkpoint.py       DDP, schedule, events, exact recovery
|   |-- preflight.py / benchmark.py    numerical gates and synthetic throughput
|   `-- generate.py / roofline.py      cached generation and analytical ceilings
|
+-- configs/                           maintained-path controls and H100 variants
+-- tests/                             maintained CPU/GPU correctness suite
+-- scripts/                           explicit launch, pin/build, and H100 campaign helpers
|
+-- terminus_next/                     independent candidate and phone-serving prototype
|   |-- ternary_lm/                    standalone training/eval/export package
|   |-- native/runtime.cpp             C++17 packed-model runner
|   |-- configs/                       shared/conventional/deep and ablation definitions
|   |-- tests/                         standalone Python/C++ validation
|   `-- tools/                         native build, smoke, H100, DDP, export support
|
+-- reports/                           prior design, audit, and H100 reports
+-- reports/evidence/ and artifacts/   receipts, logs, benchmarks, snapshots (not runtime code)
+-- plans/                             token/compute planning documents
`-- archive/...                        immutable historical 150M evidence and source
```

`reports/`, `plans/`, `artifacts/`, and `reports/evidence/` are evidence and decision-support material rather than packages the trainer imports. Some `artifacts/` copies are historical working outputs; newer evidence also lives under `reports/evidence/` and `terminus_next/evidence/`. The many JSON/JSON.GZ files are not a dataset: they record parameter counts, numerical preflights, timing, source hashes, H100 telemetry, and research analyses.

The working tree is deliberately **not clean** as inspected. The recent September 20 scaling work modifies/adds CI, root attention/preflight code, `terminus_next` attention/loss/training/H100 tools and their evidence. This report describes the filesystem currently present, including those uncommitted changes, rather than claiming that the last Git commit alone represents the system.

## The maintained path (`terminus/`)

### Primary data/control flow

```text
split-labelled local JSONL
  → NFC + SentencePiece encoding + EOS + exact-document deduplication
  → language/split token shards + tokenizer copy + manifest + SHA-256s
  → manifest validation + deterministic source schedule
  → memmapped sequential causal blocks (inputs=t[0:-1], targets=t[1:])
  → BitNetLlama forward / chunked or streaming full-vocabulary CE
  → accumulated DDP gradients → clip → optimizer update
  → event JSONL / validation by language / immutable checkpoint receipt
```

The root `scripts/train.sh` is intentionally only a launcher. It requires reviewed `CONFIG`, `MANIFEST`, `DATA_ROOT`, `OUTPUT_DIR`, and `PREFLIGHT_REPORT` environment variables, then invokes `torch.distributed.run -m terminus.train`. It uses rank/world-size environment variables for single-node or multinode DDP; the actual policy and verification live in Python.

### Model contract and candidate shape

`BitNetLlamaConfig` in `terminus/model.py` is the architectural and kernel-policy contract. It rejects invalid dimensions and incompatible execution options before allocation. The principal baseline candidate (`configs/1.5b_candidate.json`) is:

| Property | Candidate value | Meaning |
|---|---:|---|
| Vocabulary | 96,000 | tied input embedding/output head; pad token ID 0 |
| Transformer width | 2,048 | 16 query heads, therefore 128-dimension heads |
| Layers | 30 | six local/global cycles |
| KV heads | 4 | grouped-query attention; 4 query heads share one KV head |
| FFN width | 5,376 | SwiGLU hidden width |
| Context | 4,096 | trained maximum and hard generation limit |
| Attention schedule | local 1,024 for four layers, global on each fifth | local layers 1–4, 6–9, …; global layers 5,10,…,30 |
| Attention backend | FA3 | explicitly Hopper SM90 gated in production |
| Projection quantization | ternary targets, rowwise absmean scales | FP32 latent weights remain trainable |
| Activations / GEMM | BF16 activations, ordinary Torch linear | no FP8 activations in baseline |
| Memory controls | block checkpointing, 256-token checkpointed CE chunks | recompute trades time for memory |
| Optimizer | AdamW | latent/state are FP32, LR 3e-4 for matrices and 1e-4 for norms |

The model is a bias-free, decoder-only pre-RMSNorm transformer. Its execution is:

1. A token ID enters a learned embedding. Embeddings and residual stream are normally FP32 (or BF16 only if the configuration explicitly chooses `residual_dtype="bf16"`). Optional input/output embedding projections permit a factorized embedding dimension, but the baseline uses full-width tying.
2. Each block applies channelwise pre-RMSNorm, attention, residual addition, channelwise pre-RMSNorm, FFN, then a second residual addition. Residual projections are initialized at `1/sqrt(2*n_layers)` in the 30-layer config (0.12909944), damping the residual branch initially.
3. Attention uses Q, K, V ternary projection matrices plus a ternary output projection. Q has 16 heads while K/V have four. Q and K get parameter-free headwise RMS normalization and RoPE. The attention output can receive scalar SubLN before the output projection.
4. The FFN uses SwiGLU: separate/packed gate and up projections, `SiLU(gate) * up`, optional scalar SubLN, then down projection. A `relu2` alternative exists but is not selected.
5. Final RMSNorm feeds a tied linear head using `tok_embeddings.weight`. The default full-vocabulary objective is ordinary next-token cross entropy, not a sampled/adaptive vocabulary head.

No dense bias vector, learned Q/K norm scale, relative position bias, dropout in production, padding-aware production path, MoE, or packed-ternary serving engine is present in this maintained stack. Padding masks are explicitly confined to reference SDPA correctness use; production assumes packed causal streams and zero dropout.

### Ternary parameterization and precision are not 1.58-bit training

Every `BitLinear` retains a floating latent matrix `W`. It derives a scale `alpha` from mean absolute values over either rowwise width-128 groups or 128×128 blocks, then forms integer codes:

```text
q = clamp(round(W / alpha), -1, 1)
effective W = W + strength * stop_gradient(alpha*q - W)
```

This is an identity straight-through estimator: forward values become ternary targets, while the backward path sees the latent tensor. A scalar `quantization_strength` enables a curriculum on the regular Torch backend; DeepGEMM requires strength exactly one. The baseline calls this scale layout `rowwise`; hardware FP8 configurations require 128×128 `blockwise` scales. Thus a kernel change is not treated as a harmless toggle: it changes the effective quantization grouping and must be a new run/ablation.

The current `BitNetLlama` has no separately learned alpha parameters in this maintained path (`alpha_parameters()` returns empty): its alpha is derived from current weight values. Its training state is therefore not compact ternary. The checked-in parameter evidence says the roughly 1.502B candidate needs about 12.02 GB for FP32 parameters plus gradients and 24.04 GB for AdamW FP32 latent/state before activations, workspaces, caches, or communication. Ternary is a forward/deployment arithmetic target, not the storage format used for optimizer state.

When `quantize_activations=true`, activations are blockwise FP8 E4M3 quantized/dequantized through an STE. The optional DeepGEMM path has stricter semantics: 128-aligned dimensions, full FP8 activation quantization, blockwise ternary scales, E4M3 codes and FP32 scales. It can select:

- `bf16`: effective FP8 forward, BF16 input/weight gradient math;
- `dgrad`: FP8 input-gradient GEMM, BF16 weight-gradient;
- `full`: FP8 forward, input-gradient, and weight-gradient GEMMs; the weight gradient returns FP32.

The custom autograd function quantizes independent row/column orientations for inputs and gradients. It can cache versioned quantized weight operands across accumulation/checkpoint recomputation; a cache key includes tensor identity/data pointer/version/shape/stride/device/dtype, orientation, quantizer policy, and inference-mode status. It is cleared after every optimizer update. This cache reduces repeated quantization but consumes material HBM (the README reports about 2.61 GB/rank for the full path). The selected speed path uses a Triton split quantizer; the basic interfaces explicitly reject missing DeepGEMM or unsupported SM90 hardware rather than silently fall back.

### Attention, cache, and kernels

There are three selectable attention implementations:

- `sdpa_reference`: PyTorch scaled-dot-product attention. It can build an explicit causal/local/padding mask and is the CPU/correctness reference; it is forbidden as a production backend.
- `fa3`: pinned FlashAttention-3 interface. It requires CUDA BF16/FP16 operands, Hopper SM90, no padding mask/dropout, and correct bottom-right cached alignment. It receives a causal flag and local window (`window_size=(local_window-1,0)`) rather than a dense mask. FA3's exact dependency/build is an explicit external installation task (`scripts/build_h100_fa3.sh`); it is not installed by `pip` or downloaded at training time.
- `flex`: compiled PyTorch FlexAttention. It builds a cached 128-tile causal/local block mask. Recent changes pad physical query/key/value lengths to a multiple of 128 and mask logical bounds, because a pinned PyTorch 2.8 runtime exposed incorrect short/tail K/V backward behavior. Stored cache capacity is not padded.

For cached inference, `InferenceKVCache` owns one `LayerKVCache` per layer. A local layer owns only a 1,024-token bank; a global layer owns up to the 4,096-token context. Cache append uses preallocated storage while it fits; when a local window overflows, it produces concatenated attention operands and commits only the trailing capacity. RoPE uses the cache position and rejects positions beyond trained context. This supports batch-consistent incremental generation but no context extension. `generate()` accepts one prompt, samples top-k/top-p/temperature or greedy output, projects only the final prompt state on prefill, restores training/eval state, stops on optional EOS, and rejects nonfinite logits/out-of-range budgets.

### Loss implementations

All maintained modes calculate exact full-vocabulary CE over the tied 96k head; none uses sampling. They differ only in retained memory/recomputation.

- **ordinary logits**: materializes logits if `return_logits=True`.
- **checkpoint loss**: splits flattened tokens into chunks and checkpoints each `linear(hidden, head)` + CE calculation. Backward replays the logits GEMM. Baseline config uses this at 256-token chunks.
- **streaming loss**: a custom first-order autograd function casts the head once, processes logits chunks during forward, accumulates total CE and precomputes hidden/head gradients. Backward applies only the incoming scalar. Its head gradient accumulates FP32 (with CUDA `addmm` when possible); it retains gradients rather than logits. The root path supports Torch or a Triton CE kernel.

Loss is represented internally as a **sum** plus a number of non-ignored targets. Training divides each microstep sum by the exact global update-token count, not merely by number of microbatches. `-100` is the only loss-ignore sentinel and targets are already shifted by the data loader; the model does not shift a second time.

### Data, tokenizer, and multilingual invariants

`terminus/tokenizer.py` trains a SentencePiece unigram/BPE model only from `split="train"` documents. It normalizes Unicode itself with NFC, requires valid Unicode without NUL/unpaired surrogates, then configures SentencePiece identity normalization, byte fallback, preserved whitespace/joiners, no dummy prefix, and special IDs pad=0/EOS=1/BOS=2/UNK=3. It samples bounded per-language reservoirs deterministically and requires all 11 supported languages: English (`en`) plus Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Odia, and Punjabi. Audit output reports per-language document/token/bytes/codepoint/fallback metrics. These are tokenizer fertility/coverage measures, not quality scores or grapheme counts.

`prepare_data.py` consumes local JSONL records with text, language, explicit `train`/`val` split, source, and license. It:

- NFC-normalizes, skips blank text, and hashes normalized text in SQLite;
- rejects exact normalized train/validation duplicates and removes same-split duplicates;
- appends EOS after every retained document;
- writes continuous per-language, per-split binary streams (not document-isolated attention); and
- chooses `uint16` only up to 65,536 vocabulary IDs, otherwise `uint32` for the 96k vocabulary.

It records a copied tokenizer, a tokenizer fingerprint, shard SHA-256, source licenses, normalization/packing/dedup provenance, document counts, and duplicate removal count in `manifest.json`. It refuses a nonempty output directory, but the report should not confuse creation safety with a full legal/privacy/quality audit of source records.

`validate_manifest()` reopens every shard and enforces, among other things: sequence length equals model context; train/val splits both exist; unique filesystem identities and unique whole-shard digests; matching shard checksums; correct dtype/range for every token; language labels; frozen tokenizer hash and special IDs; training and validation coverage of all 11 languages; at least one complete sequence per language; NFC/continuous-EOS/dedup/license provenance; and an explicit mixture covering every language in every phase. Whole-shard digest checking is deliberately conservative rather than a substitute for semantic near-duplicate or benchmark-contamination review.

`TrainingTokenRouter` turns source mixtures into a deterministic 1,000-example weighted-fair schedule with weights constrained to positive 0.001 increments summing exactly to one. Source streams remain sequential within a source, there is no silent repetition if a source is exhausted, and it can also implement phase-specific schedules. The distributed loader assigns globally interleaved examples to ranks and forms causal blocks of `seq_len+1` tokens. `PrefetchingTokenLoader` adds a bounded producer thread/queue for batches. Validation evaluates deterministic, evenly spaced full sequences separately per language and supports valid partial final batches.

### Training, schedule, DDP, and observability

`TrainConfig` supports CPU FP32 only as a correctness reference; production requires CUDA BF16 and an explicit non-reference attention backend. A production launch validates a GPU preflight receipt whose model/config/source/runtime identity matches the current configuration. It will reject a mismatch in selected FP8 policy, optimized kernels, or CUDA graph requirement.

At startup, the trainer validates data and creates a resume contract containing model config, essential training config, optimizer config, validated manifest report, world size, hash of every root package source file, and runtime/native-package identities. All DDP ranks gather contract fingerprints before work proceeds. It uses NCCL on CUDA or Gloo on CPU, `DistributedDataParallel(..., gradient_as_bucket_view=True)`, and `no_sync()` for all but the final accumulation microstep. World-size changes are deliberately not elastic resume.

The actual update loop is:

```text
load deterministic next batches
→ select cosine-with-warmup LR from tokens after this update
→ optionally select ternary-strength curriculum
→ zero gradients
→ [autocast forward → summed CE × world_size / exact update targets → backward] × accumulation
→ finite gradient check + global-norm clip
→ optimizer step → invalidate FP8 operand caches
→ immutable event/checkpoint/optional validation
```

The candidate's nominal 4,096-token microbatch and accumulation=8 means 32,768 tokens/update on one rank. `stop-after-updates` is an absolute update limit for a rehearsal; it does not shorten the manifest horizon or silently change the LR schedule. The schedule warms 300M tokens then cosine-decays to 10% of base LR. Production refuses a final run whose usable data budget is no larger than warmup.

Events are append-only JSONL with loss, token counts, pre-clip norm, LR/quantization scales, throughput, peak allocated GPU memory, and per-language validation loss/perplexity. SIGINT/SIGTERM requests a stop after a complete optimizer update and triggers checkpointing. It cannot make a hard kill transactional mid-update.

### Checkpoint/recovery integrity

The maintained checkpoint format is schema version 1. On save, rank zero gathers each rank's Python/NumPy/Torch/CUDA RNG state, rejects nonfinite floating model/optimizer values, writes a unique `step_XXXXXXXX.pt` temporary with `torch.save`, flushes/fsyncs, atomically renames it, fsyncs the directory, writes a JSON SHA-256/size/contract/state receipt, then atomically updates `latest.json`. Existing checkpoint names are never overwritten.

Resume first verifies the receipt's name confinement, size and SHA-256, then restricted `torch.load(weights_only=True, mmap=True)`, schema, exact contract, receipt state and per-rank RNG count. It restores optimizer/data cursor/RNG. This is robust against ordinary corruption and many accidental configuration errors, but checksums do not authenticate malicious storage; atomicity also depends on filesystem semantics and external backups remain a deployment responsibility.

### H100 variants and performance status

`configs/` defines an experiment matrix rather than interchangeable presets:

| Variant family | Main difference |
|---|---|
| `1.5b_candidate.json` | rowwise ternary + BF16 Torch baseline; checkpointing; FA3 |
| `ternary_bf16_reference` | blockwise ternary BF16 control |
| `ternary_fp8` / `ternary_fp8_dgrad` | DeepGEMM FP8 forward/full or dgrad backward |
| `optimized` / `throughput` | retained activations, streaming/Triton loss, fused projections/pointwise; residual precision changes |
| `kernel_optimized` | selected 30-layer H100 speed policy: packed QKV/SwiGLU, fused RMSNorm/RoPE, `triton_split` FP8 quantizer, BF16 residual, full FP8 backward |
| `optimized_graphs` | optional single-GPU captured microsteps |
| `wide12_experimental` | approximately same parameter count but 12 layers × 3,072 width and factorized embeddings; quality unvalidated |

The latest documented long-ish selected maintained-model measurement is 43.9k tokens/s against a matched 41.5k control and 80.1 GB versus 90.6 GB peak allocation for a batch-8 configuration; the intended 67k target is unmet. The Sept. 20 shorter full-size snapshot used batch 4 / accumulation 2 (still 32,768 targets/update) and reports 45,219 tokens/s, 57.78 GB allocated / 59.35 GB reserved. These are synthetic, single-H100, optimizer-inclusive measurements. They exclude data IO, distributed communication, checkpoints, evaluation and real-data behavior; they are not delivered multilingual throughput or quality evidence.

`preflight.py` is an important gate but not a soak test. It compares local/global/cached attention outputs and gradients against SDPA over boundary lengths, checks FP8/DeepGEMM arithmetic when chosen, exercises reduced model/loss and optimized kernels, optionally checks CUDA graph replay, fingerprints sources/runtime, and records the scope explicitly. `roofline.py` produces a deliberately optimistic dense H100 analytical ceiling and explicitly does not claim ternary zeros satisfy Nvidia 2:4 sparsity, predict host/communication/IO costs, or prove achievable speed.

## The independent next path (`terminus_next/`)

### Why it exists and how it differs

This directory implements the architectural recommendations as a competitor, rather than mutating the supported root model. It has a different import namespace (`ternary_lm`), configuration schema, token format, sampling semantics, checkpoint contract, training schedule units, quantizer choices, and deployment/export system. A root-path checkpoint/manifest/config cannot be consumed by it, and the reverse is also true. This separation is a strength for controlled comparisons, but it makes it unsafe to read results from one stack as a direct continuation of the other.

The main `shared_1p5b` candidate has **1,500,613,020 parameters** according to its own evidence: 1,301,807,104 ternary-projection latent parameters and 196,608,000 embedding parameters, with remaining norms/scales/gates. It uses width 2,048, 30 layers, 16 query heads, lower local KV heads=4, memory KV heads=8, SwiGLU width 5,504, 4,096 context, 1,024 local window, tied 96k embeddings, 128×128 ternary grouping, learned block scales, optional head gates, and factorized gains. The precision policy in its default model config begins with native INT8 activation simulation on the ordinary Torch backend; the H100 preset changes to FP8/DeepGEMM/FA3/streaming CE.

### Shared-memory topology

The key architectural change is a split depth:

```text
tokens → embedding
      → lower 15 blocks of local causal self-attention (4 lower KV heads)
      → one shared-memory projection: lower hidden → 8-head memory K/V
      → upper 15 blocks: queries attend to shared memory; no per-upper self K/V
      → final RMSNorm → tied 96k head
```

For `topology="shared"`, `self_layers=15`: layers 0–14 own ordinary local self-attention K/V banks, then `memory_k`/`memory_v` project the lower output once. Layers 15–29 construct Q and output projection but cross-attend to the one shared memory bank. This cuts repeated global KV projections/cache from upper layers, and it enables a special inference optimization: during prompt prefill the lower stack consumes all prompt tokens and writes memory; only the last prompt state needs the upper stack to obtain the first next-token logits. During decode, every new token passes lower then upper. `--all-logits` deliberately forces upper evaluation for all prompt positions and therefore does more work.

The alternative `conventional` topology uses every block as self-attention, with `global_every` defining the local/global schedule similarly to the maintained path. `deep40_1p5b` is a depth/width challenger (40 × 1,536). `shared4_1p5b` reduces shared memory heads to four as a structural control. Configurable per-layer FFN widths, SwiGLU or ReLU², Q/K RMS, learned per-head temperatures, scalar/channel/absent SubLN, and optional query-dependent head gates make this a wide explicit ablation space rather than a fixed design.

`CacheBank` stores detached K/V/document-ID windows, with an absolute start position and checks for continuity/batch/head/device policy drift. `InferenceCache` creates local banks for lower shared layers only and one global shared-memory bank; conventional mode instead creates a bank per layer with local/global capacity. Deployment mode rounds small weights/gains/KV to FP16 as a simulation of native arithmetic. The model rejects cache use in training/with autograd, mismatched configurations, and capacities beyond context.

### Next quantization and optimizer semantics

This path does not reuse the maintained absmean-only STE. `TernaryLinear` retains FP32 matrices but supports five quantizer policies:

- `absmean`: derived group scale with identity gradient;
- `lsq`: learned log-step with LSQ-like masked derivative;
- `seq`: learned log-step with released-SEQ-like surrogate term;
- `wide`: a wider mask/saturation interpretation through the same policy family;
- `code`: parameters represent code coordinates and get scale-aware derivatives.

Groups may have partial edges. Learned scale gradients can be normalized by group, whole tensor, or disabled. Ternary forward is still code `round/clamp(-1,1)` times a group step, but the scale itself can become a trainable parameter. `factorized_gains` adds centered positive input and output gain vectors around each logical matrix; centering makes their group geometric mean one and avoids a trivial scale/gain degree of freedom. Activations can be none/INT8/INT4/FP8, with an optional learned clipping ratio for INT8/INT4. Embedding/head QAT uses row-group INT4 with FP16 scales.

Logical Q, K, V, gates and FFN matrices remain separate modules even when some computation might be fused, so that Muon sees individual matrices and roles remain auditable. The standalone optimizer labels parameters as ternary, scale, norm, embedding, or gate. AdamW is the control. `HybridMuon` applies its Newton–Schulz polar-like update to ternary/gate logical matrices and auxiliary AdamW to semantic nonmatrix roles; its result is an approximate orthogonal update, not an exact polar factor. Quantization diagnostics sample codes from every layer and record zero/saturation fractions, boundary margins, scale range, and code flips between reports.

The optional `deepgemm` backend is similarly explicit and SM90-only. It recomputes effective operands each call rather than caching them, so learned scale/gain changes cannot create stale FP8 codes. Its modes are BF16, FP8 dgrad, and full FP8 wgrad with FP32 output. It has no silent Torch fallback. Because this package's effective-weight STE is outside the custom GEMM, the GEMM's weight gradient reaches learned scales/gains.

### Next data, training, checkpoint, evaluation

The next data format is intentionally simpler and not multilingual-policy aware. `prepare.py` accepts JSONL rows that are ID lists, `{ids: [...]}`, or `{text: ...}` if supplied an external SentencePiece model. It appends EOS if absent, writes little-endian `uint32` `tokens.bin` and a parallel `documents.bin`, records count/vocabulary/EOS/source/tokenizer/file hashes in a `terminus-tokens-v1` manifest. `TokenStream` memory maps and verifies both files, then randomly samples sequence starts with a seeded Torch generator. Unlike the maintained deterministic finite router, it samples with replacement from the prepared stream. If `document_mask=true`, it masks a target to `-100` whenever the next token crosses a document ID; attention can also enforce same-document dependencies under Flex/reference attention. The package expects language mixing to have happened upstream and evaluates separate manifests per language; it does not enforce the root package's 11-language, licenses, source-mix, dedup, or held-out coverage policy.

The next trainer uses schedules in **optimizer steps**, not data-token horizon. Its default 1.5B presets specify 100,000 updates, microbatch 1, context 4,096, accumulation 32, warmup 1,000 steps, and save/eval every 1,000; that implies 131,072 target positions/update/rank before masking. It can have an initial dense phase and a QAT strength ramp, which are prohibited on the DeepGEMM backend. As in the maintained stack, it computes summed CE, counts unmasked targets across every rank, scales by `world_size/global_token_count`, accumulates gradients, checks/ clips global norm, then steps.

DDP contracts include model, optimizer, train config, data/validation hash, code hash, runtime, world size and optimizer parameter-role schema. The recent acceptance tool deliberately launches ranks with diverging LRs and asserts rejection before allocation, then checks identical-rank weights and exact model/optimizer resume. A checkpoint is a `terminus-next-checkpoint-v1` Torch payload written to a temporary name then atomically renamed. It stores model config/fingerprint, model/optimizer, contract, completed step, trained target count, tokenizer hash, and rank-specific sampler/Torch/CUDA RNG states. It verifies its model-config fingerprint on read; unlike the maintained format, it does not use adjacent SHA-256 receipts/fsync-directory semantics. `--initialize` is intentionally different from `--resume`: it permits selected precision/backend changes with a fresh optimizer/schedule but rejects architectural/parameterization change.

Evaluation provides per-stream perplexity, tokenized multiple-choice (sum log probability by default; optional length-normalization) and exact greedy-generation tasks. It can turn on deployment simulation to compare native-like INT4 tied table / FP16 small values/KV behavior, but this is still PyTorch simulation. It does not include real benchmark datasets or fabricated language scores.

### Packed export and native C++ runtime

`ternary_lm/packing.py` creates a `TLMQ001` container, which is the only deployment format in the checkout. It refuses an existing destination, rejects FP8 activation configurations, and requires all ternary modules at full QAT strength. It serializes:

- each ternary matrix in 128×128 (or configured) blocks, five trits per byte (`-1/0/+1` encoded base 3), with FP32 block scales;
- each optional input/output gain, RMSNorm, gate, or temperature as FP16 (clip ratio FP32);
- the tied embedding/head once as an INT4 table (`-7..7`) with row-group FP16 scales; and
- a JSON header with exact model config/fingerprint, independent tensor descriptions, 64-byte aligned offsets, per-section CRC32, rounding/precision contracts and payload length.

No FP32 model or optimizer state is present in the packed file. `PackedFile` mmap-loads it, validates magic/header bounds/config fingerprint/unique names/section bounds and, by default, CRCs. It can reconstruct dense arrays only as a diagnostic, explicitly not as a phone-serving strategy.

`native/runtime.cpp` is a standalone C++17 executable built by CMake. It mmap-loads the container, decodes one row-sized ternary segment at a time, uses ARM NEON integer dots when compiled on suitable ARM and scalar fallback elsewhere, uses bounded FP16 local/global K/V banks, and has no Python/PyTorch dependency. Optional native SentencePiece support is a separately configured CMake dependency. Standard output is JSON; stderr reports load/prefill/decode timing, OS peak RSS, exact KV allocation and upper evaluation count. It supports shared/conventional topology, variable FFN widths, SwiGLU/ReLU², implemented norm/gate/gain policies, unquantized/INT8/INT4 activations and FP16 KV. It is a reference runtime for bounded memory/correctness, not an Android/iOS app or an established fastest mobile kernel.

For its complete untrained full-size random fixture on the inspected ARM64 Mac, the reported packed file was 369.910 MB; requested 2,048 and 4,096 contexts produced peak RSS 413.942 MB and 422.035 MB with 39.846 MB and 48.234 MB KV allocation. This proves one short complete-graph native exercise with random weights, not language quality, target phone latency/energy, tokenizer overhead, long context latency or production device fitness.

## Cross-stack comparison: do not collapse these distinctions

| Concern | Maintained `terminus/` | Independent `terminus_next/` |
|---|---|---|
| Official status | supported current pretraining implementation | architecture/serving challenger |
| Main topology | 30-layer local/global decoder, GQA 16Q/4KV | 15 local self layers + shared memory + 15 cross-attention upper layers |
| Quantizer | derived absmean ternary target, identity STE | absmean/LSQ/SEQ/wide/code surrogate alternatives, optional learned scales/gains |
| Production data policy | 11-language validation, licenses, NFC, held-out split, source mixtures, document dedup | generic uint32 streams; upstream owns language/data policy |
| Sampling | deterministic finite sequential source routing | seeded random sequence starts with replacement |
| Loss | checkpointed or streaming, optional Triton CE | checkpointed or streaming PyTorch CE |
| Native serving | Python cached inference only | versioned packed ternary/INT4 export + standalone C++ runtime |
| Resume integrity | SHA-256 receipt, fsync/atomic commit, exact root source/runtime contract | atomic `.pt`, config/code/data/role contract and rank RNG state |
| H100 current snapshot | 45.2k tokens/s at b4/a2 synthetic; older 43.9k sustained selected report | 4.36k b1/a8; 9.18k b4/a2 synthetic, BF16-backward/checkpointed policy |
| Claim boundary | final launch not approved | candidate quality and full-speed work remain open |

The different input samplers, configurations, checkpoint semantics, FFN widths, topology, kernel policies, activation precision, retained-activation policy, and benchmark harnesses mean the two H100 numbers are **not a clean architecture A/B test**. The Sept. 20 report explicitly warns of this.

## Tests, operations, and evidence state

### Automated quality gates

Root CI uses Ubuntu CPU Python 3.11, installs CPU Torch and the root package, compiles root code, runs root unit tests, count-only preflight and shell syntax, then builds/tests `terminus_next` native code, standalone verification and a two-process distributed acceptance test. It does not provision CUDA, FA3 or DeepGEMM. GPU tests are intentionally conditional rather than misleading CPU fallbacks.

The Sept. 20 H100 report records: maintained local suite 105 run / 86 pass / 19 CUDA skips; updated preflight targeted CPU checks 2 pass; next local suite 39 run / 33 pass / 6 CUDA skips; next H100 suite 39 pass/no skips; selected maintained H100 suite 19 pass/no skips; and two-rank contract/recovery acceptance. `terminus_next/evidence/scaling_20260920/h100.json` records a Torch 2.8.0+cu126 Azure H100 run with 39 tests, zero failures/errors/skips and source hashes. These are real, bounded numerical/software acceptance receipts, but not equivalently a long real-data training certificate.

Native evidence includes an integration smoke model (42,930 parameters) whose synthetic loss fell 4.5395 → 0.1081, native/PyTorch logit max error was 7.15e-7, and greedy output matched; full ternary byte/FP16/NEON arithmetic checks and UBsan were exercised. ASan did not complete on the Mac because the host runtime hung before application `main`; this is documented as unavailable, not passed.

### Current operational status and open gates

The September 20 bounded full-size synthetic campaign had successfully written an approximately 18.01 GB checkpoint at update 16, resumed, and had reached update 36/256 at last recorded observation; the report expressly makes no completion claim. The campaign is single H100 NVL and synthetic. It is not safe to infer completion, current cloud state, or a certification result from that historical observation without rechecking the live evidence/process.

The remaining gates are substantive:

1. freeze/review the actual 96k tokenizer, corpus licenses/privacy/quality/language labels, mixture and held-out contamination policy;
2. run a real multilingual pilot with held-out per-language loss and real task evaluation, including equal-token and equal-compute architecture controls;
3. evaluate ternary/FP8 changes on downstream code decisions and quality, not only local numerical tolerances;
4. complete/inspect a long enough real-data stability/throughput run including IO, validation, checkpoints and failure recovery;
5. validate multi-GPU/NCCL and multi-node throughput/recovery, because current exact DDP evidence is two-process but production scaling is not certified;
6. match runtime/container/kernel source hashes and rerun preflight/H100 tests after kernel, compiler, PyTorch, FA3 or DeepGEMM changes; and
7. for the next path, establish a fair quality/latency/energy result on a target device before treating its shared-memory or native memory advantages as a product decision.

## Reports and research documents

The repository includes several thoughtful but non-executable decision documents. Their role is important but distinct from source-of-truth code:

- `PRETRAINING_AUDIT.md` records engineering readiness gaps for the maintained stack.
- `H100_OPTIMIZATION.md`, `H100_KERNEL_OPTIMIZATION_20260915.md`, `TERNARY_FP8_PATH.md`, and `H100_SCALE_VALIDATION_20260920.md` document kernel policies, measured conditions, numerical gates and limitations.
- Quality/capacity/quantization reviews in `reports/*20260919.md` motivated `terminus_next`'s shared-memory, learned-scale, gate/gain, and ablation directions.
- `FROM_SCRATCH_DISTILLATION_DECISION_20260918.md`, `SARVAM_DISTILLATION_REVIEW_20260918.md`, `alignment_research_report.md`, and planning files are strategy/compute/teacher discussions. They do not add a teacher loss to either trainer today.
- `H100_RESEARCH_PAPER.md/.pdf` is a reproducible analysis/paper built by `scripts/build_h100_paper.sh`; it is not a production hardware claim.

## Bottom line

The maintained `terminus/` system is a heavily guarded multilingual ternary-pretraining prototype: strong on explicit contracts, data validation, numerical kernel gating, full-vocabulary training, recovery receipts, and careful nonclaims; incomplete on real-data quality, final throughput, and distributed production validation. `terminus_next` is an intentionally separate and more ambitious candidate: it broadens the architecture/quantization search and uniquely carries a credible packed native inference reference, but it currently has different training semantics and significantly weaker measured full-size throughput. The archive is evidence, not a live dependency.

The most important architectural fact is therefore organizational: **the repository has a maintained baseline and a research challenger, not a merged winner.** Any next decision should preserve their separation until a predefined matched multilingual quality/compute/deployment evaluation selects one.
