# Terminus Lean 150M dense-FP8, 10B-token run

**Status: completed.** The run trained for 9,999,998,976 tokens and 25,432 updates on one H100, including interruption and exact resume. Final mixture-weighted held-out token loss was 2.4194 and no non-finite incidents were recorded. See [the final analysis](analysis/FINAL_RUN_ANALYSIS.md), [compact metrics](analysis/figures/metrics.json), and [the repository results guide](../../docs/RESULTS.md). Raw checkpoints, token shards, event logs, and cloud verification receipts are external.

The input was a deterministic 10B prefix of a shuffled, checksummed packed corpus covering English, seven Indic languages, math, and code. The frozen tokenizer, 25-layer model, DeepGEMM/FA3/Triton kernels, and Muon optimizer were unchanged from the 2026-10-01 real-token H100 benchmark.

`plan.py generate --microbatch 48` defines a nominal 393,216-token optimizer update: 48 sequences × 4,096 tokens × 2 accumulation steps × one H100. The WSD schedule used 1B warmup, 7B stable, and 2B cosine decay to 10% of the 0.01 peak base learning rates. The final update was partial. Recovery checkpoints were written every 1,000 updates, with explicit midpoint/final checkpoints and audits. `resume_prepare.py` removes non-durable telemetry before replay.

The persistent run directory contained the plan/config, environment and native-preflight receipts, validated manifest view, per-update events, stability probes, incidents, held-out source validation, SHA-256 checkpoint receipts, and audits. `launch.sh` mirrored these artifacts to an authorized S3 location. The finalizer required the completed audit, hashed every regular run file, verified external Blob sizes/Content-MD5 values, and only then requested VM deallocation. Account names, resource IDs, local usernames, and bucket names have been removed from the public templates without changing the recorded run hashes or numerical provenance.

`prepare_views.py` validates every shard checksum, tokenizer identity, mixture, and campaign scope. The sanitized `.service` files are execution templates, not ready-to-run launchers. `launch.sh` requires explicitly configured `DATA_ROOT`, `RUN_ROOT`, `S3_RUN_URI`, `PYTHON`, `PYTHONPATH`, `CUDA_HOME`, and `DG_JIT_CACHE_DIR`; it can train and synchronize remote artifacts and must not be run for documentation verification.

This is one dense trial, not an optimized recipe. The first attention-norm gain and residual-scale behavior warrant controlled follow-up, and held-out token NLL is not downstream task accuracy.
