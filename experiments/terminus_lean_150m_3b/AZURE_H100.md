# Azure H100 runbook (future execution only)

**Do not execute this runbook yet.** It requires the owner's data-ready signal and run authorization. This repository has not started or allocated an Azure VM for this experiment.

Use one Azure H100 NVL (SM90) for the first paired test. The measured historical stack was Python 3.10.12, PyTorch 2.8.0+cu126, Triton 3.4.0, FA3 commit `060c9188beec3a8b62b33a3bfa6d5d2d44975fab` and DeepGEMM commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`. Reproduce or deliberately revise those pins before preflight; record `nvidia-smi`, driver, CUDA compiler, package revisions, git commit and GPU SKU. See `docs/terminus/H100_OPTIMIZATION.md` and `scripts/build_h100_fa3.sh`. Historical ternary benchmarks are not validation of the new lean kernel branch or dense FP8 stage.

The FA3 build script enables BF16 head width 128. DeepGEMM's observed JIT needed CUDA 13.2 even with PyTorch built for CUDA 12.6. Build DeepGEMM from the pinned revision and its submodules using that revision's upstream instructions. Keep its JIT cache on the VM's fast disk. Export `PYTHONPATH` for the isolated FA3 build and `DG_JIT_NVCC_COMPILER` for the CUDA 13.2 compiler. Freeze the working runtime before numerical preflight; runtime and native sources are hashed in receipts.

The sequence **after** data readiness is:

1. On the GPU host, run `plan.py check --microbatch 32`, then native preflight for `control_dense`, `control_qat`, `lean_dense` and `lean_qat`. `terminus.preflight` now exercises dense as dense and the lean packed-QKV no-normalization branch. Run the selected Hopper kernel tests, including `tests/test_hopper_candidates.py`; CPU skips are not a GPU pass.
2. Run the bounded synthetic `scripts/h100_sweep.py` on `configs/lean_sweep.json` with `--batches 8,16,32,64 --warmup 3 --steps 8`. All cases process 262,144 tokens/update. Pick the fastest **fitting** case with memory headroom for real-data loading and checkpointing. If it differs from microbatch 32, regenerate and recheck configs with `plan.py generate --microbatch <winner>`; rerun the four preflights after any source/runtime change. Synthetic throughput is a sizing probe, not production throughput.
3. Use `prepare_views.py` on the data team's complete, checksummed, 4,096-token, frozen-tokenizer manifest. It validates the eight-language-plus-math/code scope, shards, token IDs and exposures, then writes a 1.800B dense view and 3B-total view over the same shards. It never rewrites the data. Review the realized mixture and available held-out/task coverage before launch.
4. Train control and lean dense branches independently with the same seed and token order. Each branch ends at exactly 6,867 updates. Convert each checkpoint with `terminus.transition --data-policy continue --device cuda:0`, carrying optimizer, RNG and cursor. Resume QAT to the full usable token budget and evaluate both final ternary checkpoints. Keep complete run/preflight/evaluation receipts and measured GPU-hours. If either branch is interrupted, resume it from its own checkpoint; do not start a fresh run under the same name.

The following commands are templates for **later** use from the repository root. Set paths after the prepared data and Azure filesystem are known:

```bash
python experiments/terminus_lean_150m_3b/plan.py check --microbatch 32

python -m terminus.preflight --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --output /runs/lean/preflight_dense.json
python -m terminus.preflight --config experiments/terminus_lean_150m_3b/configs/lean_qat.json --output /runs/lean/preflight_qat.json
python -m pytest -q tests/test_hopper_candidates.py tests/test_gpu.py tests/test_dense_fp8.py
python -m scripts.h100_sweep --config experiments/terminus_lean_150m_3b/configs/lean_sweep.json --batches 8,16,32,64 --warmup 3 --steps 8 --output-dir /runs/lean/microbatch_sweep

python experiments/terminus_lean_150m_3b/prepare_views.py --base-manifest /data/prepared/manifest.json --data-root /data/prepared --output-dir /runs/lean/views

python -m terminus.train --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --manifest /runs/lean/views/manifest.dense.json --data-root /data/prepared --output-dir /runs/lean/dense --preflight-report /runs/lean/preflight_dense.json
python -m terminus.transition --parent-run /runs/lean/dense --config experiments/terminus_lean_150m_3b/configs/lean_qat.json --manifest /runs/lean/views/manifest.full.json --data-root /data/prepared --output-dir /runs/lean/qat --world-size 1 --device cuda:0 --data-policy continue
python -m terminus.train --config /runs/lean/qat/qat_config.json --manifest /runs/lean/views/manifest.full.json --data-root /data/prepared --output-dir /runs/lean/qat --resume /runs/lean/qat/checkpoints --preflight-report /runs/lean/preflight_qat.json
```

Repeat the dense/QAT sequence for `control` with distinct output paths and its own two preflight receipts. Use `scripts/evaluate_run.py` on dense checkpoints and both final QAT checkpoints with the manifest's tokenizer copy and versioned development-task manifest. Keep untouched final tasks sealed until the architecture and hyperparameters are chosen. The selected candidate must win or justify a tradeoff on final ternary quality **and** measured GPU-hours; no 150M result alone certifies the 1.5B production design.
