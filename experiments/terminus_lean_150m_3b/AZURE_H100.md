# Azure H100 runbook for the single dense FP8 trial

**Future execution only.** Wait for the data owner's complete prepared manifest and explicit launch authorization. This folder has not allocated a GPU or started training.

Use one Azure H100 NVL (SM90). Reproduce or deliberately revise the measured stack: Python 3.10.12, PyTorch 2.8.0+cu126, Triton 3.4.0, FA3 `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, DeepGEMM `559d79fb6994a58b8a15b4b93bf13ccc16edf247`. Record driver, CUDA compiler, all package/native revisions, git commit and GPU SKU. See `docs/terminus/H100_OPTIMIZATION.md` and `scripts/build_h100_fa3.sh`. The FA3 build enables BF16 head width 128; the observed DeepGEMM JIT needed CUDA 13.2 even with PyTorch CUDA 12.6. Native source/runtime hashes are captured in the run receipt. Historical ternary benchmarks do not validate this dense lean branch.

After the data-ready and launch signals:

1. Run `plan.py check`, H100 dense numerical preflight, and Hopper GPU tests. CPU skips do not count as a GPU pass. Record a failed numerical preflight and stop if dense FP8 loss/gradients do not agree sufficiently with the reference.
2. Run the bounded synthetic microbatch sweep at fixed 262,144 tokens/update. Choose the fastest fitting microbatch with memory headroom, then regenerate/recheck the plan if it is not 32. Keep the batch token count fixed while comparing microbatches; record the selected configuration before training. A synthetic sweep does not establish production throughput.
3. Only after the data team delivers the full checksummed manifest, run `prepare_views.py` to validate the frozen tokenizer and eight-language-plus-math/code scope. Review its realized mixture and available held-out/task coverage.
4. Start the **one dense run**. Examine the first 100 per-update records and the first sampled layer probes for loss spikes, heavy clipping, FP8 error and update-size drift; interrupt if clearly unstable. This matters because the copied pilot LR is unvalidated with Muon at this width. Stop at update 5,722, verify the immutable checkpoint receipt and midpoint audit, then resume the **same run directory** through update 11,445. The final partial update and its token count are handled by the trainer. Do not change model/data/optimizer configuration across resume.
5. Evaluate both dense checkpoints with the versioned held-out and task sets. Preserve all event, stability, incident, preflight, environment, evaluation and checkpoint receipts. The Indic task suite remains gated until the data/evaluation workstream provides it.

Commands below are templates for later use; replace paths after the prepared data and Azure filesystem are known:

```bash
python experiments/terminus_lean_150m_3b/plan.py check --microbatch 32
python -m terminus.preflight --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --output /runs/lean/preflight_dense.json
python -m pytest -q tests/test_hopper_candidates.py tests/test_gpu.py tests/test_dense_fp8.py
python -m scripts.h100_sweep --config experiments/terminus_lean_150m_3b/configs/lean_sweep.json --batches 8,16,32,64 --warmup 3 --steps 8 --output-dir /runs/lean/microbatch_sweep
python experiments/terminus_lean_150m_3b/prepare_views.py --base-manifest /data/prepared/manifest.json --data-root /data/prepared --output-dir /runs/lean/views
python -m terminus.train --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --manifest /runs/lean/views/manifest.dense.json --data-root /data/prepared --output-dir /runs/lean/dense --preflight-report /runs/lean/preflight_dense.json --stop-after-updates 5722
python experiments/terminus_lean_150m_3b/audit.py --run-dir /runs/lean/dense --phase midpoint --output /runs/lean/dense/audit_midpoint.json
python -m terminus.train --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --manifest /runs/lean/views/manifest.dense.json --data-root /data/prepared --output-dir /runs/lean/dense --resume /runs/lean/dense/checkpoints --preflight-report /runs/lean/preflight_dense.json
python experiments/terminus_lean_150m_3b/audit.py --run-dir /runs/lean/dense --phase final --output /runs/lean/dense/audit_final.json
python scripts/evaluate_run.py --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --checkpoint /runs/lean/dense/checkpoints/step_00005722.pt --tokenizer tokenizer/tokenizer-65k.json --eval-manifest /eval/eval_manifest.json --eval-root /eval --output /runs/lean/dense/eval_midpoint.json
python scripts/evaluate_run.py --config experiments/terminus_lean_150m_3b/configs/lean_dense.json --checkpoint /runs/lean/dense/checkpoints/step_00011445.pt --tokenizer tokenizer/tokenizer-65k.json --eval-manifest /eval/eval_manifest.json --eval-root /eval --output /runs/lean/dense/eval_final.json
```

Expected normal checkpoint receipts are `step_00005722.json` and `step_00011445.json` under `/runs/lean/dense/checkpoints`, beside the matching `.pt` files. `latest.json` points to the newest one; it does not delete the midpoint. The trainer saves an extra recovery checkpoint on SIGINT/SIGTERM. The audit verifies both checkpoint SHA-256 hashes, uninterrupted per-update records, numerical-probe coverage, strong clipping, loss spikes and sampled FP8/attention/update anomalies. Its thresholds are investigation cues, not an automatic quality verdict. Use `scripts/evaluate_run.py` with each exact `.pt` path and the versioned task manifest; keep development and untouched final tasks distinct. The trainer's validation events supply held-out language/domain loss; task evaluation supplies conditional reference loss. This run tests dense quality and numerical stability only. It cannot establish ternary quality or the best 1.5B recipe.
