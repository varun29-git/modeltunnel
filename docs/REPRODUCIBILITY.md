# Reproducibility and artifact boundaries

## What is included

The repository includes source code, CPU/native-GPU tests, experiment configurations, tokenizer assets with their own license terms, compact JSON metrics, generated figures, analysis scripts, and historical documentation. It does not include pretraining corpora, token shards, ScalingBench/MILU records, model checkpoints, full training logs, credentials, or private cloud run mirrors.

Consequently, there are three different reproduction levels:

1. **Source correctness:** install dependencies and run the CPU suite; no external data or GPU is required.
2. **Compact-analysis regeneration:** obtain the exact external checkpoints/logs independently, verify the final checkpoint SHA-256 values below, and rerun the committed analysis programs.
3. **Training reproduction:** independently prepare authorized raw data and evaluation manifests, provision a native stack matching the external environment receipts, and rerun training. Source alone cannot recreate the completed runs, and no claim is made that the original private storage is publicly accessible.

## 1. CPU correctness

Use Python 3.10+ in a fresh virtual environment:

```bash
python -m pip install -e '.[muon]'
python -m compileall -q terminus tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
python -m terminus.preflight --config configs/1.5b_candidate.json --count-only
bash -n scripts/train.sh
```

This matches `.github/workflows/ci.yml`. GPU-specific tests skip when CUDA is unavailable; those skips are not native DeepGEMM/FA3 validation. `tests/test_task_eval.py` discovers and tests both `terminus/task_eval.py` and `terminus/codeexec.py`. `tests/test_corpus.py`, `tests/test_prepare_campaign.py`, and `tests/test_data.py` cover corpus hygiene, manifests, and deterministic routing without contacting cloud storage.

## 2. Regenerate the committed analyses

Install the analysis-only plotting dependency in the same environment:

```bash
python -m pip install matplotlib
```

### Dense run

Place the externally obtained dense run under `/path/to/external/dense-run`, preserving its relative file layout. The analysis consumes exactly `dense/events.jsonl`, `dense/stability.jsonl`, `audit_final.json`, and `views/manifest.dense.json`; it does not load or hash the checkpoint. Separately verify the final checkpoint against the expected value below if using it for evaluation or combined analysis.

```bash
python experiments/terminus_lean_150m_10b/analysis/analyze_final_run.py \
  --run-root /path/to/external/dense-run \
  --output-dir /path/to/output/dense-figures
```

The expected final dense checkpoint SHA-256 is `3d116e88048c93a0403ab2920b1680b066a9f29cbc07fa8ae1c2c67f2144684b`.

### Combined dense/QAT analysis

Supply the two exact final checkpoints, both event and stability logs, and the initial QAT evaluation. The script refuses checkpoint hash mismatches:

```bash
python experiments/terminus_lean_150m_qat_2b/analysis/analyze_combined_runs.py \
  --dense-checkpoint /path/to/external/dense-final.pt \
  --qat-checkpoint /path/to/external/qat-final.pt \
  --dense-events /path/to/external/dense-events.jsonl \
  --qat-events /path/to/external/qat-events.jsonl \
  --dense-stability /path/to/external/dense-stability.jsonl \
  --qat-stability /path/to/external/qat-stability.jsonl \
  --qat-initial-eval /path/to/external/qat_initial_eval.json \
  --output-dir /path/to/output/combined-analysis
```

The expected QAT checkpoint SHA-256 is `ceee823262d11e914eb63d6a396c3df90fb47e5dd9301f7db0f8add0a468c29f`. Regeneration writes `combined_metrics.json` and four figures. It does not rerun training or native inference.

## 3. Data and bounded pipeline smoke reproduction

The raw-source locations in historical inventories are access-controlled inputs, not public dataset downloads. Review source rights before using ingestion scripts. Replace the `your-source-bucket` placeholder in the relevant script with an authorized source URI and configure credentials outside the repository; the placeholder is intentional and is not a runnable default. Do not treat dataset-license labels as a completed legal review. The bundled derivative Sarvam tokenizer has separate [non-commercial research terms](../tokenizer/SARVAM_AI_RESEARCH_LICENSE.md).

The bounded route is:

```text
authorized raw sample -> corpus hygiene -> frozen tokenizer -> versioned manifest
-> tiny dense-FP8 CPU training -> checkpoint -> ternary QAT -> exact resume
-> held-out/task evaluation -> result ledger
```

After acquiring the public evaluation records at the pinned revision and preparing the authorized sample:

```bash
python scripts/fetch_eval_tasks.py --output-dir eval
python scripts/prepare_campaign_sample.py \
  --output-dir sample/corpus --cache-dir sample/cache --max-rows 1200
python scripts/roundtrip_scan.py --corpus-glob 'sample/corpus/records.*.jsonl'
python scripts/run_smoke.py --corpus sample/corpus --work runs/smoke --eval-root eval
```

`eval/`, `sample/`, and `runs/` are ignored local outputs. The smoke model is a correctness fixture, not quality evidence. `terminus/corpus.py` applies NFC normalization, stable-ID/exact/MinHash deduplication, deterministic splitting before tokenization, and shingle-based task contamination filtering; these checks do not solve privacy, source authenticity, or all semantic leakage.

## 4. Evaluate an external checkpoint

`scripts/evaluate_run.py` verifies a checkpoint against its run contract and scores versioned ScalingBench development records:

```bash
python scripts/evaluate_run.py \
  --config /path/to/external/run/config.json \
  --checkpoint /path/to/external/run/dense/checkpoints/step_00025432.pt \
  --tokenizer tokenizer/tokenizer-65k.json \
  --eval-manifest eval/eval_manifest.json --eval-root eval \
  --tasks bbh,human_eval,math,mbpp,mmlu \
  --output /path/to/output/eval_receipt.json
```

For QAT, use `/path/to/external/qat-run/qat/qat_config.json` and `/path/to/external/qat-run/qat/checkpoints/step_00005087.pt` (with that phase's `qat/run.json` contract beside `qat/checkpoints/`). Conditional reference NLL, answer-only loss, choice margins, and direct generation accuracy are distinct measurements. `terminus/codeexec.py` must only be enabled for trusted evaluation code in an appropriate outer sandbox; its subprocess rlimits are not a security boundary, and macOS does not enforce its address-space limit. The committed long-run evidence contains no completed downstream QAT suite.

## Native execution

Native FP8/FA3 reproduction requires an SM90 host and deliberately installed/pinned dependencies. `terminus.preflight` binds model configuration, source hashes, runtime, and native-source identity. Historical launch/finalizer files are sanitized execution records and templates, not one-click public launchers; they can synchronize artifacts or deallocate infrastructure and must not be run as part of documentation verification.
