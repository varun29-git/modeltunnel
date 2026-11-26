# ModelTunnel

ModelTunnel is a general research framework for multilingual decoder pretraining with dense-weight FP8 arithmetic, conversion to ternary quantization-aware training (QAT), deterministic packed data, strict checkpoint resume, and task-aware evaluation. Its reference experiments use a 146.4M-parameter model over English, seven Indic languages, math, and code.

This repository contains the implementation, tests, configurations, analysis code, and compact result artifacts. **Raw corpora, evaluation records, training checkpoints, and full run logs are external and are not included.** The results are evidence from one model/seed/recipe, not an optimized training recipe or a claim of downstream competence.

## Development timeline

| Date | Milestone |
| --- | --- |
| 29 September 2026 | Standalone training implementation and frozen research substrate established. |
| 1 October 2026 | Ten-billion-token dense-FP8 reference run completed on one H100 with interruption and exact resume. |
| October 2026 | Two-billion-token ternary-QAT continuation and combined scaling analysis completed. |
| **27 November 2026** | **Implementation, evaluations, run analyses, and reproducibility notes published as the first open-source release under Apache-2.0.** |

## Results at a glance

- A dense-FP8 run completed **9,999,998,976 training tokens** and 25,432 updates on one H100, including interruption and exact resume. Final mixture-weighted held-out token loss was **2.4194**, with no recorded non-finite incidents.
- A ternary QAT continuation completed **1,999,998,976 additional tokens** and 5,087 updates. Under the original dense validation weighting, loss was **2.6042**, versus **2.4194** for the dense parent; every tracked source loss was worse after QAT.
- These runs do **not** isolate a ternary quality tax: there is no matched same-data/same-schedule dense continuation. The QAT mixture, learning-rate schedule, and source position also changed.
- No downstream QAT evaluation suite has been completed. Held-out next-token loss is not task accuracy, code-execution ability, or evidence of scale transfer.

See [the results guide](docs/RESULTS.md), the [combined dense/QAT analysis](experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md), and the [dense final-run analysis](experiments/terminus_lean_150m_10b/analysis/FINAL_RUN_ANALYSIS.md) for methods, metrics, and limitations.

## Implemented system

| Area | Implementation | Tests and entry points |
| --- | --- | --- |
| Model and dense/ternary arithmetic | `terminus/model.py`, `terminus/quantization.py`, `terminus/fp8_kernels.py`; FP32 latent weights, separate `weight_mode="dense"` and `"ternary"`, Torch reference and native DeepGEMM paths | `tests/test_model.py`, `tests/test_dense_fp8.py`, `tests/test_fp8.py` |
| Trainer and schedules | `terminus/train.py`, `terminus/optim.py`, `terminus/logical_muon.py`; distributed-aware loading, cosine/WSD schedules, Muon and auxiliary AdamW, validation and stability events | `tests/test_training.py`, `tests/test_logical_muon.py` |
| Checkpoints and phase conversion | `terminus/checkpoint.py`, `terminus/transition.py`; immutable SHA-256-receipted checkpoints, model/optimizer/data/RNG state, exact-resume contracts, explicit dense-to-QAT receipts | `tests/test_training.py`, `tests/test_transition.py`, `tests/test_lean_10b_resume.py`, `tests/test_lean_qat_2b.py` |
| Corpus hygiene and packing | `terminus/corpus.py`, `terminus/prepare_campaign_data.py`, `terminus/data.py`; NFC normalization, stable-ID/exact/near deduplication, deterministic pre-tokenization splits, task-shingle contamination filtering, restartable cursors, finite source routing | `tests/test_corpus.py`, `tests/test_prepare_campaign.py`, `tests/test_data.py`, `tests/test_tokenset_shuffle.py` |
| Tokenization | `terminus/tokenizer65k.py` and the frozen `tokenizer/tokenizer-65k.json`; legacy RWKV/SentencePiece support remains separate | `tests/test_tokenizer65k.py`, `tests/test_rwkv_tokenizer.py`, `tests/test_tokenizer.py` |
| Task and code evaluation | `terminus/task_eval.py`: conditional reference NLL with correct first-answer-token masking, answer-only loss, choice margins, bits/byte, bootstrap intervals, deterministic sharding. `terminus/codeexec.py`: opt-in subprocess execution with time/resource limits; it is not a security boundary | `tests/test_task_eval.py` covers both modules; `scripts/evaluate_run.py` scores verified checkpoints |
| Experiment tracking | `terminus/registry.py`, `terminus/search_plan.py`, `terminus/campaign.py` | `tests/test_registry.py`, `tests/test_campaign.py`; `scripts/campaign_dry_run.py` only renders a plan |

The frozen vocabulary has 65,536 IDs and SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. Packed documents use `<eod>` (ID 4), not the legacy RWKV EOS convention. Its third-party license and provenance are described in [the freeze note](tokenizer/tokenizer-65k-freeze.md) and [NOTICE](NOTICE).

## Reproduce and verify

For CPU correctness checks, use Python 3.10+ and install the project in a virtual environment:

```bash
python -m pip install -e '.[muon]'
python -m compileall -q terminus tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
python -m terminus.preflight --config configs/1.5b_candidate.json --count-only
bash -n scripts/train.sh
```

These are the checks run by [CPU CI](.github/workflows/ci.yml); CUDA-only cases skip on CPU. Native FP8/FA3 validation requires an explicitly provisioned, pinned SM90 environment and must not be inferred from CPU skips.

[Reproducibility and artifact boundaries](docs/REPRODUCIBILITY.md) explain how to rerun compact analyses with externally obtained checkpoints/logs, prepare a bounded corpus smoke sample, and evaluate a checkpoint. [DATA_SOURCE_MAP.md](DATA_SOURCE_MAP.md) records the source-scope caveats; [RESEARCH_LIBRARY.md](RESEARCH_LIBRARY.md) links to canonical research sources rather than bundling paper PDFs.

## License and provenance

Original Terminus source code, documentation, research reports, and modifications in this repository are available under [Apache-2.0](LICENSE). Third-party tokenizer assets, datasets, evaluation records, papers, checkpoints, and other external assets are **not relicensed** by this repository; see [NOTICE](NOTICE) and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md). In particular, the derived Sarvam tokenizer retains its [non-commercial research terms](tokenizer/SARVAM_AI_RESEARCH_LICENSE.md), including the requirement for explicit written permission for commercial use or commercial-entity distribution. Historical documents under `docs/` are labeled snapshots and may describe earlier configurations or plans rather than current behavior.
