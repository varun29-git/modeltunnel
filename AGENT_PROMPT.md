# Terminus maintainer context

This repository is a public research release of an implemented ModelTunnelv2-style pipeline. It is no longer a build brief. Start with `README.md`, `docs/RESULTS.md`, and `docs/REPRODUCIBILITY.md`; use `AGENTS.md` for working rules.

## Current implementation

- `terminus/model.py`, `quantization.py`, and the FP8/native-kernel modules implement separate dense-weight FP8 and ternary-QAT modes with FP32 latent weights and Torch reference paths.
- `terminus/train.py`, `optim.py`, and `logical_muon.py` implement packed-stream training, cosine/WSD schedules, distributed reductions, validation, and Muon/auxiliary AdamW optimization.
- `terminus/checkpoint.py` and `transition.py` implement immutable checksummed checkpoints, strict resume contracts, model/optimizer/data/RNG restoration, and deliberate dense-to-QAT transitions with receipts.
- `terminus/corpus.py`, `prepare_campaign_data.py`, and `data.py` implement restartable ingestion, NFC normalization, stable-ID/exact/near deduplication, deterministic splits before tokenization, contamination filtering, source manifests, and finite deterministic routing.
- `terminus/task_eval.py` implements conditional reference NLL, answer-only loss, choice margins, bits/byte, deterministic sharding, and bootstrap intervals. `terminus/codeexec.py` provides opt-in resource-limited code execution, not a security boundary. Their tests are in `tests/test_task_eval.py`; corpus tests are in `tests/test_corpus.py` and `tests/test_prepare_campaign.py`.
- The frozen tokenizer is `tokenizer/tokenizer-65k.json`, SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. Preserve its IDs and third-party license. Packed documents use `<eod>` (ID 4). Legacy RWKV support is separate and incompatible with these token IDs.

## Evidence and open questions

The completed 146.4M-parameter experiments are a 9,999,998,976-token dense-FP8 run and a 1,999,998,976-token ternary QAT continuation. Compact metrics, figures, and analysis code are committed; raw corpora, evaluation records, checkpoints, and full logs are external.

QAT ended with worse held-out token loss than the dense parent on every tracked source. There is no matched same-data/same-schedule dense continuation, so this is not causal evidence of a ternary quality tax. No downstream QAT suite, broad Indic task suite, multi-seed study, or validated width transfer has been completed. One attention-norm/residual-scale regime merits controlled follow-up. Do not convert these observations into an optimum or downstream-quality claim.

## Change guidelines

Preserve exact-resume checks and tokenizer/data contract validation. Keep dense and ternary behavior independently testable. Treat `docs/terminus/` protocols, old inventories, and launch receipts as dated historical snapshots unless the current README explicitly identifies them as living documentation. Replace broken local references with canonical public links and do not restore removed third-party PDFs.

Run relevant tests plus the CI-equivalent CPU suite for changes. Native SM90 results require actual native execution; CPU emulation and CUDA skips are not substitutes. Cloud or GPU campaigns, data uploads, deallocation scripts, and remote state changes require separate explicit authorization and must not be triggered by documentation or analysis work.
