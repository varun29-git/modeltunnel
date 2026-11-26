# ModelTunnelv2 pipeline — implementation handoff

**Historical implementation handoff dated 2026-09-29.** Its CPU-only status and readiness gate describe that date, not the final release. Native H100 preflights and the completed 10B-token dense/2B-token QAT runs happened later; use `README.md`, `docs/RESULTS.md`, and `docs/REPRODUCIBILITY.md` for current evidence and reproduction boundaries. Private execution details are sanitized below.

## 1. What exists now (implementation map)

| Component | Location | Status |
| --- | --- | --- |
| Frozen 65k tokenizer (JSON encoder/decoder) | `terminus/tokenizer65k.py` | Implemented, gate-tested, real-data round-trip verified (0/13,717 failures) |
| Legacy RWKV / sentencepiece support | `terminus/tokenizer.py` | Preserved; archived tests pass unchanged |
| Versioned campaign scope (8 languages + math/code domains) | `terminus/integrity.py` (`CAMPAIGN_SCOPE_V1`) | Enforced by `validate_manifest` for scoped manifests; legacy manifests keep the 11-language rule |
| S3 read-only I/O (listing, ranged Parquet reads, bounded cache) | `terminus/s3io.py` | Used by the live inventory and corpus builder |
| Corpus builder (dedup, splits, contamination, restart) | `terminus/corpus.py` | Ran on the bounded sample from real S3 text |
| Campaign token preparation (streams, scope, inventories) | `terminus/prepare_campaign_data.py` | Validated in the smoke run (production gate) |
| Mixture overlays for search trials | `terminus/remix.py` | Unit-tested |
| Dense-weight FP8 stage (`weight_mode="dense"`) | `terminus/model.py`, `terminus/quantization.py`, `terminus/fp8_kernels.py` | CPU/emulated tests pass; **native SM90 numerics not run** |
| WSD schedule + phase-relative offsets | `terminus/train.py` | Unit-tested incl. endpoint and re-warmup checks |
| Dense→QAT transition with receipt | `terminus/transition.py` | CPU-tested: state carry, resume equivalence, data-cursor continuation, failure paths |
| Task evaluation (conditional reference loss, margins, CIs) | `terminus/task_eval.py` | Unit-tested; ran in the smoke route on 40 bounded ScalingBench items |
| Code-execution sandbox | `terminus/codeexec.py` | Unit-tested (limits verified; macOS cannot enforce RLIMIT_AS) |
| ScalingBench fetch/versioning | `scripts/fetch_eval_tasks.py`, `eval/eval_manifest.json` | Fetched at pinned revision `c730220d`; MILU gated (recorded, not substituted) |
| Experiment registry + ledger + frontier/transfer analysis | `terminus/registry.py` | Unit-tested |
| Staged search plans | `terminus/search_plan.py` | Generated trial lists; nothing launched |
| Live S3 inventory | `docs/s3_inventory/live_inventory_20260929.{json,md}` | Produced by `scripts/inventory_s3.py` (read-only) |
| Bounded end-to-end smoke | `scripts/run_smoke.py` | Completed on CPU (see §4) |

## 2. Reproduce the smoke test

```bash
# 0. environment (Python 3.13, macOS/Linux; CPU only)
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python \
  "torch>=2.8,<3" "numpy>=1.26,<3" "sentencepiece==0.2.1" "protobuf>=4.25,<7" \
  "regex>=2024.0" pyarrow boto3 pytest \
  "muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@f98f1cacc0263b04290753e32be8d498c1efc806"

# 1. evaluation tasks (pinned revision; writes eval/)
.venv/bin/python scripts/fetch_eval_tasks.py --output-dir eval

# 2. bounded corpus sample from an authorized S3 source (read-only).
#    The historical run used an in-region CPU worker with scripts/vm_bootstrap.sh.
#    Access keys, network rules, and transfer details are intentionally omitted.
.venv/bin/python scripts/prepare_campaign_sample.py \
  --output-dir sample/corpus --cache-dir sample/cache --max-rows 1200

# 3. tokenizer real-data gate (0 failures expected)
.venv/bin/python scripts/roundtrip_scan.py --corpus-glob 'sample/corpus/records.*.jsonl'

# 4. end-to-end smoke (tokenize -> dense FP8 -> QAT -> eval -> ledger)
.venv/bin/python scripts/run_smoke.py --corpus sample/corpus --work runs/smoke
cat runs/smoke/smoke_receipt.json
```

## 3. Launch later stages (owner decision)

- **Full corpus prep (in-region VM, owner-launched):**
  `python scripts/prepare_campaign_full.py --output-dir /data/corpus_full --cache-dir /data/cache --cache-gb 200`
  then tokenize with `terminus/prepare_campaign_data.py` (see `scripts/run_smoke.py`
  for the call shape) using the production scope and the chosen mixture. Sync
  outputs to S3 from the VM (write access verified for both buckets; the source
  bucket must be treated as read-only by convention).
- **Stage 1 width transfer / stage 3 dense screen / stage 4 QAT branches:**
  generate trials with `terminus.search_plan` (`stage1_width_transfer()`,
  `stage3_dense_screen(...)`, `stage4_qat_branches(...)`), render configs with
  `terminus.registry.make_run_config`, and record every attempt in a
  `terminus.registry.Ledger`. Templates: `configs/campaign_200m_dense_fp8_wsd.json`,
  `configs/campaign_200m_qat_ternary_wsd.json`. Transitions:
  `python -m terminus.transition --parent-run <run> --config <qat.json> ...` —
  run it on the training host with the final world size (contracts bind runtime).
- **Native-GPU gates before any production FP8 run:** `python -m terminus.preflight
  --config <cfg> --output <report>` on the H100 host for both dense and QAT
  configs. The dense-FP8 cases in `terminus/fp8_validation.py` now use dense
  weights and references; they still need an actual SM90 execution receipt.

## 4. What was actually executed

- Full CPU test suite: **158 passed, 15 skipped** (GPU/Hopper classes skip
  cleanly without CUDA). Includes tokenizer gates, corpus/dedup/split/
  contamination, dense-FP8 numerics vs emulated operands and the exact torch
  reference, transition/resume, task-eval masking/margins/CIs, ledger semantics.
- Live S3 inventory (read-only): `docs/s3_inventory/live_inventory_20260929.json`.
- ScalingBench fetched and hashed at the pinned revision; MILU probe: **gated**.
- Bounded sample: 13,717 cleaned documents from 12 real sources built on the
  in-region VM; dedup and contamination counts in `sample/corpus/corpus_report.json`
  (37 ScalingBench-overlapping rows dropped; 20 exact + 3 near dups in
  sangraha-hin; sample data not committed).
- Tokenizer round-trip scan on all 13,717 documents: **0 failures**.
- Smoke train: dense FP8 (emulated on CPU) 12,288 tokens / 6 updates →
  transition (`continue` policy) → QAT to 32,768 tokens with phase-relative
  re-warmup and an in-QAT interrupt/resume; held-out validation per coverage
  key (math/code separated); task eval macro loss 8.61 (random-scale tiny
  model — plumbing proof only, **not a quality signal**).

## 5. Live inventory highlights and missing sources

See the JSON/MD for the full tables. Corrections to the 27 Sep dated inventory:

- `datasets/nemotron-climbmix/karpathy-400b-shuffle/` **exists**: 13,089 objects,
  558.6 GiB (the freeze note's English-share source).
- Cosmopedia translations: v4 covers ben/guj/hin/pan/tam/tel (132.1 GiB), v5
  covers kan/mal/mar/ory (51.4 GiB) — **complementary, not just versions**;
  together they cover all seven campaign Indic languages. Version-preference
  dedup still applies where ids overlap.
- Fact-Seeking r16 lives at
  `translations/Nemotron-Pretraining-Specialized-v1.2/Fact-Seeking/indictrans2-en-indic-dist-200M-beam1-20260902/fact-seeking-70shard-10lang-r16-20260903/`
  (170.6 GiB translated) — the old doc's path was wrong.
- **Missing**: the recorded `krutrim-nemotron-cc-v21-hq-synth-v1` committed run
  has no objects at its recorded location (only review chunks). CodeAlchemy,
  `datasets/nemotron-code`, finemath, SwallowMath confirmed absent.
- Sangraha `synthetic/` uses `lang_Script` partitions with `*_Latn` twins
  (excluded); `verified/` and `unverified/` use ISO-3 partitions.
- Sangraha has no `eng` tier (verified eng exists; synthetic eng_Latn excluded).

Measured fertility on the sample (frozen tokenizer, tokens/byte): en 0.229,
math 0.347, code 0.322, Indic mean ≈ 0.114 (hi 0.108, mr 0.101, kn 0.105,
te 0.124, ta 0.106, gu 0.136, bn 0.124). Indic UTF-8 is ~3 bytes/codepoint, so
per-character fertility is much better than the per-byte number suggests.

Volume estimates (measured fertility × ListObjects bytes): en web ≈ 137B
(ClimbMix) + 254B (CC-HQ) + 28B (SmolLM snapshot); math ≈ 53–90B depending on
bands; code ≈ 10.5B; Indic native+translated ≈ 45–60B. These support the
planned proxy budgets; the 800B-scale target needs the full English pools.

GPU-hour anchors (synthetic, one H100 NVL, existing receipt): 1.5B model at
46,395 tokens/s → ~6.0 H100-hours per 1B tokens at 1.5B scale. 200M proxies are
faster but unmeasured — measure, don't extrapolate, before costing the campaign.

## 6. Risks and open items

1. **Native FP8/FA3 numerics not run here** (no SM90 on this host). The dense
   DeepGEMM path and the new Triton dense quantizer are CPU-emulated only;
   run `preflight` including the fp8_validation dense cases on the H100 before any
   production dense-FP8 run. The dense stage's *quality* behavior vs BF16 is
   unmeasured — that is a stage-3 search question, not a settled fact.
2. **Tokenizer Gate 1** (piece parity with the original Sarvam model) is not
   reproducible here: `tokenizer.model` is not in the repo. Encoder choices
   E1–E4 (documented in `terminus/tokenizer65k.py`) are self-consistent and
   real-data round-trip exact, but differ from a hypothetical reference for
   digit/punct-initial spans and spans after English spaces.
3. **MILU is gated** — Indic task evaluation currently rests on held-out NLL
   only. Acquire MILU (or an equivalent licensed Indic task set) before the
   campaign relies on Indic task-aware signals; the untouched final test set
   is not yet provisioned at all.
4. Contamination guard covers ScalingBench; extend it to any future Indic/final
   tasks when they land.
5. Full-corpus prep throughput: the sample ran at ~15 docs/s single-process on
   c6i.2xlarge; the full run needs the VM path with parallelism (shard the
   `--sources` list across instances) — costed as a VM-days item, not done here.
6. The Muon rule-B transfer and conditional-loss→downstream calibration are
   **unvalidated hypotheses** by design; the stage-1/2 gates exist to measure
   them. Do not promote recipes that fail `registry.ranking_transfer`.
7. `test16/` and the TinyLlama packs remain excluded (incompatible token IDs).
8. Smoke used `development_mode` CPU training; production contract/preflight
   binding is exercised only by unit tests.

## 7. Cost-capped first campaign (dry run only — nothing launched)

**Hard cap: 60 total H100 GPU-hours**, including training, preflight,
evaluation, and checkpoint overhead. Implemented in `terminus/campaign.py`
(budget ledger, launch refusal, repricing) + `terminus/search_plan.py`
(staged plan) + `scripts/campaign_dry_run.py` (reviewable plan renderer).
The dry-run artifact is `docs/campaign_dry_run.md`.

Design rules enforced by code:

- Jobs carry conservative estimates: tokens / (assumed rate x 0.6) + eval +
  checkpoint overhead. Assumed rates are labeled: only the 1.5B synthetic
  46,395 tok/s is measured; 200M/700M rates are ASSUMED placeholders to be
  replaced by stage-0 measurements.
- `Campaign.check_launch` refuses any job whose conservative estimate exceeds
  the remaining budget. Failed, interrupted, and evaluation time count toward
  the cap (ledger `cost.gpu_hours` on every finish record).
- After stage 0, `reprice` re-estimates pending jobs with measured rates and
  drops lowest-priority jobs if the projection exceeds the cap.
- One training run supplies both losses: held-out LM loss from the run's own
  validation events, task loss from scoring the same checkpoints. There are no
  duplicate training campaigns.
- Microbatch is manifest-controlled (`remix_manifest(..., batch_size=...)`);
  `validate_job_manifest` enforces global batch = microbatch x DP world x
  accumulation x sequence length.
- QAT branches share one dense parent (costed once), decay fully within their
  phase budgets, and are compared against a dense continuation and a
  ternary-from-start run at matched total token exposure (asserted by
  `matched_exposure_check`).

Staged plan (dry-run totals: **51.5 conservative / 32.4 expected H100-hours**):

| Stage | Content | Jobs | Conservative hours |
| --- | --- | ---: | ---: |
| 0 | H100 numerical preflight + sustained throughput probes (dense FP8, ternary QAT) | 2 | 1.5 |
| 1 | 200M: 2 Muon LR rules x {0.001, 0.003, 0.009}, 600M tokens each | 6 | 11.5 |
| 2 | top-2 promoted to 700M (1B tokens each) + one short 1.5B check (500M) | 3 | 18.8 |
| 3 | 4 mixture overlays (baseline/indic_heavy/stem_heavy/en_heavy), 600M each | 4 | 7.7 |
| 4 | shared 600M dense parent; 4 QAT branches (switch 0.5/0.8 x 120M/240M); dense-continuation and ternary-from-start controls at matched exposure | 7 | 8.1 |
| 5 | second seed for up to two finalists, only if budget remains | 2 | 3.8 |

Width-transfer evaluation is diagnostic, not declarative: `transfer_diag.py`
measures per-role parameter/gradient/update-to-weight ratios and
activation/logit scales across widths (logical Q/K/V and gate/up slices, tied
embedding/readout, norms). Until a coherent Muon-compatible μP
parameterization is established, results are labeled **"empirical Muon width
transfer"** with stated limits. The frozen 65K vocabulary makes the tied
embedding 25.1% of the 200M proxy (33% at a hypothetical 100M, 40% at 50M), so
no sub-200M proxy is in the family — it would measure embedding scaling, not
trunk scaling.

Exact commands for the H100 host (after owner approval):

```bash
# stage 0 gates (dense + ternary configs)
python -m terminus.preflight --config configs/campaign_200m_dense_fp8_wsd.json --output preflight_dense.json
python -m terminus.preflight --config configs/campaign_200m_qat_ternary_wsd.json --output preflight_ternary.json
# reviewable plan (never launches)
python scripts/campaign_dry_run.py --cap-hours 60
# campaign execution is owner-launched; the runner refuses over-budget jobs
```

## 8. Ready/not-ready gate

**Ready for owner review and stage-0/1 launch on GPU hardware:** tokenizer,
data path, dense-FP8 implementation, phase transition, evaluation, ledger,
search plans — all present, tested, and smoke-verified end-to-end.

**Not ready (explicit):**
- Native H100 preflight for the dense-FP8 path has not run (no GPU here).
- Full corpus preparation has not run (bounded sample only).
- Indic task suite and untouched final tests are not provisioned (MILU gated).
- No hyperparameter/mixture optimum is claimed; no campaign was launched.

## Milestone commits (oldest first)

- `e921233` frozen 65k tokenizer + campaign scope + gate tests
- `e9ff2e9` dense-FP8 stage + WSD + transition
- `657c16e` task eval + sandbox + registry + search plans (+ ScalingBench fetch)
- `b2f2149` S3 corpus pipeline + campaign prep + sample driver
- `9a92afa` phase-relative schedules + effective QAT configs
- `86a5b6e` ignore sample outputs
- `105fab9` sample driver/cache-policy fixes + remix overlays
- `925c8ea` untrack accidentally committed sample files
- `c1f01fc` tokenizer real-data edge fixes (heap BPE, E2–E4, decode classes)
- `05b43c2` smoke orchestrator + full-prep VM driver + inventory summarizer
- `0b8eec0` CI dependency fix (pyarrow/boto3); CI-equivalent suite green
- `d6bfc26` cost-capped campaign runner + staged plan + transfer diagnostics
