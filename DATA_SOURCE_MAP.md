# Data and evaluation source map

This file records source scope and access caveats for reproduction. **No training corpus or evaluation dataset is committed.** `docs/s3_inventory/live_inventory_20260929.{json,md}` is a dated inventory of the authorized source snapshot and contains small schema/sample excerpts, not the corpus itself; `training-curriculum.md` and `curriculum-plan.md` are older historical leads, not a selected 1.2B or 15%-Indic recipe. Bucket names below are placeholders for private source/storage locations; substitute only locations you are authorized to access.

## Campaign scope

Target languages: `en`, `hi`, `mr`, `kn`, `te`, `ta`, `gu`, `bn`. Additional domains: math and code. This matches the supplied tokenizer's direct-script design. Campaign manifests enforce this scope through `CAMPAIGN_SCOPE_V1`; the legacy 11-language Terminus manifest path remains separate.

## S3 locations to re-inventory

The prior inventory identifies `s3://your-source-bucket/` as the main raw-text bucket. It includes:

| Role | Recorded prefixes or source families |
| --- | --- |
| English web | `datasets/nemotron-climbmix/karpathy-400b-shuffle/` (ClimbMix), `datasets/nemotron-cc-hq/`; `source-snapshots/HuggingFaceTB/smollm-corpus/` for Cosmopedia. The dated live inventory records all three; revalidate current availability and rights before new runs. |
| Indic native text | `datasets/sangraha/verified/`, `datasets/sangraha/synthetic/`, `datasets/sangraha/unverified/`, `datasets/fineweb2-indic/`. Select only the campaign languages and inspect language/script labels. |
| Translated Indic | Committed `translations/` prefixes for Cosmopedia, Nemotron-CC HQ synthetic, and Fact-Seeking. Prefer the v5 Cosmopedia record when source IDs overlap v4; exclude `translation-staging/` from training. |
| Math | `datasets/nemotron-cc-math-v1/4plus/`, `4plus_MIND/`, optionally score band `3/`; `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/`. |
| Code | `datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/` and `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding/`. The freeze note says CodeAlchemy and a separately named Nemotron Code corpus were not present in the inspected bucket. |

`s3://your-research-bucket/` represents the separate research store that held counted TinyLlama-tokenized packs and research material. Those token IDs and recorded token budgets are **not usable as counts under the selected tokenizer**. Obtain original text and retokenize. Avoid counting duplicate copies or staging runs. Do not place AWS credentials in this folder or in experiment receipts.

Before choosing new mixtures, re-list applicable prefixes, sample and validate Parquet schemas, establish source IDs and rights/provenance, count usable documents and bytes, and measure token inventories with the frozen tokenizer. Use deterministic source-ID splits, cross-source near-deduplication, overlap checks for translations, and benchmark-contamination checks. Record actual per-language and per-domain exposure for every candidate; nominal sampling weights are insufficient.

## Public evaluation sources for reproduction

No evaluation records are bundled. Fetch and version:

- `openbmb/DensingLaw-ScalingBench`, revision `c730220daf6dbd4885ea254bcaca0e1f86384098`: five separate JSONL files under `bbh/`, `human_eval/`, `math/`, `mbpp/`, and `mmlu/`. These are **evaluation** records, not pretraining text. URL: https://huggingface.co/datasets/openbmb/DensingLaw-ScalingBench
- Suitable licensed development tasks for the seven Indic languages, including MILU if accessible: https://huggingface.co/datasets/ai4bharat/MILU
- Untouched final tests and domain holdouts for all eight languages, math, and code. Public tasks repeatedly used for HPO are development data, not final independent evidence.

The exact prompt templates, answer keys, continuation masks, task versions, licenses, and hashes belong in a versioned evaluation manifest. Score conditional reference loss and direct accuracy separately. Do not let generated rationales replace verified answers.
