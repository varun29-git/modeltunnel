# 1.2B dense decoder curriculum

Inventory date: 27 September 2026. Account `023894314260`, caller `arn:aws:iam::023894314260:user/BedrockAPIKey-7qur`, CLI `/opt/homebrew/bin/aws`. Sizes are `ListObjects` sums. No object was downloaded in full, uploaded, deleted, or overwritten. Parquet token totals were not counted. Where a footer or a manifest gives a number, that number is labeled as measured. Where a figure is a planning choice, it is labeled as an assumption.

## Assumption

Train a **~1.2B-parameter dense decoder-only Transformer from scratch**, then a short supervised tail. Target domain: an **English-primary generalist with an Indic track**.

That stage is inferred from what is stored. The buckets hold pretraining text (web, math, synthetic textbooks, code-like specialized sets, Indic web and Sangraha) plus one assistant-only SFT pack. The listed prefixes contain tokenized corpora and pipeline artifacts. They do not contain a model checkpoint, so continued pretraining is not the default. If a 1.2B checkpoint already exists elsewhere, discard the from-scratch token budget and write a continued-pretraining plan against that checkpoint.

The Indic track is inferred from FineWeb2-Indic, Sangraha, and the committed Krutrim / IndicTrans2 translations sitting next to a much larger English crawl (`datasets/nemotron-cc-hq/`).

Tokenizer is unset. Packs that already exist use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6` on the OpenHermes manifest). Mixture weights below are sampling weights on **training tokens after the chosen tokenizer**. Until that tokenizer exists, epoch caps are enforced on source bytes and document ids, not on a guessed token count.

## What was listed

`aws sts get-caller-identity` succeeded. `aws s3 ls` returned five buckets:

| Bucket | Role in this plan |
| --- | --- |
| `nemotron-training-data-us-east-1` | Pretraining parquet, Cosmopedia snapshot, translation outputs |
| `quark-research-repository-storage` | Counted token packs, OpenHermes SFT, papers, a small domain-classifier prefix |
| `quark-research-repository-storage-eu` | Copies of the same research PDFs as the US bucket `data/` papers. Excluded. |
| `bedrock-invocation-logs-023894314260-us-east-1` | Invocation logs. Excluded. |
| `cdk-hnb659fds-assets-023894314260-us-east-1` | CDK zip/json assets. Excluded. |

Local scripts under the worker directory point at `s3://nemotron-training-data-us-east-1` for Cosmopedia, Nemotron-CC, and Fact-Seeking translation jobs. Those URIs match the prefixes below. Bucket names were taken from `aws s3 ls` and those scripts, not guessed.

Top-level keys under `nemotron-training-data-us-east-1`: `artifacts/`, `bootstrap/`, `datasets/`, `source-snapshots/`, `translation-benchmark/`, `translation-staging/`, `translations/`. `bootstrap/` is worker code. `artifacts/` is one 1,574-byte sample parquet. `translation-benchmark/` is 30,258,432 bytes (3 objects). None of those three is a training mixture component.

## Inventory

Format is parquet unless noted. “Objects” counts every key under the prefix. GiB is bytes / 1024³.

### English and specialized pretraining (`datasets/` and `source-snapshots/`)

| Prefix | Bytes | GiB | Objects | Inferred type |
| --- | ---: | ---: | ---: | --- |
| `datasets/nemotron-cc-hq/` | 1,110,302,262,436 | 1,034.05 | 843 | English high-quality crawl text. Schema `id`, `text`, `metadata.category`, `metadata.models_used`. On `part_000000` and `part_000400`, 30 sampled rows were category `crawl-hq` with empty `models_used`. |
| `datasets/nemotron-cc-math-v1/3/` | 107,417,646,757 | 100.04 | 57 | Web math, score band 3. Schema includes `text` plus FineMath and NeMo Curator scores and WARC ids. |
| `datasets/nemotron-cc-math-v1/4plus/` | 62,189,080,483 | 57.92 | 46 | Web math, score band 4+. Thirty sampled rows on `part_000000` had category `math` and `models_used` `Phi-4`. |
| `datasets/nemotron-cc-math-v1/4plus_MIND/` | 90,331,100,051 | 84.13 | 90 | Web math, score band 4+, MIND slice. Same score/WARC schema. |
| `datasets/nemotron-cc-v2/Diverse-QA/` | 27,897,238,747 | 25.98 | 32 | Synthetic QA. Thirty sampled rows were category `diverse-qa`, model `Mistral-Nemo-12B-Instruct`. This is the only config stored under `datasets/nemotron-cc-v2/`. |
| `datasets/Nemotron-Pretraining-Specialized-v1/` | 350,922,825,780 | 326.82 | 330 | Synthetic specialized text. Subsets below. |
| `datasets/Nemotron-Pretraining-Specialized-v1.1/` | 34,883,344,936 | 32.49 | 68 | Synthetic specialized text. Subsets below. |
| `datasets/Nemotron-Pretraining-Specialized-v1.2/` | 53,621,158,028 | 49.94 | 90 | Synthetic specialized text. Subsets below. |
| `source-snapshots/HuggingFaceTB/smollm-corpus/` | 122,361,179,294 | 113.96 | 105 | English Cosmopedia-v2. Manifest `SOURCE_MANIFEST.json` records 104 parquet files, revision `3ba9d605774198c5868892d7a8deda78031a781f`, `total_bytes` 122,361,137,711, status `completed`. Schema: `prompt`, `text`, `token_length`, `audience`, `format`, `seed_data`. |
| `datasets/fineweb2-indic/` | 111,083,389,045 | 103.45 | 70 | Indic (and related) web. 54 language-script prefixes. Schema includes `text`, `url`, `language`, `language_score`, `language_script`, `minhash_cluster_size`. |
| `datasets/sangraha/verified/` | 217,875,912,671 | 202.91 | 538 | Curated Sangraha, including an English split. Schema on Hindi: `doc_id`, `text`, `type`. |
| `datasets/sangraha/synthetic/` | 408,149,462,763 | 380.12 | 1,792 | Synthetic Sangraha, native and Latin scripts. Hindi sample schema: `doc_id`, `text`. |
| `datasets/sangraha/unverified/` | 78,579,572,063 | 73.18 | 212 | Unverified Sangraha. Hindi sample schema: `doc_id`, `text`. |

Specialized subsets:

| Prefix | Bytes | GiB | Objects | Sampled signal |
| --- | ---: | ---: | ---: | --- |
| `.../v1/Nemotron-Pretraining-RQA/` | 184,550,910,116 | 171.88 | 175 | `part_000000`: 100,000 rows, ~25,605 uncompressed text bytes/row |
| `.../v1/Nemotron-Pretraining-STEM-SFT/` | 96,755,548,278 | 90.11 | 104 | 30 sampled rows: category `Nemotron-Pretraining-STEM-SFT`, model `DeepSeek-R1-0528`. ~24,486 text bytes/row on `part_000000` (200,000 rows) |
| `.../v1/Nemotron-Pretraining-Math-Textbooks/` | 30,841,263,255 | 28.72 | 13 | ~7,783 text bytes/row on `part_000000` (1,000,000 rows) |
| `.../v1/Nemotron-Pretraining-InfiniByte-Reasoning/` | 28,345,959,943 | 26.40 | 30 | Specialized reasoning text |
| `.../v1/Nemotron-Pretraining-Wiki-Rewrite/` | 9,880,668,134 | 9.20 | 7 | ~3,864 text bytes/row on `part_000000` (1,000,000 rows) |
| `.../v1/Nemotron-Pretraining-Scientific-Coding/` | 548,476,054 | 0.51 | 1 | Small code-adjacent set |
| `.../v1.1/Nemotron-Pretraining-Code-Concepts/` | 32,090,527,236 | 29.89 | 61 | Metadata fields include `function_name`, `tags`, `problem_prompt`, `generated_solution`. ~1,806 text bytes/row on `part_000000` (250,000 rows) |
| `.../v1.1/Nemotron-Pretraining-Multiple-Choice/` | 2,448,706,604 | 2.28 | 4 | Multiple-choice text |
| `.../v1.1/Nemotron-Pretraining-Unconditional-Algorithmic/` | 208,343,534 | 0.19 | 1 | Short algorithmic text |
| `.../v1.1/Nemotron-Pretraining-Formal-Logic/` | 73,074,239 | 0.07 | 1 | Formal logic |
| `.../v1.1/Nemotron-Pretraining-Economics/` | 62,693,323 | 0.06 | 1 | Economics |
| `.../v1.2/Nemotron-Pretraining-Fact-Seeking/` | 45,648,373,013 | 42.51 | 72 | Short documents. Footer sample of 590,209 / 8,000,000 rows on `part_000000` averaged ~264 uncompressed text bytes/row |
| `.../v1.2/Nemotron-Pretraining-Multiple-Choice/` | 7,294,607,966 | 6.79 | 16 | Multiple-choice text |
| `.../v1.2/Nemotron-Pretraining-Generative/` | 660,980,923 | 0.62 | 1 | Generative text |
| `.../v1.2/Nemotron-Pretraining-Moral-Scenarios/` | 17,196,126 | 0.02 | 1 | Moral scenarios |

FineWeb2-Indic languages at or above 1 GiB (41 smaller prefixes sum to 1,952,006,542 bytes, 1.82 GiB):

| Prefix | Bytes | Objects |
| --- | ---: | ---: |
| `hin_Deva` | 34,272,968,716 | 8 |
| `ben_Beng` | 23,894,953,700 | 5 |
| `tam_Taml` | 9,438,382,231 | 2 |
| `urd_Arab` | 6,868,759,323 | 2 |
| `npi_Deva` | 6,683,397,712 | 2 |
| `mar_Deva` | 6,278,522,796 | 2 |
| `mal_Mlym` | 5,915,341,648 | 2 |
| `tel_Telu` | 3,950,350,913 | 1 |
| `kan_Knda` | 3,526,351,166 | 1 |
| `guj_Gujr` | 3,338,859,951 | 1 |
| `sin_Sinh` | 2,012,430,591 | 1 |
| `pan_Guru` | 1,577,744,291 | 1 |
| `ory_Orya` | 1,373,319,465 | 1 |

Sangraha verified splits at or above 1 GiB. The rest of `verified/` (mai, gom, mni, brx, kas, sat, doi) is under 50 MB each.

| Split | Bytes | Objects |
| --- | ---: | ---: |
| `verified/hin` | 37,340,748,989 | 100 |
| `verified/eng` | 29,919,629,677 | 50 |
| `verified/ben` | 29,900,341,277 | 77 |
| `verified/tam` | 18,707,825,069 | 53 |
| `verified/nep` | 16,863,948,296 | 25 |
| `verified/tel` | 15,104,046,970 | 41 |
| `verified/mar` | 12,781,930,636 | 34 |
| `verified/urd` | 12,778,305,030 | 27 |
| `verified/mal` | 12,170,327,689 | 37 |
| `verified/guj` | 9,931,759,912 | 27 |
| `verified/kan` | 7,649,517,444 | 21 |
| `verified/san` | 4,182,068,434 | 11 |
| `verified/pan` | 4,144,576,198 | 12 |
| `verified/ori` | 3,920,475,843 | 11 |
| `verified/snd` | 1,411,291,244 | 2 |
| `verified/asm` | 979,454,978 | 3 |

`sangraha/synthetic/` is 28 language-script prefixes, almost all with 64 parquet files, about 10–19 GiB each. Fourteen are `*_Latn` companions of a native-script prefix. Treat each Latin prefix as a second view of the same synthetic source until an id join says otherwise.

### Counted token packs

These are the only token totals in the inventory. They use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and `uint16` tokens.

| Location | What the manifest counts |
| --- | --- |
| `s3://quark-research-repository-storage/data/nemotron_cc_v2_hqsynth_tinyllama_15b_parallel_20260730T185305Z/` | `data_manifest.json`: 15,000,000,001 stored train tokens and 50,000,001 stored validation tokens. Config target 15,000,000,000 train / 50,000,000 val. `seq_len` 2048. Source `nvidia/Nemotron-CC-v2` config `High-Quality-Synthetic`, text field `text`. 32 workers. Val partition: blake2b-64 of the document key, bucket 0, modulus 500. |
| `s3://quark-research-repository-storage/openhermes25_assistant_sft_20260803/` | `MANIFEST.json`, format `assistant_only_sft_v1`, dataset `shreydan/OpenHermes-2.5-Simple600k-SFT`. Train: 251,398,213 tokens, 587,987 docs, 180,633,498 supervised tokens. Val: 5,145,272 tokens, 12,013 docs, 3,699,398 supervised tokens. Loss on assistant content and its trailing EOS only. Val if `SHA256("openhermes25-sft-v1" + id) mod 100 < 2`. Rejected records: 0. |

`data/nemotron_cc_v2_hqsynth_tinyllama_15b_20260730T1840Z/` has `state.json` with status `running`, 986,891,321 train tokens toward a 15,000,000,001 target, and no token bin in that prefix. Leave it out.

`quark-research-repository-storage/test16/` (58,480,544,328 bytes, 5 objects) was not unpacked:

| Key | Bytes |
| --- | ---: |
| `test16/cosmo15b.bin` | 30,000,441,182 |
| `test16/nemotron_pretraining_fact_seeking_10b_tinyllama_tokenized_20260714.tar` | 20,000,071,680 |
| `test16/fineweb_edu_score_4_dedup_tinyllama_tokenized_20260714.tar` | 7,968,133,120 |
| `test16/openhermes_2_5_simple600k_sft_tinyllama_tokenized_20260714.tar` | 511,897,600 |
| `test16/test2_new_sources.summary.json` | 746 |

The summary JSON describes an older job (TinyStories 200,001,537 tokens, Cosmopedia 1,000,000,001 tokens, val 50,000,001 tokens) and says FineWeb-Edu was excluded from that job. It does not describe the four large files above. Do not add `test16` on top of the counted packs until the archives are opened and matched. The OpenHermes tar is the same dataset name as the manifest pack.

`data/nemotron-domain-classification/` is 29,290,009 bytes (14 parquet). Classifier output, not a mixture component. `architecture/` and `marketing/` are papers and notes.

### Translations

Committed translation prefixes overlap each other and overlap `translation-staging/`. Summing them would multi-count the same source documents. Staging is 976,308,359,536 bytes (33,731 objects: json, parquet, jsonl, gz) under `translation-staging/Cosmopedia-v2/Krutrim-decoupled/`. Train from committed `translations/` after a source-id dedup. Do not also train on staging.

Parquet bytes inside committed runs (json manifests excluded from this column):

| Prefix | Parquet bytes | Parquet objects | Use |
| --- | ---: | ---: | --- |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v4-20260905/` | 141,810,207,849 | 1,569 | Older Cosmopedia Indic run. Keep only source ids absent from v5. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v5-20260906/` | 55,220,425,599 | 588 | Preferred Cosmopedia Indic run where the same source id exists in v4. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-nemotron-cc-v21-hq-synth-v1-20260906/` | 52,802,126,378 | 771 | Indic translation of Nemotron-CC v2.1 high-quality synthetic. The English source for this slice is not stored as a full parquet tree. The counted English slice is the 15B-token pack. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-l4-first-batch-20260905-v3-quality-salvage/` | 5,663,476,359 | 66 | Salvage batch. Include only ids that failed into quarantine and then passed salvage, and that are absent from v5. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-l4-first-batch-20260904-v2-row-checkpoints/` | 2,545,200,911 | 30 | Checkpoint-era output. Same dedup rule. |
| `translations/Nemotron-Pretraining-Specialized-v1.2/Fact-Seeking/.../fact-seeking-70shard-10lang-r16-20260903/` | 183,215,542,075 | 3,655 | The Fact-Seeking Indic run large enough to train on. Other Fact-Seeking children of that prefix are smoke or pilot (each under ~0.3 GB). |
| `translations/Cosmopedia-v2/Krutrim/source-snapshots/` | 166,043,838,733 | 86 | English pipeline source snapshot, 86 parquet files. Distinct byte total from the HuggingFaceTB Cosmopedia snapshot (104 files, manifest `total_bytes` 122,361,137,711). Join on document id before using both. |
| `translations/Nemotron-CC-v2.1/High-Quality-Synthetic/` | 1,840,312 total, not a corpus | 155 | Review chunks (`translated_documents.parquet` on the order of tens of KB). Ignore for the mixture. |

Each committed chunk carries `quality_audit_report.json`, `quarantine_manifest.jsonl`, and `translated_documents.parquet`. Drop quarantined rows. That pass rate was not aggregated in this inventory.

### Uncompressed text samples

Measured from parquet footers (full file unless noted). These are UTF-8 byte lengths of the `text` column, not tokens.

| Shard | Rows in file | Uncompressed text bytes | Bytes/row |
| --- | ---: | ---: | ---: |
| `nemotron-cc-hq/part_000000.parquet` (file 1,229,506,288 bytes) | 976,959 | 3,172,635,310 | 3,247.5 |
| `nemotron-cc-hq/part_000400.parquet` (file 1,174,788,663 bytes) | 984,348 | 2,954,838,523 | 3,001.8 |
| `nemotron-cc-hq/part_000842.parquet` (file 256,524,514 bytes) | 186,696 | 672,964,269 | 3,604.6 |
| `Diverse-QA/part_000000.parquet` | 988,153 | 2,328,420,123 | 2,356.3 |
| `nemotron-cc-math-v1/4plus/part_000000.parquet` | 983,993 | 3,236,420,805 | 3,289.1 |
| `nemotron-cc-math-v1/3/part_000000.parquet` | 993,681 | 5,218,012,081 | 5,251.2 |
| `nemotron-cc-math-v1/4plus_MIND/part_000000.parquet` | 992,247 | 3,040,053,128 | 3,063.8 |
| `RQA/part_000000.parquet` | 100,000 | 2,560,468,754 | 25,604.7 |
| `STEM-SFT/part_000000.parquet` | 200,000 | 4,897,197,864 | 24,486.0 |
| `Math-Textbooks/part_000000.parquet` | 1,000,000 | 7,782,530,544 | 7,782.5 |
| `Wiki-Rewrite/part_000000.parquet` | 1,000,000 | 3,864,224,408 | 3,864.2 |
| `Code-Concepts/part_000000.parquet` | 250,000 | 451,514,377 | 1,806.1 |
| `Fact-Seeking/part_000000.parquet` | 8,000,000 | ~2,111,830,080 scaled from 590,209 rows | ~264 |
| `fineweb2-indic/hin_Deva/train/000_00000.parquet` | 2,910,000 | ~18,292,382,220 scaled from 2,000 rows | ~6,286 |
| `sangraha/verified/hin/data-0.parquet` | 174,763 | 982,289,888 | 5,620.7 |
| `sangraha/unverified/hin/data-0.parquet` | 149,796 | 988,762,945 | 6,600.7 |
| `sangraha/synthetic/hin_Deva/wiki_hin_Deva_0000_of_0063.parquet` | 90,237 | ~641,612,095 scaled from 2,000 rows | ~7,110 |
| Cosmopedia `train-00000-of-00104.parquet` | 376,289 | ~1,412,323,622 scaled from 2,000 rows | ~3,753 |

On the three CC-HQ shards, uncompressed text is 2.52× to 2.62× the stored file size. A full footer scan is required before anyone converts this bucket into a token count. The Cosmopedia `token_length` column, on the first row group of `train-00000` (1,000 rows), ranged from 28 to 2,011. That column’s tokenizer is not recorded in the schema, and that range is one row group.

## Token budget

Parameter count used for the budget: 1.2×10⁹.

| Plan | Tokens / parameter | Tokens seen | Role |
| --- | ---: | ---: | --- |
| Smaller | 10 | 12×10⁹ | Loader, dedup, and gate dry-run. Uses the counted 15B high-quality-synthetic pack as the backbone. |
| Chinchilla | 20 | 24×10⁹ | Compute-efficient pretraining if the run is short. |
| **Default** | **100** | **120×10⁹** | Overtraining past Chinchilla, still aimed at well under one pass of CC-HQ if the text-byte samples generalize. |
| Larger | 200 | 240×10⁹ | Use after the tokenizer census shows CC-HQ alone exceeds ~240B tokens. |

The default is 120B because recent small dense models are trained far past 20 tokens/parameter, and this store is large enough that 120B does not require repeating the crawl if the sampled CC-HQ expansion holds. Repeating CC-HQ, Sangraha synthetic, or Sangraha unverified is not part of any of these budgets. High-quality subsets may repeat, under the epoch caps below.

Supervised tokens are extra. OpenHermes train has 180,633,498 supervised tokens per epoch. One to three epochs (0.18B–0.54B supervised tokens) run after pretraining. They are not inside the 12B / 24B / 120B / 240B figures.

## Phase schedule

Fractions are of the pretraining token budget. The long-context phase is optional and starts only if Gate D passes. If it does not, add its 5% to annealing.

| Phase | Fraction | 12B | 24B (Chinchilla) | 120B default | 240B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 5% | 0.6B | 1.2B | 6B | 12B |
| Stable mix | 75% | 9.0B | 18B | 90B | 180B |
| Quality annealing | 15% | 1.8B | 3.6B | 18B | 36B |
| Long context | 5% | skip; fold into annealing | skip; fold into annealing | 6B if Gate D passes | 12B if Gate D passes |

On the 12B and 24B budgets, fold the 5% into annealing (annealing becomes 20%). A long-context tail that small will not move a 1.2B model.

### What is held out for annealing

Split every annealing source by document-id hash **before** the stable phase. Stable may see hash bit 0. Annealing sees hash bit 1. The two phases do not replay the same documents.

Reserved for annealing (hash bit 1), and upweighted there:

- `nemotron-cc-math-v1/4plus/` and `4plus_MIND/`
- Math textbooks, wiki rewrite, code concepts, formal logic, scientific coding
- English Cosmopedia-v2 (HuggingFaceTB snapshot, after the id join against `translations/.../source-snapshots/`)
- `sangraha/verified/` except `verified/eng`, which stays in the stable English curated slice
- English Fact-Seeking
- Deduped committed translations: Cosmopedia v5 (v4 only for missing ids), HQ-synthetic Indic v1, Fact-Seeking `r16`

Present in warmup and stable, and absent from annealing:

- `nemotron-cc-hq/`
- `nemotron-cc-math-v1/3/`
- FineWeb2-Indic
- `sangraha/synthetic/` native scripts
- `sangraha/unverified/`
- `sangraha/verified/eng`
- Diverse-QA at a low stable weight (synthetic QA is easy to memorize; it is not an annealing pillar)

Absent from pretraining entirely:

- OpenHermes train and val
- The HQ-synthetic val bin (50,000,001 tokens, hash bucket 0)
- `translation-staging/`
- Fact-Seeking smoke and pilot prefixes
- `translations/Nemotron-CC-v2.1/High-Quality-Synthetic/` review chunks
- Latin-script Sangraha synthetic (`*_Latn`), until an id join shows they are not copies of the native-script files
- FineWeb2 prefixes under 1 GiB, until a manual sample shows real prose rather than boilerplate. Their combined 1.82 GiB can be added later at a hard cap of one epoch
- `test16/` until unpacked
- The incomplete 20260730T1840Z tokenization run
- Domain-classifier parquet, papers, logs, CDK assets, bootstrap scripts

## Mixture weights

Weights are shares of **tokens seen in that phase**. They sum to 100. Apply epoch caps after the weight, so a small prefix cannot be repeated to fill its share. If a cap binds, give the unused share to `nemotron-cc-hq` in warmup and stable, and to math `4plus` + `4plus_MIND` in annealing.

### Epoch caps, whole pretraining run

| Stored size of the subset | Max epochs |
| --- | ---: |
| Under 1 GiB (economics, formal logic, moral scenarios, unconditional algorithmic, scientific coding, generative, FineWeb languages under 1 GiB) | 4 |
| 1–20 GiB (both multiple-choice subsets, wiki rewrite) | 3 |
| Larger specialized sets (RQA, STEM-SFT, textbooks, code concepts, Fact-Seeking, InfiniByte, Cosmopedia, math bands, Sangraha verified) | 2 |
| `nemotron-cc-hq`, FineWeb2-Indic, Sangraha synthetic, Sangraha unverified | 1 |

### Warmup

| Source | Weight |
| --- | ---: |
| `nemotron-cc-hq` | 70% |
| FineWeb2-Indic, languages ≥ 1 GiB, weight proportional to stored bytes | 8% |
| Math `4plus` + `4plus_MIND`, hash bit 0 only | 6% |
| Specialized bundle, hash bit 0, caps applied | 6% |
| Diverse-QA | 4% |
| Cosmopedia-v2 English, hash bit 0 | 3% |
| Sangraha verified Indic, hash bit 0, per-language cap so Hindi cannot take the whole 3% | 3% |

### Stable

| Source | Weight |
| --- | ---: |
| `nemotron-cc-hq` | 55% |
| Math `4plus` | 5% |
| Math `4plus_MIND` | 5% |
| Math score `3` | 2% |
| Diverse-QA | 3% |
| Specialized bundle (table below) | 8% |
| Cosmopedia-v2 English, hash bit 0 | 3% |
| FineWeb2-Indic ≥ 1 GiB, proportional to bytes | 5% |
| Sangraha verified Indic, hash bit 0, language-capped | 4% |
| Sangraha `verified/eng` | 2% |
| Sangraha synthetic, native script only, one epoch | 3% |
| Sangraha unverified, one epoch, drop a language if a 1,000-doc perplexity spot check loses to verified in that language | 1% |
| Deduped Indic translations, hash bit 0 | 4% |

Stable sums to 100. The translation 4% is split by parquet bytes after dedup among Cosmopedia v5, HQ-synthetic Indic v1, and Fact-Seeking r16.

Specialized bundle (the 6% warmup share and the 8% stable share use the same internal weights):

| Subset | Share of the bundle |
| --- | ---: |
| RQA | 20% |
| STEM-SFT | 15% |
| Math textbooks | 15% |
| Code concepts | 15% |
| InfiniByte reasoning | 10% |
| Wiki rewrite | 8% |
| Fact-Seeking English | 8% |
| Multiple-choice v1.1 and v1.2 together | 5% |
| Formal logic, economics, scientific coding, unconditional algorithmic, generative, moral scenarios | 4% combined, each still under its epoch cap |

RQA is 171.88 GiB and would dominate a byte-proportional specialized mix. The bundle weights exist so textbooks, code, and Fact-Seeking stay visible at 1.2B.

### Annealing

| Source | Weight |
| --- | ---: |
| Math `4plus`, hash bit 1 | 16% |
| Math `4plus_MIND`, hash bit 1 | 16% |
| Math textbooks, hash bit 1 | 10% |
| Wiki rewrite, hash bit 1 | 8% |
| Code concepts, hash bit 1 | 8% |
| Cosmopedia-v2 English, hash bit 1 | 12% |
| Sangraha verified Indic, hash bit 1 | 12% |
| Fact-Seeking English, hash bit 1 | 6% |
| Fact-Seeking r16 translations, hash bit 1 | 6% |
| Cosmopedia Indic translations (v5, plus v4 ids missing from v5), hash bit 1 | 6% |

Annealing has no CC-HQ, no score-3 math, no FineWeb, no Sangraha synthetic, and no Sangraha unverified.

### Long context, if Gate D passes

Fill this phase with documents whose length in the **training tokenizer** exceeds 4,096 tokens. Candidates, from the byte/row samples, are STEM-SFT and RQA (about 25 KB of text per row on the sampled files) and, second, math textbooks (~7.8 KB/row). Fact-Seeking (~264 bytes/row) and the sampled Cosmopedia `token_length` max of 2,011 do not feed this phase. If those long documents cannot fill the phase at under two epochs, Gate D fails and the tokens move to annealing.

## Quality, dedup, contamination

Do these before the first training step. They change the weights.

1. **CC-HQ internal dedup.** MinHash on `text`. The sampled schema has no WARC id.
2. **Math vs CC-HQ.** Math carries `warc_filename` and `warc_id`. CC-HQ does not. Dedup the pair with the same MinHash, and prefer the math row on a collision so scores survive.
3. **FineWeb2.** The column `minhash_cluster_size` is already present. Measure its distribution per language. Downweight clusters only after that histogram exists. No threshold is set here.
4. **Sangraha vs FineWeb2, per language.** Exact-hash normalized text. On a collision, keep Sangraha verified, then synthetic, then FineWeb, and drop unverified.
5. **Native vs Latin Sangraha synthetic.** Join on `doc_id`. If they match, drop `*_Latn` from the mixture.
6. **Translation runs.** Dedup on source document id across v4, v5, salvage, L4 checkpoints, HQ-synthetic v1, and staging. Keep a row only from `translated_documents.parquet` when the quarantine manifest does not list it. Prefer v5 over v4.
7. **Two English Cosmopedia snapshots.** Join HuggingFaceTB `smollm-corpus` (104 files) with `translations/Cosmopedia-v2/Krutrim/source-snapshots/` (86 files, 166,043,838,733 parquet bytes). Train on the union once.
8. **Holdouts carved before shuffling.**
   - OpenHermes val: 12,013 docs, 5,145,272 tokens. Out of pretraining and out of SFT train.
   - HQ-synthetic val bin: 50,000,001 tokens. Out of pretraining.
   - CC-HQ: 50M training tokens, document hash reserved, never in the train mix. This slice does not exist yet; create it at tokenize time.
   - Math `4plus`: 5M tokens reserved the same way.
   - Sangraha verified: 2,000 documents per language that has at least 10,000 documents, held out. For languages smaller than that, hold out 5% of docs.
9. **Translation quality.** Use the existing quarantine manifests. Do not impute a pass rate.
10. **PII and boilerplate.** Not measured in this inventory. Run a filter. Record the fraction removed per source. That fraction is a falsifier below.

## Ordering at 1.2B

Worth doing:

- The phase mixture above, with epoch caps and a hash split between stable and annealing.
- Warmup-stable-decay aligned with those phases (next section).
- Document packing to the phase sequence length, with an end-of-document token. If the stack has block-diagonal attention, use it so packed documents do not attend across the boundary.
- Shuffle sources inside the phase. Restart from a deterministic shard index.
- Sequence-length schedule: 2,048 in warmup, 4,096 in stable and annealing, 8,192 only in the optional tail.
- A single global learning rate. One AdamW state.

Leave out at this size:

- A difficulty sort across the whole run. The phase change is the curriculum. Sorting individual documents by perplexity needs a teacher model, and no teacher checkpoint is in these buckets.
- Per-source learning rates.
- A proxy mixture search (the quark bucket has the DoReMi, RegMix, and related PDFs). Run the weights in this document first. Start a proxy search only if Gate B fails.
- Training native-script and Latin Sangraha synthetic at full weight together.
- Multi-epoch crawl, FineWeb, or unverified Sangraha.
- A 16k or 32k pretraining length. Nothing measured here shows that the mass of documents is that long.

## Sequence length, batch, learning rate

Planning ranges, not a tuned sweep.

**Sequence length.** Warmup 2,048, which matches the counted HQ-synthetic pack (`seq_len` 2048). Stable and annealing 4,096. Optional tail 8,192, and only on documents longer than 4,096 in the training tokenizer. Pack short documents in their own stream during that tail so the batch is not padding.

**Global batch, in tokens.**

| Budget | Global batch | Steps (pretrain) |
| --- | ---: | ---: |
| 12B | 1×10⁶ | 12,000 |
| 24B | 1×10⁶ | 24,000 |
| 120B default | 2×10⁶ | 60,000 |
| 240B | 2×10⁶ to 4×10⁶ | 120,000 down to 60,000 |

Stay inside 0.5×10⁶ to 4×10⁶ tokens per step. Microbatch and GPU count are a systems choice; the curriculum constraint is global tokens. Default warmup is 6B / 2×10⁶ = 3,000 steps. Stable is 45,000 steps. Annealing is 9,000 steps. The optional long-context tail is 3,000 steps.

**Learning-rate shape.** AdamW. Warmup-stable-decay, one schedule:

- Linear warmup from 0 to the peak across the warmup phase (3,000 steps at the default).
- Constant peak through the stable phase.
- Linear or cosine decay from the peak to 0.1× peak across annealing.
- If the long-context tail runs, keep the decayed learning rate. Do not warm up again.

Peak learning-rate range at ~2×10⁶ tokens/step and 1.2B parameters: **3×10⁻⁴ to 6×10⁻⁴**. Start the default at **4×10⁻⁴**. Weight decay 0.1, betas 0.9 and 0.95, gradient clip 1.0. These optimizer constants are conventional defaults, not measurements from this data.

If the global batch moves from 2×10⁶ to 4×10⁶, try a peak of 5×10⁻⁴ to 6×10⁻⁴ and use the warmup gate. Doubling the batch does not justify doubling the learning rate.

## Evaluation gates

Scores below are comparisons to this run’s own curves. No external benchmark number is assumed.

**Gate A, end of warmup.** Train loss is down from step 500. Loss on the HQ-synthetic val bin and on the 50M-token CC-HQ holdout is below the loss at step 500. Hindi Sangraha-verified holdout loss is finite and lower than at the first eval. Stop the run if loss diverges and is still diverged 500 steps later. On a spike in the first 500 steps, restart from scratch at 2×10⁻⁴.

**Gate B, halfway through stable.** English holdout loss is still falling over the last 10% of steps. The math `4plus` holdout is not rising while web loss falls. OpenHermes val loss must not fall the way a memorized training set falls. OpenHermes is absent from pretraining; a collapse there means the crawl or the synthetic sets overlap the SFT data, and the supervised comparison is invalid.

**Gate C, start annealing.** Gate B has passed. Switch to the annealing mixture and start the learning-rate decay on the same step. Expect the textbook, math, wiki, and Sangraha-verified holdouts to improve. A rise on the CC-HQ holdout is acceptable in this phase. A rise on the math holdout or the Sangraha-verified holdout means the annealing mix is wrong; stop the upweight and return those tokens to the stable mixture.

**Gate D, long context.** Enter only when a length histogram, in the training tokenizer, shows enough documents longer than 4,096 tokens to fill the phase under the two-epoch cap. STEM-SFT and RQA are the first places to measure. If the histogram fails, fold the phase into annealing and finish at 4,096.

**Gate E, supervised tail.** Start after Gate C (and after Gate D if that phase ran). Train OpenHermes with the manifest loss mask: assistant tokens and the trailing EOS, 180,633,498 supervised tokens per epoch, one to three epochs. Eval on the 12,013-doc val split every epoch. Stop when val loss rises for two evals in a row. Keep the val split out of the train offsets.

## What would falsify this plan

1. **Tokenizer census.** After tokenization, unique CC-HQ tokens under about 200B make the 240B plan a crawl repeat. Cut it. Unique CC-HQ tokens well above 240B make the 120B default leave data unused; run 240B. The three sampled shards at 2.52×–2.62× stored size are not that census.
2. **Dedup yield.** If MinHash drops more than 30% of any source this plan treats as independent (CC-HQ vs math, v4 vs v5, the two Cosmopedia snapshots, native vs Latin Sangraha), recompute the weights before launch.
3. **Quarantine.** If the existing translation quarantine manifests remove a large share of a language, drop that language from annealing.
4. **Indic stall.** If the Sangraha-verified holdout is flat in the second half of stable while CC-HQ loss is still falling, raise verified Indic and the deduped translations inside the stable 4%+4% band. Leave unverified Sangraha at one epoch.
5. **Annealing regression.** If Gate C sees the math or textbook holdout get worse, the reserved slice is noisy or duplicated. Move those tokens back to the stable mix.
6. **Length histogram.** If almost all mass is under 2,048 training tokens, train the whole run at 2,048 and cancel 4,096 and 8,192.
7. **SFT leak.** If OpenHermes val tracks a memorization curve during pretraining, the pretrain mixture contains instruction text and Gate E is not a clean comparison.
8. **Filter shock.** If the PII/boilerplate filter removes a large unexpected fraction of CC-HQ, rebuild the token census before locking 120B vs 240B.
9. **Wrong stage.** A 1.2B checkpoint appearing outside this inventory falsifies from-scratch pretraining. Continued pretraining would use a much smaller token budget on the annealing sources plus Indic, and would not replay 120B of crawl.

## Order of work

1. Choose the tokenizer. Run the CC-HQ census on a full footer pass plus a token sample, and lock 120B vs 240B with falsifier 1.
2. Carve the holdouts. Dedup. Apply quarantine manifests. Drop Latin Sangraha synthetic if ids match.
3. Materialize the hash-bit split and the epoch caps as a shard manifest. Dry-run it on the 12B budget.
4. Pass Gate A, then Gate B, then anneal under Gate C.
5. Measure lengths. Run Gate D or fold it in.
6. Supervised tail on OpenHermes under Gate E.
