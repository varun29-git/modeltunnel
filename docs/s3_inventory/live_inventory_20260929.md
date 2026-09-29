# Live S3 inventory — 2026-09-29T15:44:48.960834+00:00

Bucket: `s3://nemotron-training-data-us-east-1` (read-only ListObjects + one Parquet footer/row sample per family).
Sizes are ListObjects byte sums; document/token counts are estimates labeled by basis.
Token estimate basis: measured on the bounded S3 sample (13,717 docs) with the frozen 65k tokenizer.

## English / math / code sources

| source | prefix | objects | GiB | est. tokens (B) | schema text/id columns |
| --- | --- | ---: | ---: | ---: | --- |
| climbmix-400b-shuffle | `datasets/nemotron-climbmix/karpathy-400b-shuffle/` | 13089 | 558.6 | 137.2 (measured-fertility) | text=text id=- |
| code-concepts | `datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/` | 61 | 29.9 | 10.3 (measured-fertility) | text=text id=- |
| math-textbooks | `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/` | 13 | 28.7 | 10.7 (measured-fertility) | text=text id=- |
| nemotron-cc-hq | `datasets/nemotron-cc-hq/` | 843 | 1034.0 | 253.9 (measured-fertility) | text=text id=id |
| nemotron-cc-math-3 | `datasets/nemotron-cc-math-v1/3/` | 57 | 100.0 | 37.3 (measured-fertility) | text=text id=id |
| nemotron-cc-math-4plus | `datasets/nemotron-cc-math-v1/4plus/` | 46 | 57.9 | 21.6 (measured-fertility) | text=text id=id |
| nemotron-cc-math-4plus-mind | `datasets/nemotron-cc-math-v1/4plus_MIND/` | 90 | 84.1 | 31.3 (measured-fertility) | text=text id=id |
| scientific-coding | `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding/` | 1 | 0.5 | 0.2 (measured-fertility) | text=text id=- |
| smollm-corpus-snapshot | `source-snapshots/HuggingFaceTB/smollm-corpus/` | 105 | 114.0 | 28.0 (measured-fertility) | text=text id=- |

## Sangraha (campaign languages only)

| tier/lang | objects | GiB | est. tokens (B) |
| --- | ---: | ---: | ---: |
| synthetic/ben_Beng (bn) | 64 | 16.3 | 2.0 |
| synthetic/eng_Latn (en) | 0 | 0.0 | 0.0 |
| synthetic/guj_Gujr (gu) | 64 | 16.1 | 2.0 |
| synthetic/hin_Deva (hi) | 64 | 16.0 | 2.0 |
| synthetic/kan_Knda (kn) | 64 | 17.3 | 2.1 |
| synthetic/mar_Deva (mr) | 64 | 16.5 | 2.0 |
| synthetic/tam_Taml (ta) | 64 | 17.9 | 2.2 |
| synthetic/tel_Telu (te) | 64 | 16.9 | 2.1 |
| unverified/ben (bn) | 36 | 12.7 | 1.6 |
| unverified/eng (en) | 0 | 0.0 | 0.0 |
| unverified/guj (gu) | 4 | 1.3 | 0.2 |
| unverified/hin (hi) | 113 | 39.8 | 4.9 |
| unverified/kan (kn) | 4 | 1.4 | 0.2 |
| unverified/mar (mr) | 7 | 2.3 | 0.3 |
| unverified/tam (ta) | 18 | 5.6 | 0.7 |
| unverified/tel (te) | 6 | 2.2 | 0.3 |
| verified/ben (bn) | 77 | 27.8 | 3.4 |
| verified/eng (en) | 50 | 27.9 | 6.8 |
| verified/guj (gu) | 27 | 9.2 | 1.1 |
| verified/hin (hi) | 100 | 34.8 | 4.3 |
| verified/kan (kn) | 21 | 7.1 | 0.9 |
| verified/mar (mr) | 34 | 11.9 | 1.5 |
| verified/tam (ta) | 53 | 17.4 | 2.1 |
| verified/tel (te) | 41 | 14.1 | 1.7 |

## FineWeb2-Indic (campaign languages)

| partition | lang | objects | GiB | est. tokens (B) |
| --- | --- | ---: | ---: | ---: |
| ben_Beng | bn | 5 | 22.3 | 2.7 |
| guj_Gujr | gu | 1 | 3.1 | 0.4 |
| hin_Deva | hi | 8 | 31.9 | 3.9 |
| kan_Knda | kn | 1 | 3.3 | 0.4 |
| mar_Deva | mr | 2 | 5.8 | 0.7 |
| tam_Taml | ta | 2 | 8.8 | 1.1 |
| tel_Telu | te | 1 | 3.7 | 0.5 |

## Committed translations (staging excluded)

| run | objects | translated GiB | language partitions (GiB) |
| --- | ---: | ---: | --- |
| cosmopedia-v4 | 1569 | 132.1 | ben_Beng: 22.8, guj_Gujr: 22.8, hin_Deva: 21.9, pan_Guru: 17.6, tam_Taml: 23.7, tel_Telu: 23.3 |
| cosmopedia-v5 | 588 | 51.4 | kan_Knda: 13.6, mal_Mlym: 12.6, mar_Deva: 13.9, ory_Orya: 11.3 |
| fact-seeking-r16 | 3655 | 170.6 | ben_Beng: 16.8, guj_Gujr: 16.7, hin_Deva: 16.7, kan_Knda: 17.0, mal_Mlym: 17.6, mar_Deva: 17.3, ory_Orya: 17.0, pan_Guru: 17.1, tam_Taml: 17.5, tel_Telu: 16.9 |
| hq-synth-v1 | 0 | 0.0 | missing: prefix has no objects |

## Presence checks (named sources)

- `datasets/CodeAlchemy/`: ABSENT
- `datasets/SwallowMath/`: ABSENT
- `datasets/finemath/`: ABSENT
- `datasets/nemotron-code/`: ABSENT
- `translation-benchmark/`: present
- `translation-staging/`: present

Excluded from training: `translation-staging/`, `translation-benchmark/`, TinyLlama-tokenized packs in `quark-research-repository-storage` (incompatible token IDs).
