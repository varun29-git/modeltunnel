# Training curriculum: 1.2B dense decoder, English + Indic

Pretraining mix for a **1.2B-parameter dense decoder-only** model. English and Indic are in the same run. Data is already cleaned. The tokenizer is not chosen. Every billion below is the token count from Varun’s sheet, or an Indic figure marked as an assumption.

The 27 September 2026 inventory (account `023894314260`) is used only for measured sizes, prefixes, and the translation-dedup rule. It is not the token budget. This plan does not use a 120B run.

No model checkpoint is in that inventory, so these tokens are a from-scratch run. If a 1.2B checkpoint already exists, stop and write a continued-pretraining plan instead.

---

## Summary

The model reads **710B English tokens** from the twelve sources in the sheet, plus Indic in every phase.

**Indic share, default assumption: 15% of all training tokens.** English stays at the sheet’s 710B. Indic is then `710 × 0.15 / 0.85` = **125.294B**. The run is **835.294B** tokens, about 696 tokens per parameter. English alone is about 592 tokens per parameter, so this is a long run for a 1.2B model. The 15% is a planning choice, not a measurement.

Two alternatives use the same sources, the same phase order, and the same five-way Indic split:

| Choice | Indic share of the run | Indic tokens | Total tokens |
| --- | ---: | ---: | ---: |
| Lower | 8% | 61.739B | 771.739B |
| **Default (assumption)** | **15%** | **125.294B** | **835.294B** |
| Higher | 25% | 236.667B | 946.667B |

**Order.** Three phases. Indic is 15% of each one. Shares below are of that phase under the default.

1. **Warmup, 5% (41.765B).** ClimbMix is 59.86%. Math, code, and synthetic HQ together are 25.14%. Indic is 15%.
2. **Stable, 70% (584.706B).** ClimbMix is still the bulk, at 60.71%. Math, code, and synthetic HQ together are 24.29%. Indic is 15%.
3. **Quality annealing, 25% (208.824B).** ClimbMix falls to 9.58%. Math, code, and synthetic HQ together rise to 75.42%. Indic stays 15%.

Skip a separate long-context phase; the last quarter of tokens is the quality mix, and the inventory has no length histogram for these sources.

**English, as a share of the 710B seen total.** ClimbMix (general web) 400B, 56.34%. Math 121B, 17.04%. Code 85B, 11.97%. Synthetic HQ 104B, 14.65%. The sheet’s printed total is 755B. The crop above Finemath hides **45B**. That 45B has no source and no weight. It is not in the 835.294B.

**Indic, inside the 15%.** Weights use measured bytes where a prefix exists. IndicCorpV2 has no measured size, so it gets the average of the four measured sources, which is 20% of Indic.

| Indic source | Share of Indic | Tokens at 15% |
| --- | ---: | ---: |
| Sangraha | 38.23% | 47.898B |
| IndicCorpV2 | 20.00% | 25.059B |
| Translated Fact-Seeking | 26.28% | 32.925B |
| Translated Cosmopedia (v5 bytes only) | 7.92% | 9.923B |
| Translated Nemotron-CC HQ synthetic | 7.57% | 9.489B |

Hindi is the largest measured Sangraha language (19.87% of verified Indic bytes) and is capped at **25% of Indic tokens** in each phase. It does not receive other languages’ unused tokens.

**Known epoch collision.** The sheet allocates 50B to Nemotron-CC v2.1 high-quality synthetic. The only counted English pack is **15,000,000,001** train tokens. Cap that pack at **2 epochs** (30.0B on the tokenizer that counted it). Do not run a third pass to fill 50B.

**What changes the mix.** After warmup, a math holdout that is flat while ClimbMix is still falling moves 5B of stable ClimbMix onto Nemotron Math 4+. Halfway through stable, Indic holdouts that have stalled while English is still falling switch the rest of the run from 15% Indic to 25%. English loss that stalls while Indic is still falling switches the rest to 8%. A textbook or math holdout that rises in annealing returns the rest of annealing to the stable mix.

---

## English mixture

Visible rows sum to 710B: `9+400+52+35+10+15+50+40+45+30+15+9`. Shares are of that 710B. Roles are general web, math, code, synthetic HQ, and held for annealing. “Held for annealing” means most of that row’s tokens sit in the last phase. Math and code are also a larger share of annealing than of stable; the phase table is the schedule.

| Source | Billions | Share of 710B | Role |
| --- | ---: | ---: | ---: |
| Finemath-4plus | 9 | 1.27% | math |
| ClimbMix 400B | 400 | 56.34% | general web |
| Nemotron Math 4+ | 52 | 7.32% | math |
| Nemotron MIND 72B | 35 | 4.93% | math |
| Math Textbooks 25B | 10 | 1.41% | math, held for annealing |
| SwallowMath-v2 32B | 15 | 2.11% | math |
| Nemotron-CC v2.1 HQ Synth | 50 | 7.04% | synthetic HQ, held for annealing |
| CodeAlchemy 55B | 40 | 5.63% | code |
| Nemotron Code v2 45B | 45 | 6.34% | code |
| RQA 40B | 30 | 4.23% | synthetic HQ |
| Fact-Seeking 35B | 15 | 2.11% | synthetic HQ, held for annealing |
| Specialized v1.1 9B | 9 | 1.27% | synthetic HQ, held for annealing |
| **Seen total** | **710** | **100%** | |
| Unseen crop | 45 | — | no source, no weight |

The names “72B”, “25B”, “32B”, “55B”, “45B”, “40B”, “35B”, and “9B” are labels on the sheet. They are not inventory measurements. The training amount is the billions column.

Group totals inside the 710B: math `9+52+35+10+15` = 121B (17.04%), code `40+45` = 85B (11.97%), synthetic HQ `50+30+15+9` = 104B (14.65%), general web 400B (56.34%).

### Where each English row sits in storage

Bucket `nemotron-training-data-us-east-1` unless noted. “Not measured” means the inventory has no size for that source. Do not fill those cells with a public-dataset figure.

| Sheet row | Inventory | Measured size | Tokens measured? |
| --- | --- | --- | --- |
| Finemath-4plus | No prefix | Not measured | No |
| ClimbMix 400B | No prefix | Not measured | No |
| Nemotron Math 4+ | `datasets/nemotron-cc-math-v1/4plus/` | 62,189,080,483 bytes, 57.92 GiB, 46 objects | No |
| Nemotron MIND 72B | `datasets/nemotron-cc-math-v1/4plus_MIND/` | 90,331,100,051 bytes, 84.13 GiB, 90 objects | No |
| Math Textbooks 25B | `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/` | 30,841,263,255 bytes, 28.72 GiB, 13 objects | No |
| SwallowMath-v2 32B | No prefix | Not measured | No |
| Nemotron-CC v2.1 HQ Synth | Counted pack `s3://quark-research-repository-storage/data/nemotron_cc_v2_hqsynth_tinyllama_15b_parallel_20260730T185305Z/` | Manifest: 15,000,000,001 train tokens and 50,000,001 validation tokens. Source `nvidia/Nemotron-CC-v2`, config `High-Quality-Synthetic`. `seq_len` 2048. Tokenizer `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `uint16`. | Yes, this pack only |
| CodeAlchemy 55B | No prefix | Not measured | No |
| Nemotron Code v2 45B | No prefix | Not measured | No |
| RQA 40B | `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-RQA/` | 184,550,910,116 bytes, 171.88 GiB, 175 objects | No |
| Fact-Seeking 35B | `datasets/Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Fact-Seeking/` | 45,648,373,013 bytes, 42.51 GiB, 72 objects | No |
| Specialized v1.1 9B | `datasets/Nemotron-Pretraining-Specialized-v1.1/` | 34,883,344,936 bytes, 32.49 GiB, 68 objects | No |

Nearby prefixes that are **not** rows on the sheet, so they are not in this mix:

- `datasets/nemotron-cc-hq/` is 1,110,302,262,436 bytes (1,034.05 GiB). It is not ClimbMix and it is not the 45B gap.
- `datasets/nemotron-cc-math-v1/3/` is 107,417,646,757 bytes (100.04 GiB). Score band 3 is not the 4+ row.
- `datasets/nemotron-cc-v2/Diverse-QA/` is the only config stored under `datasets/nemotron-cc-v2/`. It is not Nemotron Code v2.
- `datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/` is 32,090,527,236 bytes (29.89 GiB). That is inside Specialized v1.1, not a substitute for Nemotron Code v2.
- English Cosmopedia at `source-snapshots/HuggingFaceTB/smollm-corpus/` is 122,361,179,294 bytes (113.96 GiB). It is not an English row on the sheet.
- `datasets/fineweb2-indic/` is 111,083,389,045 bytes (103.45 GiB). It is not IndicCorpV2.

Specialized v1.1 subsets, all measured, summing to the 32.49 GiB prefix. Draw the 9B in proportion to these bytes. Code-Concepts is 29.89 of 32.49 GiB, so it is about 92% of this row. The three subsets under 1 GiB stay in the row and then hit the epoch cap below.

| Subset | Bytes | GiB |
| --- | ---: | ---: |
| Code-Concepts | 32,090,527,236 | 29.89 |
| Multiple-Choice | 2,448,706,604 | 2.28 |
| Unconditional-Algorithmic | 208,343,534 | 0.19 |
| Formal-Logic | 73,074,239 | 0.07 |
| Economics | 62,693,323 | 0.06 |

One shard sample is not a corpus token count. Cited so the census knows what it is counting: Math 4+ `part_000000` has 3,289.1 uncompressed text bytes per row; MIND `part_000000` has 3,063.8; textbooks `part_000000` has 7,782.5; RQA `part_000000` has 25,604.7; Fact-Seeking `part_000000` averages about 264 text bytes per row (scaled from 590,209 of 8,000,000 rows); Code-Concepts `part_000000` has 1,806.1. Fact-Seeking is the short-document set. Its 15B allocation is the one most likely to surprise a token census. The census decides. This plan does not convert those byte-per-row figures into a corpus total.

`quark-research-repository-storage/test16/nemotron_pretraining_fact_seeking_10b_tinyllama_tokenized_20260714.tar` is 20,000,071,680 bytes and was not unpacked. The `10b` in the filename is not a measured token count. Leave `test16/` out.

---

## Phases

Fractions are of the combined run (English sheet + assumed Indic), not of the 710B alone. Default run, 835.294B:

| Phase | Fraction | Tokens | English | Indic |
| --- | ---: | ---: | ---: | ---: |
| Warmup | 5% | 41.765B | 35.500B | 6.265B |
| Stable | 70% | 584.706B | 497.000B | 87.706B |
| Quality annealing | 25% | 208.824B | 177.500B | 31.324B |

English phase tokens sum to 710B. Indic phase tokens sum to 125.294B. The same 5% / 70% / 25% split applies to the 8% and 25% Indic alternatives. Only the Indic column changes.

Annealing is a quarter of the run so math, code, textbooks, and synthetic HQ can dominate the last phase while 355B of the 400B ClimbMix allocation still lands in stable.

### Tokens by source and phase

English cells are the sheet, split across phases. They sum across each row to the billions in the mixture table. Indic cells are the default 15% assumption, rounded to 0.01B. The sampler should use the byte weights in the Indic section, not the rounded display.

Percentages are the share of **that phase**.

| Source | Warmup B | Stable B | Annealing B | Warmup | Stable | Annealing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Finemath-4plus | 0.5 | 4.0 | 4.5 | 1.20% | 0.68% | 2.15% |
| ClimbMix | 25.0 | 355.0 | 20.0 | 59.86% | 60.71% | 9.58% |
| Nemotron Math 4+ | 2.0 | 25.0 | 25.0 | 4.79% | 4.28% | 11.97% |
| Nemotron MIND | 1.5 | 16.5 | 17.0 | 3.59% | 2.82% | 8.14% |
| Math Textbooks | 0.5 | 3.0 | 6.5 | 1.20% | 0.51% | 3.11% |
| SwallowMath-v2 | 0.5 | 7.0 | 7.5 | 1.20% | 1.20% | 3.59% |
| HQ Synth | 1.0 | 17.0 | 32.0 | 2.39% | 2.91% | 15.32% |
| CodeAlchemy | 1.0 | 21.0 | 18.0 | 2.39% | 3.59% | 8.62% |
| Nemotron Code v2 | 1.5 | 23.5 | 20.0 | 3.59% | 4.02% | 9.58% |
| RQA | 1.0 | 17.0 | 12.0 | 2.39% | 2.91% | 5.75% |
| Fact-Seeking | 0.5 | 5.0 | 9.5 | 1.20% | 0.86% | 4.55% |
| Specialized v1.1 | 0.5 | 3.0 | 5.5 | 1.20% | 0.51% | 2.63% |
| Sangraha | 2.39 | 33.53 | 11.97 | 5.73% | 5.73% | 5.73% |
| IndicCorpV2 | 1.25 | 17.54 | 6.26 | 3.00% | 3.00% | 3.00% |
| Translated Cosmopedia | 0.50 | 6.95 | 2.48 | 1.19% | 1.19% | 1.19% |
| Translated HQ synth | 0.47 | 6.64 | 2.37 | 1.14% | 1.14% | 1.14% |
| Translated Fact-Seeking | 1.65 | 23.05 | 8.23 | 3.94% | 3.94% | 3.94% |

English subtotals: warmup 35.5B (85%), stable 497B (85%), annealing 177.5B (85%). Indic subtotals: 6.26B, 87.71B, 31.32B (15% each). Rounded Indic cells can miss the exact total by 0.01B.

Every math row, both code rows, and every synthetic-HQ row is a larger share of annealing than of stable. ClimbMix goes the other way (60.71% of stable, 9.58% of annealing).

Grouped, as a share of the phase:

| Group | Warmup | Stable | Annealing |
| --- | ---: | ---: | ---: |
| ClimbMix | 59.86% | 60.71% | 9.58% |
| Math (five rows) | 11.97% | 9.49% | 28.97% |
| Code (two rows) | 5.99% | 7.61% | 18.20% |
| Synthetic HQ (four rows) | 7.18% | 7.18% | 28.25% |
| Indic (five sources) | 15.00% | 15.00% | 15.00% |

Grouped shares are rounded to 0.01 percentage point. The stable row sums to 99.99 after that rounding. Warmup and annealing sum to 100.

Textbooks are 0.51% of stable and 3.11% of annealing. HQ synth is 2.91% of stable and 15.32% of annealing. Fact-Seeking is 0.86% of stable and 4.55% of annealing. Specialized v1.1 is 0.51% of stable and 2.63% of annealing.

### How to build the streams

Assign each document to one phase before training, by a hash of the document id, in proportion to that source’s warmup : stable : annealing billions. A document is in one phase. Repeats allowed by an epoch cap stay inside the assigned phase.

Hold out, and never train on:

- The existing HQ-synth validation bin: 50,000,001 tokens, blake2b-64 of the document key, bucket 0, modulus 500.
- Created at tokenize time (assumption): 50 million tokens from ClimbMix, 5 million from Nemotron Math 4+, 5 million from Math Textbooks, 5 million from whichever code source is tokenized first, and 2,000 documents per Sangraha verified language that has at least 10,000 documents (otherwise 5% of that language’s documents). For each translation source, hold out 2,000 documents per language when that many exist.

Decay the learning rate across annealing. Do not warm it back up. Optimizer constants are not fixed by this data.

The HQ-synth row in the table is the sheet’s 50B target. The stored pack cannot fill it. See the epoch cap below before launching that row.

---

## Indic

Indic is in warmup, stable, and annealing at the same share. It is not a second run and it is not saved for annealing only.

### Default share

**Assumption: Indic is 15% of training tokens.** Lower alternative 8%. Higher alternative 25%.

Why 15% as the default: the English sheet is already fixed at 710B, and a 1.2B Indic track still needs a large absolute count because the tokens are split across many languages. 125.294B is about 104 tokens per parameter for Indic as a group. 8% (61.739B) is the choice if English holdout loss is the only target. 25% (236.667B) is the choice if Indic quality is the point of the model. Gates below switch among these three. They do not invent a fourth share.

### Five-way split

Measured bytes are the sampling weights. IndicCorpV2 is not measured. Its weight is the mean of the four measured byte totals. With one unmeasured source, that mean is exactly one fifth of the Indic budget. The other four fifths follow the bytes.

Sangraha’s weight uses the two Indic prefixes whose bytes are fully measured and are not a Latin twin of another prefix:

- `datasets/sangraha/verified/` except `verified/eng`: 217,875,912,671 − 29,919,629,677 = **187,956,282,994 bytes** (175.05 GiB).
- `datasets/sangraha/unverified/`: **78,579,572,063 bytes** (73.18 GiB).

Together, **266,535,855,057 bytes**. `verified/eng` is 29,919,629,677 bytes (27.86 GiB) of English. It is not part of the Indic budget, and it is not an English row on the sheet. Leave it out.

`datasets/sangraha/synthetic/` is measured at 408,149,462,763 bytes (380.12 GiB, 1,792 objects, 28 language-script prefixes). Fourteen prefixes are `*_Latn` companions of a native-script prefix. Treat each Latin prefix as a second view of the same synthetic source until a `doc_id` join says otherwise. Unique native-script bytes are not measured, so synthetic bytes are not in the five-way weight. Synthetic is still read, inside the Sangraha allocation, under the draw rules below.

Translated Cosmopedia’s weight uses the preferred v5 run only: **55,220,425,599** parquet bytes. v4-only ids are extra documents inside this same source. They do not get a second share. The deduped union was not measured.

| Source | Weight basis | Bytes | Share of Indic |
| --- | --- | ---: | ---: |
| Sangraha | Verified Indic + unverified | 266,535,855,057 | 38.2285% |
| Translated Cosmopedia | v5 parquet bytes | 55,220,425,599 | 7.9201% |
| Translated HQ synth | v1 parquet bytes | 52,802,126,378 | 7.5733% |
| Translated Fact-Seeking | r16 parquet bytes | 183,215,542,075 | 26.2781% |
| IndicCorpV2 | Mean of those four | 139,443,487,277.25 | 20.0000% |

The four measured totals sum to 557,773,949,109. The mean is that sum divided by 4. All five weights sum to 697,217,436,386.25. Recompute this table if a real IndicCorpV2 size arrives, or if the Cosmopedia id join shows a large v4-only remainder.

At the default 125.294B Indic tokens, the budgets are Sangraha 47.898B, IndicCorpV2 25.059B, translated Cosmopedia 9.923B, translated HQ synth 9.489B, translated Fact-Seeking 32.925B. At 8% and at 25%, multiply those budgets by `61.739 / 125.294` and `236.667 / 125.294`.

### Map of the five names

| Name the user gave | Prefix | Measured |
| --- | --- | --- |
| Sangraha | `datasets/sangraha/verified/`, `datasets/sangraha/synthetic/`, `datasets/sangraha/unverified/` | 202.91 GiB, 380.12 GiB, 73.18 GiB |
| IndicCorpV2 | No prefix | Not measured |
| Translated Cosmopedia | `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v5-20260906/` and the v4 rule below | v5: 55,220,425,599 parquet bytes, 588 objects. Deduped union not measured |
| Translated Nemotron-CC HQ synthetic | `translations/Cosmopedia-v2/Krutrim/krutrim-nemotron-cc-v21-hq-synth-v1-20260906/` | 52,802,126,378 parquet bytes, 771 objects (49.18 GiB) |
| Translated Nemotron Fact-Seeking | `translations/Nemotron-Pretraining-Specialized-v1.2/Fact-Seeking/.../fact-seeking-70shard-10lang-r16-20260903/` | 183,215,542,075 parquet bytes, 3,655 objects (170.63 GiB) |

The Fact-Seeking path is quoted with the inventory’s ellipsis. The run name says 70 shards and 10 languages. Per-language bytes for that run were not measured. Other Fact-Seeking children of that prefix are smoke or pilot, each under about 0.3 GB. Train on the r16 run only.

`translations/Nemotron-CC-v2.1/High-Quality-Synthetic/` is 1,840,312 bytes of review chunks. Ignore it.

IndicCorpV2 is not `datasets/fineweb2-indic/` (103.45 GiB). FineWeb2-Indic is not one of the five named sources. It is not in this mix.

### Inside Sangraha

**Assumption**, because synthetic’s unique size is not measured. Of Sangraha’s tokens:

| Slice | Share of Sangraha | Epoch cap |
| --- | ---: | ---: |
| Verified Indic, excluding `verified/eng` | 70% | 3 per language |
| Synthetic, native script only | 20% | 1 |
| Unverified | 10% | 1 |

The 20% synthetic reserve is carved out of Sangraha’s 38.23%. It is not added on top. Drop `*_Latn` until a `doc_id` join shows those files are not copies. If one epoch of native synthetic is smaller than the 20% reserve, do not repeat it.

Unverified’s per-language bytes are not in the inventory. Until they are counted, give unverified the same language fractions as verified Indic. That copy is an assumption.

### Per-language balance

Sample Sangraha verified in proportion to stored bytes. Hindi is 19.87% of verified Indic, not 100%. These shares are of the verified-Indic slice (187,956,282,994 bytes).

| Split | Language | Bytes | Share of verified Indic |
| --- | --- | ---: | ---: |
| `hin` | Hindi | 37,340,748,989 | 19.87% |
| `ben` | Bengali | 29,900,341,277 | 15.91% |
| `tam` | Tamil | 18,707,825,069 | 9.95% |
| `nep` | Nepali | 16,863,948,296 | 8.97% |
| `tel` | Telugu | 15,104,046,970 | 8.04% |
| `mar` | Marathi | 12,781,930,636 | 6.80% |
| `urd` | Urdu | 12,778,305,030 | 6.80% |
| `mal` | Malayalam | 12,170,327,689 | 6.48% |
| `guj` | Gujarati | 9,931,759,912 | 5.28% |
| `kan` | Kannada | 7,649,517,444 | 4.07% |
| `san` | Sanskrit | 4,182,068,434 | 2.23% |
| `pan` | Punjabi | 4,144,576,198 | 2.21% |
| `ori` | Odia | 3,920,475,843 | 2.09% |
| `snd` | Sindhi | 1,411,291,244 | 0.75% |
| `asm` | Assamese | 979,454,978 | 0.52% |
| `mai`, `gom`, `mni`, `brx`, `kas`, `sat`, `doi` | seven small splits | 89,664,985 together | 0.05% |

Rounded shares sum to 100.02. Sample from the byte column.

**Cap: no language gets more than 25% of Indic tokens in a phase**, summed across Sangraha, IndicCorpV2, and the three translation sources. If a source would push Hindi over 25%, downsample Hindi in that source and spread the extra across the other languages in that source that are still under their epoch cap.

Translation runs and IndicCorpV2 have no measured per-language bytes. Until a count exists, split each of those sources evenly across the languages present in that source, then apply the 25% cap. The even split is an assumption. The Fact-Seeking prefix name says 10 languages; the inventory does not list them. Do not assume the run is Hindi.

Assamese (0.91 GiB) and the seven small splits (0.08 GiB together) are included. They use the repeat cap below. They are not dropped to make the Hindi math easier, and they are not upsampled without a cap.

### Repeat cap

Apply this when the token allocation would walk a small set more times than the cap. Unused tokens are not given to Hindi.

| Slice | Max epochs |
| --- | ---: |
| Any language split or subset stored under 1 GiB | 4 |
| Sangraha verified language at or above 1 GiB | 3 |
| Sangraha synthetic, native script | 1 |
| Sangraha unverified | 1 |
| A translation source (after the dedup below) | 3 |
| IndicCorpV2, once a size exists | 1 if it is web-scale (tens of GiB or more); 3 if the measured set is small |

If a cap binds inside Indic, move the unused tokens to another **named** Indic source that is still under 1 epoch. If none is under 1 epoch, the Indic total shrinks. Record the shortfall. Do not invent a sixth Indic source to fill it.

At the default, translated Cosmopedia is 9.923B tokens against 51.43 GiB of v5 parquet, and translated HQ synth is 9.489B against 49.18 GiB. Token totals for those prefixes were not counted, so this plan does not claim the cap already binds. Run the census, then apply the 3-epoch cap.

---

## Combined budget

| Piece | Tokens | Status |
| --- | ---: | --- |
| English, twelve visible rows | 710B | Sheet |
| Indic at the 15% assumption | 125.294B | Assumption |
| **Default run** | **835.294B** | |
| Unseen crop | 45B | Gap. No source. Not added. |

If the missing 45B is real English, the sheet total of 755B replaces 710B and the 15% assumption scales with it:

`Indic = 755 × 0.15 / 0.85` = **133.235B**. Total **888.235B**.

The phase fractions stay 5% / 70% / 25%. The five-way Indic split stays the same. The 45B is not assigned to ClimbMix, to math, or to a new row. When the source is known, put it in the phase that matches its role: with ClimbMix if it is general web, or with the annealing upweight if it is math, code, textbooks, or high-quality synthetic. Until then, train 710B of English.

Holding Indic fixed at 125.294B while adding 45B of English would drop the Indic share from 15% to 14.2%. The rule above keeps the share and scales Indic instead. Same rule at the alternatives: 8% of a 755B English base is 65.652B Indic (820.652B total); 25% is 251.667B Indic (1,006.667B total).

---

## Epoch caps

A cap is a maximum number of passes over the stored documents. When it binds, train fewer tokens. Do not raise another English row to absorb the gap. ClimbMix in particular does not grow past 400B.

### HQ synth, the cap that already binds

The sheet says 50B. Storage has one finished counted pack:

`s3://quark-research-repository-storage/data/nemotron_cc_v2_hqsynth_tinyllama_15b_parallel_20260730T185305Z/`

- Train tokens: 15,000,000,001
- Validation tokens: 50,000,001 (held out of every phase)
- Tokenizer: TinyLlama, as above

**Cap: 2 epochs of those train documents.** On that tokenizer, 2 epochs are 30.0B tokens. The other 20B of the sheet’s 50B is not in this pack. A third epoch is how those 20B would be manufactured. Do not take it.

The full English High-Quality-Synthetic text is not stored as a parquet tree. The Indic translation of that slice is a different source and does not count as English tokens. If the full English text is added later and the **training** tokenizer counts at least 50B unique tokens on it, use the phase table’s 1B / 17B / 32B schedule, still at most 2 epochs.

Until then, on the TinyLlama count, place the tokens like this:

- First epoch (15B): warmup 0.83B and stable 14.17B, which is the table’s 1:17 ratio.
- Second epoch (15B): annealing only.
- Unfilled versus the sheet: 0.17B warmup, 2.83B stable, 17B annealing.

Recompute those three numbers as `2 × T` after the training tokenizer is applied to the same documents. `T` is not known until then. The TinyLlama figure is a count of stored tokens, not a promise that every future tokenizer will emit 15B.

Leave out the unfinished run `data/nemotron_cc_v2_hqsynth_tinyllama_15b_20260730T1840Z/`. Its `state.json` says `running`, with 986,891,321 train tokens toward a 15,000,000,001 target, and no token bin in that prefix.

If the HQ cap drops about 20B of English and Indic stays at 125.294B, Indic’s share of the tokens actually trained drifts from 15% to about 15.4%. Keep the Indic token count fixed in that case. The drift is smaller than the gap between the 8%, 15%, and 25% choices.

### Other English rows

Tokens for these rows were not counted. The cap is the rule for the census. It is not a claim that the allocation already overruns the files.

| Source | Cap | Why this number |
| --- | ---: | --- |
| ClimbMix | 1 epoch | Allocation is 400B and the name says 400B. Not in the inventory. If the unique count is under 400B, train one pass and stop. |
| Nemotron Code v2 | 2 epochs | Allocation equals the 45B in the name. Not in the inventory. |
| Specialized v1.1, Code-Concepts and Multiple-Choice | 2 epochs | The parent prefix is 32.49 GiB. Tokens not measured. |
| Specialized v1.1 subsets under 1 GiB (formal logic, economics, unconditional algorithmic) | 4 epochs | 0.07 GiB, 0.06 GiB, 0.19 GiB. |
| Finemath, Nemotron Math 4+, MIND, textbooks, SwallowMath, CodeAlchemy, RQA, Fact-Seeking | 2 epochs | Several names are larger than the allocation (MIND 72B vs 35B trained, textbooks 25B vs 10B, SwallowMath 32B vs 15B, CodeAlchemy 55B vs 40B, RQA 40B vs 30B, Fact-Seeking 35B vs 15B). Those names were not measured here. If they are real, the cap does not bind. Finemath, SwallowMath, and CodeAlchemy are not in the inventory at all. |

Nemotron Math 4+ is 57.92 GiB on disk against a 52B token allocation. Fact-Seeking is 42.51 GiB against 15B, and the sampled rows are short. Either one might fail the census. Wait for the census. Do not turn GiB into tokens in advance.

---

## Translation dedup

Committed Cosmopedia runs overlap each other and overlap staging. Summing them would train the same source documents more than once. Dedup on source document id. Train from `translated_documents.parquet` only. Drop any row listed in that chunk’s `quarantine_manifest.jsonl`. The quarantine pass rate was not aggregated. Do not impute one.

| Run | Parquet bytes | Objects | Rule |
| --- | ---: | ---: | --- |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v5-20260906/` | 55,220,425,599 | 588 | Preferred. Keep when the source id exists here. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v4-20260905/` | 141,810,207,849 | 1,569 | Keep only source ids absent from v5. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-l4-first-batch-20260905-v3-quality-salvage/` | 5,663,476,359 | 66 | Keep only ids that failed into quarantine, then passed salvage, and are absent from v5. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-l4-first-batch-20260904-v2-row-checkpoints/` | 2,545,200,911 | 30 | Same as salvage: ids absent from v5 only. |
| `translation-staging/Cosmopedia-v2/Krutrim-decoupled/` | 976,308,359,536 (all objects, not only parquet) | 33,731 | Leave out. |
| `translations/Cosmopedia-v2/Krutrim/krutrim-nemotron-cc-v21-hq-synth-v1-20260906/` | 52,802,126,378 | 771 | Separate source (HQ synth). Not part of the Cosmopedia sum. |
| Fact-Seeking `fact-seeking-70shard-10lang-r16-20260903` | 183,215,542,075 | 3,655 | The Fact-Seeking translation to train. |
| `translations/Cosmopedia-v2/Krutrim/source-snapshots/` | 166,043,838,733 | 86 | English pipeline snapshot, not Indic text. |

v4 is larger than v5 on disk (132.07 GiB vs 51.43 GiB). That does not make v4 the preferred run. Overlap between them was not measured, so there is no deduped byte total to put in the budget. The Indic weight uses v5’s bytes only.

The English HuggingFaceTB Cosmopedia snapshot (104 parquet files, manifest `total_bytes` 122,361,137,711) and `translations/.../source-snapshots/` (86 files, 166,043,838,733 bytes) are two English copies. They are not in the 710B. If they are ever trained, join on document id and train the union once.

---

## Evaluation gates

Scores are this run’s own holdout losses. No external benchmark number is used. Check English and Indic separately. Eval at least Hindi, Bengali, and Tamil on the Sangraha verified holdouts. Those three are the largest measured verified splits (34.78 GiB, 27.85 GiB, 17.42 GiB).

**Gate A, end of warmup.**

- English. ClimbMix holdout loss and Nemotron Math 4+ holdout loss are both lower than at the midpoint of warmup. If the math holdout is flat or rising while ClimbMix is falling, the warmup was too web-heavy. Before stable, move **5B** of stable ClimbMix onto stable Nemotron Math 4+ (355B → 350B, and 25B → 30B). Leave the sheet’s totals otherwise unchanged.
- Indic. Hindi, Bengali, and Tamil holdout losses are finite and lower than the first eval. If Hindi improves and Bengali or Tamil does not, the stream is over-sampling Hindi. Enforce the verified byte shares and the 25% language cap before stable.

**Gate B, halfway through stable.**

- English. ClimbMix holdout loss is still falling over the last 10% of steps. The Math 4+ holdout is not rising. The HQ-synth validation bin does not fall the way a memorized training set falls. If that val loss collapses, the 15B pack is leaking or has already been repeated enough. Cut HQ synth to **1 epoch** for the rest of the run and drop the second pass.
- Indic. Hindi, Bengali, and Tamil are all still falling. If all three are flat while ClimbMix is still falling, Indic is too small: train the remaining stable tokens and all of annealing at the **25%** Indic alternative. If ClimbMix stalls and those three Indic losses are still falling, train the remainder at the **8%** alternative.
- One small language whose loss falls much faster than the others and approaches the training loss is being memorized. Stop repeating it. Keep the epochs it has already had. Do not refill that language’s slot with Hindi.

**Gate C, annealing.**

- English. Textbook, Math 4+, code, and HQ-synth holdouts should improve relative to the end of stable. A rise on the ClimbMix holdout is acceptable in this phase. If the textbook holdout or the Math 4+ holdout rises for two evals in a row, the upweight is wrong. For the rest of annealing, sample with the **stable** percentages instead of the annealing percentages. Keep decaying the learning rate.
- Indic. Sangraha verified holdouts should improve. If the translated Fact-Seeking holdout rises while Sangraha verified improves, cut the remaining annealing tokens of translated Fact-Seeking in half and give that half to Sangraha verified, still under the language cap and the epoch cap. If Sangraha verified itself rises for two evals, switch the rest of annealing to the stable mix, same as the English rule.

OpenHermes (`openhermes25_assistant_sft_20260803`, 251,398,213 train tokens) is an SFT pack in the inventory. It is not part of these three phases.

---

## Tokenizer

The tokenizer is unset. The sheet’s billions are the user’s token counts. Use them as the training tokenizer’s counts once that tokenizer exists.

The counted HQ pack and the OpenHermes pack were tokenized with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6` on the OpenHermes manifest). That is an English-only tokenizer. On Indic text it splits words into many more pieces than it does on English. The same Indic documents then take more of each sequence and more of the optimizer steps than the 15% document share. In practice the Indic share of training steps rises, and the English share shrinks, relative to the budgets in this plan. A multilingual tokenizer keeps the 15% closer to the share written here. Retokenizing the 15B TinyLlama pack does not turn it into the sheet’s 50B.

---

## Before the first step

1. Choose the tokenizer. Recount the HQ-synth pack with it. Lock the 2-epoch cap in that tokenizer’s tokens.
2. Dedup Cosmopedia on source id. Prefer v5. Keep v4, salvage, and checkpoint rows only under the rules above. Leave staging out. Drop quarantined rows. Drop Sangraha `*_Latn` until the id join. Drop `verified/eng` from the Indic stream.
3. Count unique training tokens per source and per Indic language. Apply the epoch caps. Write down every shortfall. Do not reassign an English shortfall to ClimbMix.
4. Hash documents into warmup, stable, and annealing in the ratios in the phase table. Carve the holdouts first.
5. Train warmup. Gate A. Stable. Gate B. Annealing. Gate C.
