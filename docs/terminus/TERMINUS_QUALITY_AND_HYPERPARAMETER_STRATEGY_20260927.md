# Terminus: a cost-controlled quality and hyperparameter strategy

Date: 27 September 2026. Scope: maintained `terminus/`, not `terminus_next/`.

**Recommendation:** use ModelTunnelv2's task-aware evaluation, validated proxy transfer, and staged experiments to search Terminus's full design space. That includes model size and shape, tokenizer, data, effective batch, optimizer, precision, quantizer, dense/ternary schedule, context, and training duration. Reuse public implementations and data infrastructure, then verify promising complete recipes at the target scale. No current setting is declared optimal in advance.

This is a research and engineering plan, not a measured quality result. No GPU was started for this work. **The previously discussed 1.5B/25-layer/64K-tokenizer/30B-token recipe, including 2,097,152 tokens per update, is the measured baseline, not a requirement.** Candidate designs can change any of these. A reproducible experiment still needs a stated resource budget and a common quality measurement; those are comparison rules, not frozen model hyperparameters. The exact spend must be decided from real throughput and the desired quality/cost frontier.

## 1. Starting measurement: the current model

The existing measured configuration is [`configs/terminus_25l_2304d_64k_h100.json`](../configs/terminus_25l_2304d_64k_h100.json). Every property below is a candidate for change if a measurable gain justifies its engineering and retraining cost.

| Property | Current maintained Terminus |
| --- | --- |
| Parameters | 1,500,334,898 |
| Transformer | 25 layers; width 2304; SwiGLU FFN width 5760 |
| Attention | 18 query / 6 KV heads; head dimension 128; four local layers then one global layer |
| Sequence geometry | Context 4096; local window 1024 |
| Vocabulary | 65,536 model entries; BlinkDL/rwkv-7-world tokenizer; NFC preprocessing |
| Normalization | Pre-RMSNorm, parameter-free Q/K RMS normalization, scalar SubLN before output/down projections |
| Weights | FP32 latent weights; blockwise 128×128 absmean ternary forward weights; identity straight-through estimator |
| Training arithmetic | BF16 residuals; FP8 DeepGEMM forward/backward; FA3; fused projections and pointwise operations |
| Optimizer | Muon on logical projection matrices; auxiliary AdamW on embedding/readout and small parameters |
| Measured global batch | 2,097,152 tokens per optimizer update; one candidate among others |

At the measured context of 4096, that baseline batch is 512 sequences globally. For any candidate:

`tokens/update = microbatch × data-parallel world size × accumulation × sequence length`.

The measured single-H100 Muon case used microbatch 8, accumulation 64, without activation checkpointing. Its short synthetic benchmark was **46,395 tokens/s**, including the optimizer step. This excludes real data loading, validation, checkpoint writing and distributed communication. It is a planning reference, not sustained production throughput. [L1]

The current Muon LR 0.003, auxiliary AdamW LR 0.003 and norm LR 0.0002 are copied from the archived 150M policy. **They are starting points, not tuned values for this model.** Muon is an optimizer; μP is a parameterization and scaling prescription. Restoring Muon did not implement μP.

The baseline 30B tokens / 1.5B parameters gives about 20 tokens per parameter, roughly the classical dense compute-optimal scale but far shorter than many small models trained to minimize deployment cost. The best choice depends on whether training spend, inference spend, memory, latency, or absolute quality matters most. Search model size **together with** training duration and measure the Pareto frontier; a 1.5B model and 30B tokens need not be the selected endpoint.

## 2. What to take from existing work

| Source | Reuse | Adapt or verify for Terminus |
| --- | --- | --- |
| MiniCPM4 / ModelTunnelv2 [S1] | Select recipes using downstream conditional loss as well as pretraining loss; test small models before large ones | Its complete pipeline and exact training mixture are not publicly released in the sources inspected. Build the method around our own data and tasks. Its fitted task-loss-to-accuracy curves are not universal. |
| Public ScalingBench [S2] | Input/output examples for BBH, HumanEval, MATH, MBPP and MMLU | These are English benchmark-derived development examples with generated rationales. Add Indic evaluation and protect an independent final test. Load each of the five task JSONL files explicitly. |
| BitCPM4 [S1, S3] | Dense pretraining followed by ternary QAT; learning-rate re-warmup; compare stage allocation | It did not use ternary weights from the first token throughout its full training. Released model code is inference/fake-quantized checkpoint code, not a complete reproducible QAT trainer. It also used weight-only QAT, unlike our FP8 activation path. |
| Continual QAT [S4] | Compare dense/QAT splits; compare retained versus reset optimizer state | Its 1B OLMo/Dolma experiments favored approximately 20% dense / 80% QAT. This is evidence for a candidate, not a universal optimum. |
| ParetoQ [S5] | Quantizer reference implementation; learned-scale ternary candidate; independently scheduled dense and QAT stages | Its approximately 90% dense / 10% QAT optimum came from a different model, quantizer and training horizon. SEQ is not our existing absmean quantizer. Official code has a BSD-style license; preserve its notices. |
| Microsoft μP [S6] | Base/delta shape tooling, coordinate checks, tied-readout handling | MuAdam rules do not automatically apply to Muon. Our tied embedding/head, GQA, normalization and ternary STE require an explicit compatible scaling contract. |
| Moonlight Muon [S7] | Published update-RMS adjustment and weight-decay practice; distributed implementation if later necessary | Treat the current logical Q/K/V and gate/up partitions as a defined Muon baseline. Changing the partitions changes the optimizer and needs a quality comparison. |
| FineWeb2 / DataTrove [S9] | Published data filtering, deduplication and sharding infrastructure | Choose appropriate language subsets; measure quality and candidate-tokenizer token counts ourselves. Do not download the entire multilingual collection. |
| Existing evaluation frameworks [S12] | Task adapters, answer scoring and standard metrics | Add the Terminus model/tokenizer adapter, answer-only loss masks and multilingual task definitions. Avoid writing a new general evaluation framework. |

BitCPM4 reports that a QAT portion of about 40% or more worked well in its small-model comparison, with additional validation at 150M. Its final models received 350B QAT tokens. Its 5M-model learning rates are not useful numeric defaults for our 1.5B model. Continual QAT, BitCPM4 and ParetoQ offer **different experimental hypotheses**, not three interchangeable recipes.

The BitCPM4 pilot held **total training at 2B tokens**, allocated 20% of that total to decay, and found that QAT exceeding about 40% of the total approached ternary-from-scratch loss. The authors then **explicitly say** they build BitCPM4 with QAT tokens equal to twice the decay-phase tokens. Thus, if an FP8 dense model's decay is 100B, **200B additional QAT is a faithful application of their stated heuristic**. It is not a demonstrated optimum for Terminus: their pilot's 40% QAT and 20% decay were shares of one fixed 2B-token budget, while 800B already-finished dense + 200B QAT totals 1T with only 20% QAT. The paper reports a 350B-token QAT process for the released BitCPM4 models but does not provide a universal conversion-token law or a Terminus-specific guarantee. Compare 200B with earlier/later ternary checkpoints and, if training has not started, earlier-switch schedules at matched **total training cost**. Re-warm the QAT learning rate as a new training stage and judge final ternary task quality. The 40% pilot fraction cannot be imposed mechanically on an already-finished model; over 533B additional QAT would be needed to make that fraction of an 800B-plus-QAT total, but the paper does not prescribe that number for this case.

The 2026 width-and-horizon transfer work for Muon/MLA MoE models reinforces separating those two scaling questions. Its numerical law is not established for a dense GQA ternary model. [S8]

## 3. Establish our training data and quality target first

All proxies should train from scratch on representative samples of **our intended corpus**. ModelTunnel is not a special pretraining dataset, and ScalingBench is not the corpus to pretrain on. We cannot reproduce MiniCPM's private mixture merely by reproducing its HPO method.

### Starting corpus

1. Use the relevant **Sangraha verified** subsets as an initial Indic pool. The eleven languages are English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Odia and Punjabi. Validate samples despite the subset's name. [S10]
2. Supplement with selected **FineWeb2** language/script subsets after cross-source deduplication. Use original FineWeb/FineWeb-Edu for English. Filter language identification errors, bad extraction, boilerplate and excessive repetition. [S9]
3. Consider a bounded share of public educational/synthetic and code data, such as appropriate SmolLM-Corpus subsets, before paying to generate a new large teacher corpus. Dataset-reported token counts use other tokenizers; recalculate them with **each candidate tokenizer**. Code subsets retain source-specific license requirements. [S11]
4. Audit domain, date, document length, repetition and unique usable tokens per language. Split held-out documents **before** packing. Near-deduplicate across sources and remove evaluation overlaps, including translated or near-duplicate questions when identifiable.
5. First create a versioned **raw-text** corpus with document IDs, language labels and source provenance. Then pretokenize separately for each shortlisted tokenizer, with tokenizer/preprocessing hashes on the shards. A pretok corpus produced for one vocabulary cannot be reused unchanged with another. Maintain a disjoint raw-document reserve for final evaluation.

Do not invent a precise language mixture without this inventory. Set explicit minimum coverage and repetition limits for every language, then compare a few mixtures. A useful initial family is language sampling proportional to `available_tokens^α`, with α = 0.3, 0.7 or 1.0, subject to those limits. Reserve domain shares separately so that language balancing does not accidentally remove useful math/code or flood the run with duplicate low-resource data. Final proportions are selected using the quality objective below. The mixture is a search variable, including any late curriculum.

Use the same mixture within a narrow optimizer comparison so its result is interpretable, then compare mixtures and repeat a small LR/batch check for the leading ones. This temporary experimental control does not remove mixture from the overall search.

### Evaluation that decides the winner

- **Generalization:** start with roughly 0.5–1M baseline-token-equivalent held-out text per language; report per-language NLL and changes against a same-tokenizer baseline. Compare tokenizers on the **same raw documents** using bits per byte or another tokenizer-independent text measure. Token NLL and token counts from different vocabularies cannot be compared directly. Do not let the largest language dominate the average.
- **Downstream development:** answer-only conditional loss, multiple-choice answer margins and ordinary accuracy on knowledge, reasoning, reading and code tasks. Use public ScalingBench as one English component. Generated rationale loss is an auxiliary signal; verbosity or imitation of the rationale style is not the quality target.
- **Indic coverage:** use MILU development/validation splits for the eleven-language knowledge component, plus suitable reading and native-language tasks. MILU is public but currently requires accepting its gated dataset terms. Do not treat translated English MMLU alone as complete Indic evaluation. [S13]
- **Final test:** keep source-level disjoint tasks/documents out of HPO. If a public test set is used repeatedly for recipe selection, label it development data and obtain a separate final test. Report the standard public benchmark separately with the tuning caveat.

Predeclare a macro-average objective across languages and a small set of domain scores, with per-language regression limits. Do not pick the score weights after seeing which recipe wins. For close candidates, use paired item-level bootstrap intervals and at least two training seeds; small task subsets cannot resolve arbitrarily small gains.

Begin with 500–1,000 development examples per language where available, then increase only where uncertainty changes the choice. Cheap loss evaluation can run often; full generation evaluations run at milestones. Fit a task-loss-to-score relationship only after our own checkpoints show a useful relationship; do not fit elaborate curves to a handful of points.

## 4. Search the design, then build valid proxy families

The following is **one controlled width-transfer family**, anchored to the current architecture. It is a tool for checking a particular scaling claim, not a restriction on the broader model search:

| Role | Width | Query / KV heads | Head dim | FFN | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main search model | 768 | 6 / 2 | 128 | 1920 | 200,284,466 |
| Transfer check | 1536 | 12 / 4 | 128 | 3840 | 700,396,082 |
| Target | 2304 | 18 / 6 | 128 | 5760 | 1,500,334,898 |

All three happen to have 25 layers and 65,536 vocabulary entries. These dimensions satisfy the current 128-aligned native path. The earlier 768-wide idea with 12/4 heads would have changed head dimension to 64; use **6/2 heads** only within this matched-head-dimension family. Different depths, heads and tokenizer sizes are searched in separate families.

Search architecture and tokenization before assuming the existing 1.5B shape is optimal:

| Variable | Practical first comparisons | What decides |
| --- | --- | --- |
| Model scale and token horizon | Around 0.7B, 1.0B, 1.5B and 2B parameters; horizon hypotheses around 15B, 30B, 60B and longer only if learning curves justify the spend | Quality per total training GPU-hour, inference memory/latency, and absolute quality |
| Depth versus width | At matched parameter and training-compute budgets, compare shallower/wider and deeper/narrower designs around the current 25×2304 | Downstream quality and actual native throughput; recheck LR scaling |
| FFN and attention shape | FFN expansion, head dimension 64 versus 128 where supported, GQA ratio, tied versus untied head | Quality, KV memory, parameter count and token/s |
| Attention allocation | Current four local/one global against more or fewer global layers; local window and 2048/4096/longer context curricula | Long-context quality, ordinary task quality and wall cost |
| Tokenizer | Current RWKV 64K against at least a smaller and, if coverage warrants, a larger multilingual vocabulary, e.g. 32K/96K candidates | Fertility by language, byte-level compression, embedding/head cost, same-raw-text quality and inference latency |
| Normalization and initialization | Existing RMSNorm/SubLN/QK norm versus limited coherent alternatives | Stability and full model quality; measure native implementation cost |

The ranges are screening suggestions. Parameter counts and supported geometry must be computed for each candidate; the current DeepGEMM alignment and complete-cycle constraints may require engineering for a new shape. Do not treat an unsupported candidate as evidence that its design is poor. Most architecture alternatives should be compared on matched parameter or **measured compute** budgets before a few receive longer runs.

Tokenizers change what “30B tokens” means. Keep the raw documents identical for an early tokenizer comparison, report bytes or documents processed and training GPU-hours, and evaluate the same untouched raw text and tasks. Preprocess/retokenize separately; compare fertility for each of the eleven languages and inspect whether a small vocabulary fragments Indic text excessively.

A 5M model can test scheduler plumbing, but its embedding budget, attention geometry and quantization behavior would poorly represent this target. Even the 200M model has a larger embedding share than the target; this is a reason to validate transfer, not to assume it.

The μP/Muon work has a bounded acceptance gate:

1. Specify initialization, residual/readout scaling, parameter roles and LR/decay scaling across widths. Include the tied embedding/head and scalar norms. The Microsoft package supplies useful tooling and `MuSharedReadout`, but its MuAdam optimizer is not a replacement for Muon.
2. Run coordinate checks across widths: activation and residual magnitudes, Q/K logits, gradient norms, update-to-weight ratios, clipping, ternary occupancy, scale values and code changes. Preserve head dimension rather than blindly applying attention scaling intended for changing head dimension.
3. Compare a small LR bracket across widths under a clearly matched experimental design. Matching global batch is useful for an isolated width-transfer claim; **global batch itself is also searched** separately and its optimum may vary with width. Check whether the region of good LRs and recipe ranking actually transfers.
4. Test horizon extension separately. A 200M/4B-token run and a 1.5B/30B-token run have similar tokens per parameter but different update counts. This is not proof of zero-shot LR transfer.

Do not stack additional μP LR factors onto the existing Muon shape scaling without checking the resulting update. If a coherent Muon-compatible parameterization cannot be validated within the pilot budget, compare the current published RMS-adjusted Muon variant empirically and spend on a narrow target-size LR bracket. Call that empirical transfer, not proven μP.

**Width-scaling detail:** The pinned upstream Muon update first approximately orthogonalizes each logical projection gradient and multiplies its result for an `r × c` matrix by `sqrt(max(1, r/c))`. Terminus also sets that group's learning rate to `base_lr × 0.2 sqrt(c)`. An ideal partial-isometry update has elementwise RMS `1/sqrt(max(r,c))`, so these two factors yield an approximate per-element update RMS of `0.2 × base_lr`, independent of matrix size. This is a specific RMS-matching rule, not a proof of μP transfer or of correct function-space scaling. The tied token embedding/output matrix goes through auxiliary AdamW, not Muon; any proposed μP tied-readout or embedding rule must be worked out for that AdamW path and the model forward. An extra `1/d` or `1/sqrt(d)` factor could change the update substantially, but **under-updating or collapse at width 2304 is a hypothesis to test**, not an established consequence. Check per-role update/weight and activation/output scaling across widths.

**Concrete Muon/μP comparison:** The August 2026 Muon/μP paper [S8, §2.1 and Appendix A] combined the same `0.2 sqrt(max(fan_in, fan_out))` RMS adjustment with an additional `fan_in_base/fan_in` LR factor for hidden matrices, while its embedding/vector-like roles did not receive that LR factor. It also held head dimension and depth fixed in its width-transfer test. This makes a useful candidate for Terminus, although its dense experiments used MLA, ordinary dense weights and a different readout. Compare (A) current RMS-matched Muon and (B) the paper-inspired hidden-matrix factor, with the present logical Q/K/V and gate/up Muon partitions in both. An exponent halfway between the two factors can be a diagnostic candidate if the transfer curves call for it. Use the **base fan-in for each corresponding logical matrix role**, rather than one literal `1/d_model` factor on every parameter. Check current initialization before applying any μP initializer, and specify the tied embedding/readout separately.

Terminus currently sets each Muon group's `lr = base_lr × S` and `weight_decay = base_decay / S`, so the per-step decay product is `base_lr × base_decay`. If candidate B multiplies the group LR by `m = fan_in_base/fan_in`, compensate its group weight decay by `1/m` in the first controlled comparison; otherwise the comparison changes both update scaling and decay. A later independent weight-decay search can choose another policy. First run coordinate and update diagnostics at widths 768, 1536 and 2304, then compare a small base-LR bracket at a **common batch and horizon within this transfer experiment**. The overall batch search remains open. Promote the μP-inspired recipe only if stable activations, useful LR transfer, and final quality support it; otherwise use the measured width-specific LR brackets. Retest after changing global batch, depth, quantizer, or dense/QAT schedule.

## 5. Search the training recipe in stages

### First: optimizer stability and learning rates

Use coarse log-spaced screening, then narrow around promising regions. Suggested **search bounds**, not final settings:

| Parameter | Initial treatment |
| --- | --- |
| Muon LR | Bracket the archived 0.003 with approximately 0.001, 0.003, 0.009; parameterization may require redefining the base LR |
| Embedding/readout AdamW LR | Approximately 0.0003–0.003; search separately from Muon |
| Norm/scalar LR | Approximately 0.0001–0.0004; initially center on 0.0002 |
| Weight decay | Baseline 0.1; compare 0.03 and 0 for QAT after stabilizing LR |
| Muon momentum; AdamW betas | Current 0.95 and (0.9, 0.95) are baselines; bracket them after LR and batch |
| Gradient clipping | Include disabled and active clipping candidates; monitor clipping frequency and instability |
| Warmup and decay | Compare token- and update-aware warmup choices, cosine and warmup–stable–decay; tune length with horizon and batch |

A single shared decay setting currently controls optimizer groups, while norms have zero decay. Separate dense/QAT or embedding/matrix decay is an implementation change. Do not claim those knobs already exist independently.

Use an experiment ledger and a simple tool such as Optuna for bounded random/log sampling and promotion. A massive Cartesian grid is unnecessary. Separate the dense and QAT LR searches; the transition may require a new LR even when the optimizer is retained. Include AdamW and a suitable published Muon variant as optimizer controls where their memory and cost permit. Test optimizer details such as Newton–Schulz steps/precision only if they affect stability, quality or whole-update cost enough to matter.

Short screening runs need a deliberate short-run warmup. A 250M-token screening run cannot use the baseline 300M-token warmup and tell us about peak-LR behavior. Define warmup and promotion-compatible schedules before launching trials. At 2.1M tokens/update, 250M tokens is only about 119 updates; at 262K it is about 954; at 4.2M it is about 60. An equal-token short test can therefore give misleading rankings across batch sizes. Require adequate optimizer updates and later equal-compute/equal-data checks.

### Effective batch, microbatch, accumulation and checkpointing

**Global batch is a hyperparameter.** Sweep a coarse range around the current value, for example approximately **0.26M, 0.52M, 1.05M, 2.10M and 4.19M tokens/update**, conditional on stable LR, available memory and enough updates at the chosen horizon. Probe outside that range if the quality/throughput curve has not turned. At context 4096 on one GPU with microbatch 8, these correspond to accumulation 8, 16, 32, 64 and 128; those are examples, not mandatory microbatch settings.

For each global-batch candidate, retune at least a small Muon/auxiliary-LR bracket and warmup, then compare validation loss/tasks **after the same training tokens or measured compute**. Record optimizer updates, gradient norm/noise, clipping frequency, peak memory and tokens/s. Report both quality at a fixed cost and the highest throughput among configurations meeting the quality target. Changing accumulation alone is a legitimate global-batch experiment; keeping global batch unchanged while changing microbatch/accumulation isolates implementation efficiency.

Separately search microbatch, accumulation decomposition and activation checkpointing for **each promising global batch**, including recomputation granularity. This tests whether saved memory permits a faster microbatch or sequence length. The largest fitting microbatch may or may not maximize tokens/s. If data parallelism changes, include communication and optimizer-sharding cost in the same comparison. Never infer global-batch quality from a throughput-only sweep.

Sequence length and packing change the sequence count represented by a token batch. Compare context curricula and long-context extension on the same raw data and task objective. Revisit global batch and LR when sequence length changes.

### Second: dense-to-ternary allocation

Compare the following first at a common total raw-text and compute budget for a chosen architecture/tokenizer, with the current absmean quantizer as one reference:

| Candidate | Dense portion | Ternary portion | Reason |
| --- | ---: | ---: | --- |
| A | 0% | 100% | Required control: current native-ternary training |
| B | 20% | 80% | Continual-QAT-inspired candidate |
| C | 60% | 40% | BitCPM4-inspired candidate |
| D | 90% | 10% | ParetoQ-inspired boundary candidate; most vulnerable to insufficient QAT recovery on our short horizon |

If a 30B-token horizon using one tokenizer were selected, these would mean 0/30B, 6/24B, 18/12B and 27/3B respectively. That horizon and tokenizer are also search variables. The percentages are hypotheses; none is presently selected.

Start with an abrupt switch plus a short LR re-warmup and retained optimizer state. Compare reset state or a short smooth quantization ramp only if the transition diagnostics justify another branch. Continual QAT found that preserving optimizer state and smoothing the switch could reduce the immediate spike without necessarily improving the final result. Do not spend most of the budget on a cosmetically smoother curve.

Use one data mixture within the first transition comparison to identify its effect. Then compare curated annealing mixtures with the leading schedules, including a small LR/batch recheck. Data quality, quantization and LR can interact.

Compare both **equal-token quality** and **quality per measured GPU-hour**, because dense and ternary training may run at different rates. Evaluate final candidates using their intended ternary weights. Allow candidates to finish their planned QAT recovery before ranking them; do not prune a late-switch candidate based only on its pre-switch loss or immediate loss spike.

Checkpoint branching saves compute only while the histories are actually identical: parameters, data order, optimizer state and LR schedule. A shared flat-LR prefix permits useful branches. Independently decayed dense stages generally do not share that prefix. Do not count nonexistent checkpoint reuse in the budget.

### Quantizer, precision and loss choices

Include existing blockwise absmean + STE as the low-engineering baseline. Compare a **learned block-scale ternary** implementation informed by ParetoQ, alternative grouping/threshold/scale rules, and an explicit weight-only versus weight-plus-activation quantization comparison where deployment and training hardware justify it. The symmetric `{-scale, 0, +scale}` target and 128×128 grouping are **candidates**, not immutable constraints, unless a deployment requirement later makes them necessary.

Learned scales require a correct scale-gradient/surrogate implementation and native-path validation. Changing ParetoQ's grouping and gradient prescription means it is an adaptation, not an exact reproduction. Compare precision paths (current full FP8 backward, FP8 forward/dgrad with BF16 weight gradients, and BF16 reference) on **both quality and sustained cost**. Investigate loss implementation/masking and long-sequence memory strategies where they alter throughput or numerical results. Broader changes such as shifts, extra norms or teacher distillation can enter the search after their incremental cost is priced.

If the learned-scale candidate wins, repeat the shortlisted LR and transition comparisons with it before target transfer. The earlier absmean optimum is not automatically its optimum.

Terminus already has SubLN. Do not blindly add normalization layers because another conversion paper found them useful. Likewise, use expensive teacher distillation only after measuring a remaining capability gap and teacher cost. Existing public synthetic data is the first economical option.

## 6. Engineering required before this campaign is trustworthy

| Work | Existing support | Required addition / acceptance check |
| --- | --- | --- |
| Raw corpus and tokenizer comparison | RWKV pretokenization, manifest-based training | Versioned raw documents; candidate tokenizer training/selection and separate token shards; byte-based held-out metrics and fertility by language |
| Real data and evaluation | Per-language held-out loss | Representative prepared corpus; contamination checks; task adapter and answer masks; reproducible metric versions |
| Experiment management | JSON configuration and training logs | Fully resolved run configs; parent checkpoint and source/data hashes; search choices; GPU-hour accounting; failure/recovery classification |
| Batch/context sweep | Configurable data-manifest batch size and accumulation | Generate separate manifests/configs per candidate; verify global-batch formula, loss normalization, partial final updates, update counts and matched-cost comparisons |
| Architecture/precision sweep | Configurable model geometry and reference/native backends | Parameter and shape calculators; support or explicitly price candidates outside current 128-aligned native path; quality and throughput parity receipts |
| Width transfer | Muon with logical projection slices | Explicit scaling contract, coordinate diagnostics, LR-ranking tests across the proxy family |
| Phase training | Reference quantization-strength interpolation | Dense hold → transition → QAT phase scheduler; independent phase/global token counters; re-warmup; intentional optimizer-state policy |
| Native dense stage | DeepGEMM ternary path | A validated dense-FP8 operand path, or an explicitly measured BF16 dense stage; do not assume a flag already enables dense DeepGEMM |
| Phase checkpointing | Strict exact-contract resume | Explicit phase initialization with checked parameter/optimizer mappings, RNG and data cursor; preserve strict resume within a phase |
| Alternative quantizers | Fixed absmean scale | Learned-scale and alternative grouping/threshold candidates with correct gradients, state, export and native implementation before relying on their rankings |
| Final acceptance | Short synthetic timing receipts | Real-data sustained run including optimizer, validation/checkpoint overhead; numerical and resume checks; native quality measurements |

The production trainer currently rejects partial quantization; the DeepGEMM configuration requires quantization strength 1. Its existing warmup ramp begins at token zero, and its LR scheduler is cosine. Therefore **dense → ternary with re-warmup is not currently a config-only experiment**. [L2]

Use a BF16/reference implementation to establish mathematical correctness, then validate on the intended native path before trusting quality rankings. A BF16 weight-only QAT experiment does not by itself validate full FP8 forward/backward training. The final numerical policy, including whether weight gradients remain FP8, is a quality-relevant choice.

The current checkpoint loader intentionally rejects changes to the recorded run contract. Do not weaken that check to make a phase transition work. Add a separate explicit transition operation, map compatible state, preserve the global data cursor, set the new quantization-strength buffer deliberately, and record the parent checkpoint and changed fields. Avoid accidentally reapplying initialization/scaling when loading weights. [L3]

Meaningful correctness checks include gradient/update parity for the phase and quantizer changes, non-repeated data across a transition, optimizer-state mapping for fused logical slices, loss-mask correctness and exact interrupted/resumed training within the documented tolerance. These are prerequisites for interpreting HPO results.

## 7. A staged campaign and its cost

There is no honest way to exhaustively run the Cartesian product of all the choices above. **Every major choice is eligible**, but test them in a sequence that spends more only when evidence justifies it. Keep the entire quality/cost Pareto frontier, rather than forcing one parameter count or batch from the start.

| Stage | Search and comparison | Promotion rule |
| --- | --- | --- |
| 0. CPU/data study | Candidate vocabularies, fertility and byte compression; unique documents, language mix, leakage | Remove clearly poor data/tokenizer options; keep uncertain ones |
| 1. Native system study | Geometry, context, global batch decomposition, checkpointing, precision/backends, optimizer cost | Reject unsupported designs; remove slower variants only when they are numerically equivalent. Retain quality-relevant precision choices for training tests. |
| 2. Broad quality search | Small/medium designs; batch × LR × warmup; data mix; optimizer and ternary controls | Train enough **updates** for meaningful signals; promote by multilingual quality at matched cost, with uncertainty |
| 3. Interaction checks | Leading architecture × tokenizer × batch/LR; dense/QAT split × quantizer/precision; model size × token horizon | Extend finalists; rerun LR/batch when another selected choice changes its optimum |
| 4. Target confirmation | Several distinct complete recipes near the quality/cost frontier, with longer horizons and more than one seed for close choices | Choose deployment/training tradeoff using final held-out and task results |

Start by measuring actual sustained throughput on representative pretokenized data for a sample of sizes, batches, context lengths and precision modes. Include data loading, optimizer, validation/checkpoint overhead and distributed communication where relevant. Price proposed trials in **H100-hours or dollars**, rather than using parameter count × tokens alone.

The previous synthetic baseline gives a reference only: 30B tokens ÷ 46,395 tokens/s ≈ **179.6 active H100 hours** for that specific 1.5B/2.1M-batch native run. A different tokenizer changes tokens per document; a different global batch changes optimizer steps; different architecture/precision and dense/QAT proportions change token/s. The earlier **27–36 H100-hour pilot estimate cannot cover a credible search over all these axes** and is withdrawn as a compulsory ceiling.

For a candidate with multiple training phases, estimate cost from each measured rate and include overhead:

`GPU-hours = Σ(phase_tokens / measured_phase_tokens_per_second × GPU_count / 3600) + measured GPU-hour overhead`.

Compare training on matched **raw documents or bytes**, matched training GPU-hours, and the selected downstream quality objective. Report all three because equal token counts are invalid across tokenizers and equal updates are invalid across batches. Also report inference cost when choosing model size or quantization. Reserve a meaningful portion of any funded search for target confirmation, repeated close contenders, numerical correctness and phase-transition tests. Set an explicit spend limit **for each stage** after pricing it, then decide whether the next stage's expected information is worth its cost. This is cost control, not a frozen hyperparameter.

At the baseline batch, 500M tokens is only about 238 optimizer updates and its baseline 300M-token warmup takes about 143 updates. At another batch these counts change. A short run can reject numerical failures and gross LR mistakes, but cannot establish the long-horizon optimum. Evaluate learning curves at multiple horizons, and do not resume a fully decayed short trial as though it had been scheduled for a longer one.

For each serious dense/QAT candidate, test conversion of an **actually trained checkpoint** at its planned switch point. A randomly initialized target-size smoke test cannot validate that transition. Keep rollback and recovery capacity until post-switch training is accepted.

## 8. Decision gates and what we produce

1. **Data gate:** all eleven languages have documented usable token inventories, fixed splits and interpretable held-out/task metrics. Output: a versioned corpus manifest and evaluation manifest.
2. **Numerical gate:** the optimizer/scaling/quantizer combination behaves correctly across widths and on the intended native backend. Output: a parameterization specification, correctness receipts and runtime pins.
3. **Design and proxy gate:** retain the nondominated architecture/tokenizer/batch/optimizer/data/phase recipes with acceptable per-language tradeoffs and reasonable uncertainty. Output: complete trial ledger, actual GPU-hours, learning curves, transition recovery and quality/cost comparisons.
4. **Transfer gate:** check each finalist's LR/batch region and recipe on a larger size/horizon. Output: an explicit statement of which transfer claims are measured versus extrapolated.
5. **Production gate:** real-data throughput, checkpoint/resume, and any planned phase transition work for the chosen recipe. Output: a recorded training manifest with checkpoints and stop/recovery criteria.
6. **Final quality gate:** evaluate the selected final ternary model on untouched tasks; test the exported numerical representation if deployment uses packed ternary weights. Output: reproducible model card, per-language results and actual training cost.

Use stage-specific schedules deliberately. Do not resume a fully decayed short trial as if it had always been planned for a longer horizon. WSD can support useful shared prefixes and continuation, but it does not guarantee that longer-training hyperparameters remain optimal.

If the product is an assistant, compare multilingual instruction-tuning recipes on selected base-model candidates and account for that cost. Evaluate forgetting and language balance. Start from curated public supervised data before funding expensive RL or teacher distillation. Base pretraining quality and instruction-following quality require different evaluations.

## 9. What is searchable, and what an individual run must record

**No model or training hyperparameter is permanently held constant by this strategy.** The 64K tokenizer, 25 layers, 4:1 attention pattern, Muon, FP8 backward, 30B-token horizon and **2,097,152-token global batch** can all be challenged. The search can also change the corpus, objective, normalization, quantization, checkpointing and hardware plan. Some changes require code and new data shards; their engineering cost belongs on the same quality/cost ledger.

Three different meanings of “hold constant” matter:

| Situation | What is held constant, and why |
| --- | --- |
| Narrow comparison | Keep unrelated variables equal *within that experiment* to learn whether the changed choice helped. In later experiments change those other variables and test interactions. |
| Fair model selection | Use the same protected raw-document evaluation set, task definitions and scoring rules across candidates. Tokenizer-specific token NLL is not a cross-tokenizer score; use bits/byte and downstream tasks. Resource budgets can be equalized by measured GPU-hours, bytes or other explicitly stated axes. |
| Exact resume of one run | Record tokenizer map, model shapes, optimizer/quantizer policy, precision, global batch, data cursor, source/runtime and phase schedule. Resume under the same recorded contract, or perform an explicit validated migration/new run. This is reproducibility, not a ban on testing another recipe. |

Global batch is selected experimentally. For a specific target global batch, changing microbatch and accumulation together can isolate implementation efficiency. Changing their product changes the optimizer's statistical batch and needs its own LR/warmup and quality study. For example, on one GPU at 4096 context, 8×32 gives **1,048,576** tokens/update, 8×64 gives **2,097,152**, and 8×128 gives **4,194,304**. If microbatch 8 is slow or does not fit after another change, test other microbatches and gradient checkpointing at each global batch.

Activation checkpointing, data prefetch, kernel fusion, precision, optimizer arithmetic and GPU count are also testable. Some are numerically equivalent implementation alternatives; some can change learning and therefore need full quality comparisons. The current checkpoint contract rejects arbitrary source/runtime/config/world-size changes on exact resume. Add intentional transition/migration support when the winning schedule requires a change mid-run. A new tokenizer changes IDs, the embedding/readout and the training-token stream; that candidate generally starts a new model run rather than resuming old weights. Test changed FP8 rounding or Newton–Schulz coefficients as a distinct numerical policy.

For fixed-size accumulation, select a compatible integer number of sequences per optimizer update, or implement correctly weighted partial updates. Never round accumulation silently and claim the global batch is unchanged. A chosen final run needs a fully resolved configuration and pinned artifacts so its result can be reproduced; that does not make the choice permanent for future experiments.

## 10. Priority relative to our earlier kernel discussion

1. Prepare raw data, candidate tokenizers and comparable downstream evaluation.
2. Measure design, batch, precision and memory choices on real data; build the missing phase/architecture support.
3. Search optimizer, batch, model size/shape, data mixture, token horizon, quantization and dense/QAT allocation in stages.
4. Confirm quality and sustained throughput for complete finalists at larger scale.
5. Optimize the remaining measured bottleneck for the selected quality/cost tradeoff.

DeepGEMM already supplies weight-gradient functionality in the inspected environment; do not budget a replacement before profiling the pinned implementation. Recheck the exact available commit/API if upgrading. TE's operation fuser and CODA's GEMM-epilogue ideas are worthwhile references for later fusion work, but a full library migration is not part of selecting this recipe. [S14]

Gram Newton–Schulz is a measured cost/quality candidate, not a substitute for tuning Muon. At the **2.1M-token baseline batch only**, our separate diagnostic optimizer phase was about 271 ms against a roughly 45-second global update, around 0.6%. Halving the entire optimizer phase would improve that iteration by at most about 0.3%; halving just Newton–Schulz saves less. At a much smaller global batch there are more optimizer steps per token, so reprofile the optimizer fraction before assigning it low priority. Preserve optimizer mathematics or treat changed coefficients/precision as a new numerical policy. [L1, S15]

The most useful first deliverable is **a reproducible, staged design search on our data**, with trustworthy arithmetic, a quality/cost frontier and a real budget ledger. A copied model architecture or a faster isolated kernel is not evidence that we have found the best complete recipe.

## Sources and inspected local evidence

### Research and reusable components

- **[S1]** [MiniCPM4: Ultra-Efficient LLMs on End Devices, sections 2.3 and 3.3](https://arxiv.org/html/2506.07900v2). ModelTunnelv2 and BitCPM4 training experiments. Public release claims do not include a complete ModelTunnelv2 trainer or exact proxy corpus.
- **[S2]** [openbmb/DensingLaw-ScalingBench](https://huggingface.co/datasets/openbmb/DensingLaw-ScalingBench), inspected revision `c730220daf6dbd4885ea254bcaca0e1f86384098`; Apache-2.0 metadata. Five task directories; avoid assuming the default dataset-server view includes every task.
- **[S3]** [BitCPM4-1B model and files](https://huggingface.co/openbmb/BitCPM4-1B), inspected revision `70d989c4972efa636e8a2211975515d188e000b0`; Apache-2.0. Card identifies MiniCPM3-1B as its source model. Its attention geometry and vocabulary differ from Terminus.
- **[S4]** [Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit?](https://arxiv.org/html/2502.11895). Dense/QAT stage allocation and transition ablations.
- **[S5]** [ParetoQ paper](https://arxiv.org/html/2502.02631) and [official implementation](https://github.com/facebookresearch/ParetoQ), inspected repository head `7b36b8de958aada6508e620c8dc544e6ed6c39b4`; [BSD-style license text](https://github.com/facebookresearch/ParetoQ/blob/main/LICENSE).
- **[S6]** [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466) and [Microsoft μP implementation](https://github.com/microsoft/mup), inspected repository head `19814971934ef91dd546f88e913fc963e096d11c`; MIT. The README documents tied readouts, coordinate checks and checkpoint scaling caveats.
- **[S7]** [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) and [Moonlight implementation](https://github.com/MoonshotAI/Moonlight), inspected repository head `c2ad5b20c605086526a179d36901bfc41b52b44b`; MIT.
- **[S8]** [Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts](https://arxiv.org/abs/2608.20061). Supporting evidence for treating width and horizon separately; not direct validation of our dense ternary model.
- **[S9]** [FineWeb2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [released filtering pipeline](https://github.com/huggingface/fineweb-2/blob/main/fineweb-2-pipeline.py), [DataTrove](https://github.com/huggingface/datatrove), and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). FineWeb2 metadata uses ODC-By; DataTrove is Apache-2.0. Retain source provenance and applicable dataset notices.
- **[S10]** [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha), verified/unverified/synthetic subsets; CC-BY-4.0 metadata.
- **[S11]** [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). FineWeb-Edu dedup, Cosmopedia-v2 and Python-Edu subsets; dataset token counts are not RWKV counts.
- **[S12]** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [OpenCompass](https://github.com/open-compass/opencompass). Reuse suitable evaluation adapters rather than a new general task framework.
- **[S13]** [MILU](https://huggingface.co/datasets/ai4bharat/MILU), eleven-language knowledge evaluation; CC-BY-4.0 metadata and gated access at inspection.
- **[S14]** [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/abs/2605.19269). Kernel abstraction relevant to the later performance work, not evidence about our optimal training recipe.
- **[S15]** [Gram Newton–Schulz](https://github.com/Dao-AILab/gram-newton-schulz). Evaluate equivalent math and whole-update savings before adopting.

### Local evidence

- **[L1]** [Maintained 1.5B Muon benchmark and archive-policy receipt](evidence/muon_25l_20260927/README.md), [selected configuration](../configs/terminus_25l_2304d_64k_h100.json), [optimizer implementation](../terminus/optim.py), and [logical Muon implementation](../terminus/logical_muon.py). Pinned upstream Muon commit: `f98f1cacc0263b04290753e32be8d498c1efc806`.
- **[L2]** [Model constraints and reference quantization](../terminus/model.py), [training schedule and production checks](../terminus/train.py), [native quantization](../terminus/quantization.py). These establish why a dense/transition phase requires new engineering.
- **[L3]** [Checkpoint loader](../terminus/checkpoint.py): exact source/config/runtime/data contract is enforced for resume.

Research checked against public sources available on the report date. Repository-head hashes describe inspected sources, not a request to upgrade production dependencies. Numeric search ranges and run allocations in this document are proposals; no quality gain has yet been measured for them.
