# Historical training constraints

This dated planning note describes an abandoned 1.2B draft, not the implemented 146.4M experiments or a selected recipe. Private controller links and account identifiers have been removed; available local documents are linked relatively.

- Model size assumption: 1.2B parameters, from-scratch pretraining unless a checkpoint already existed.
- Scope: English plus Hindi, Marathi, Kannada, Telugu, Tamil, Gujarati, and Bengali, mixed in the same run.
- Tokenizer was not yet frozen. Existing TinyLlama token packs required retokenization after selection; the later freeze is recorded in `tokenizer/tokenizer-65k-freeze.md`.
- English allocations were planning-sheet rows summing to 710B. A cropped 45B row had no identified source.
- Indic source leads: Sangraha, IndicCorpV2, translated Cosmopedia, translated Nemotron-CC HQ synthetic, and translated Nemotron Fact-Seeking.
- Working draft: [English-Indic curriculum](curriculum-plan.md). Its default 15% Indic share was an assumption, not a measured optimum.
- Dated inventory: [Training curriculum](training-curriculum.md), with account identifiers redacted.
- A separate approximately 1T-token idea remained unconfirmed and did not replace the written draft.
