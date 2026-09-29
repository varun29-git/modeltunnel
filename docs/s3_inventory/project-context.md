# Training constraints

- Model size: 1.2 billion parameters. From-scratch pretraining unless a checkpoint already exists.
- Data is already cleaned.
- Scope: English plus Hindi, Marathi, Kannada, Telugu, Tamil, Gujarati, and Bengali, mixed in the same run. Kannada and Telugu are separate scripts. Hindi and Marathi share Devanagari.
- Tokenizer: train 65,536 first. Train 81,920 only if that fertility misses. Plan: [65k tokenizer](/cursor/stores/bc-3ba733cf-aea0-4cb0-a71b-053ddf16e5ee/docs/tokenizer-plan.md). Plain version: [Tokenizer explained](/cursor/stores/bc-3ba733cf-aea0-4cb0-a71b-053ddf16e5ee/docs/tokenizer-explained.md). Existing TinyLlama token packs must be retokenized after the chosen tokenizer is frozen.
- English token allocations are the sheet rows that sum to 710B. A cropped 45B on that sheet has no source yet.
- Indic sources: Sangraha, IndicCorpV2, translated Cosmopedia, translated Nemotron-CC HQ synthetic, translated Nemotron Fact-Seeking.
- Working plan: [English-Indic curriculum](/cursor/stores/bc-3ba733cf-aea0-4cb0-a71b-053ddf16e5ee/docs/curriculum-plan.md). Default Indic share is an assumption of 15%.
- S3 inventory: [Training curriculum](/cursor/stores/bc-3ba733cf-aea0-4cb0-a71b-053ddf16e5ee/docs/training-curriculum.md). Account `023894314260`.
- Run Varun described: about 1 trillion tokens, 750B English and 250B for the other languages together. The written curriculum is still the smaller draft until that replaces it.
- Still open: confirm that 1T mix against the sheet, and name the missing 45B. The 65,536 tokenizer is not frozen yet.
