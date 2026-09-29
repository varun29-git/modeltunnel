# Tokenizer 65,536 freeze

Licensed by Sarvam AI under the Sarvam AI Research License. Indic cuts come from BPE scores inside `tokenizer-65k.json`. The `sarvam_id` list is gone. Our id is the index in `pieces`. The space mark is Ġ (U+0120). This is not an official Sarvam product.

- sha256 of the original Sarvam `tokenizer.model`: `4cd33409a577e8b416247587b0f5bd7a3eec245a1f18d4ec7793ff299ad3fbe2`
- sha256 of `tokenizer-65k.json`: `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`
- sha256 of `frozen_indic_ids.json`: `d0eabf4c1458b8ac505bb62aa3ca32992a0638e41f6ee936ef583d24caf27ccd`
- K = 40059. U = 435. English tail = 25189. That is 65,536 − 32 − 256 − 40059.

Indic cuts come from rules inside the json. The model type is BPE. Each kept piece has its score in `indic_scores`, beside `pieces`. A higher score merges first. Before that cut, the span is put in Unicode NFC, a leading Ġ is added, and each ordinary space becomes Ġ. `sarvam_id` is gone. The space mark is Ġ. Encoding and decoding an Indic span reads this file and does not open `tokenizer.model`.

`tokenizer_spec` states those rules. The encoder refuses to load if that block disagrees with the rules it implements. The router rule names protected whitespace, and it says how a mixed word attaches digits, punctuation, and other symbols. The vague digits line is gone. Twelve spare space ids keep their places and their empty scores, and their pieces are now `<reserved_space_39992>`, `<reserved_space_545>`, `<reserved_space_551>`, `<reserved_space_654>`, `<reserved_space_1137>`, `<reserved_space_1370>`, `<reserved_space_1697>`, `<reserved_space_2777>`, `<reserved_space_4233>`, `<reserved_space_4542>`, `<reserved_space_7859>`, and `<reserved_space_9042>`. The scored Ġ piece stays the real space piece. Every piece string is unique.

The earlier status recorded indic-id sha256 `555661c77c0ba23b8441726cb9ae15bf1b5d79e8a3c1cee74393e161165c3e2d`. That file was not on disk. The ids were assigned again from `tokenizer.model` with the same keep rules. The loaded file still gives K = 40059, U = 435, English tail = 25189, and `नमस्ते` = `Ġनम` + `स्ते`. U+200D stays. Ids 0 through 65535 were not moved. The new sha is the bytes of this file.

The vocabulary file did not change after the mixed-line space fix. The router joins Indic words that are separated by one ordinary space and sends that run to Sarvam as one span. A space next to a Latin word stays a real space token. A newline, a tab, or two spaces in a row stay on the English path. `main जा रहा hoon` is English `main`, a real space, Indic `जा रहा`, a real space, English `hoon`.

## Gate 1

Piece strings match original Sarvam on every round-tripping sentence, ignoring a leading `<s>` and a trailing `</s>` on Sarvam’s side only.

- Hindi: 100 sentences
- Marathi: 100 sentences
- Kannada: 100 sentences
- Telugu: 100 sentences
- Tamil: 100 sentences
- Gujarati: 100 sentences
- Bengali: 100 sentences

- `नमस्ते` → Ġनम + स्ते
- `हिन्दी` → Ġह + िन् + दी
- `ಕನ್ನಡ` → Ġಕ + ನ್ನ + ಡ

## Gate 2

- '\n': '\n'
- '\n  ': '\n  '
- '\n    ': '\n    '
- '\n\t': '\n\t'
- newline-four-spaces-return: '\n    ' | 'return'
- Each of these is one token: `\frac`, `\partial`, `\circ`, `\overrightarrow`, `\subseteq`, `\setminus`, `\text`, `\pi`, `\cdot`, `\sin`, `\cos`, `\int`, `\times`, `\left`, `\right`, `\quad`, `\sqrt`, `\alpha`, `\equiv`, `\theta`, `\varphi`, `\beta`, `\begin`, `\end`, `\binom`, `\Gamma`, `\to`, `\pmod`, `\sigma`, `\mathbb`, `\epsilon`, `\Delta`, `\geq`, `\sum`, `\lim`, `\ln`, `\infty`, `\pm`, `\tan`, `\Rightarrow`, `\omega`, `\approx`, `\mod`

## Gate 3

Tokens per word on the held-out files.

- en: ours 1.4254 (117091 tokens / 82147 words), original Sarvam 1.5328 (125916 tokens / 82147 words)
- code: ours 2.0299 (136908 tokens / 67446 words), original Sarvam 2.3825 (160689 tokens / 67446 words)
- math: ours 1.8574 (157362 tokens / 84722 words), original Sarvam 1.9336 (163821 tokens / 84722 words)

## Gate 4

- Broken UTF-8 tokens: 0
- Frozen Sarvam pieces that start with a dependent vowel mark: 2702

## Gate 5

- Round-trip mismatches: 0

## Gate 7

- Cross-script pieces: 0

## Training text

- CodeAlchemy is not in the bucket
- A dataset named Nemotron Code is not in the bucket. Code share uses Nemotron-Pretraining-Code-Concepts and Nemotron-Pretraining-Scientific-Coding, half and half.
- Finemath is not a separate corpus in the bucket
- SwallowMath is not in the bucket
- Nemotron CC HQ is in the bucket and was not used. ClimbMix is present, so it takes the English share.
- en: train 2430216382 UTF-8 bytes, 398536878 words, 1060557 documents; eval 500321 bytes
- code: train 1078357443 UTF-8 bytes, 144922690 words, 387675 documents; eval 501212 bytes
- math: train 491427650 UTF-8 bytes, 84538425 words, 145476 documents; eval 500547 bytes
