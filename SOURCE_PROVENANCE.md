# Baseline provenance

This standalone repository was initialized on 29 September 2026 from the maintained Terminus working tree at `/Users/varundaiya/terminus-flagship-150m-30btokens`.

- Original repository: `https://github.com/varun29-git/terminus-flagship-30b`
- Original HEAD when copied: `fb4a966867656ada78dc276b25bae1fa5e0e392c`
- The snapshot includes **uncommitted working-tree changes and untracked current configs** from that repository. It is therefore not identical to the original HEAD. The first commit here is the reproducible baseline for future work.
- Copied into this repository root: maintained `terminus/`, its configs, tests, relevant scripts, legacy tokenizer test assets, `pyproject.toml`, and selected research/evidence notes.
- Excluded: `terminus_next/`, archived runs, large training data, model checkpoints, generated benchmarks, credentials, and copied scripts/tests that depended solely on Terminus Next or omitted archive/paper-figure assets.
- User-selected tokenizer artifact: `tokenizer/tokenizer-65k.json`, SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. It is not yet integrated into the code at initialization.

The original repository remains separate. Do not assume later edits there have been copied here; make intentional, reviewable ports when needed.
