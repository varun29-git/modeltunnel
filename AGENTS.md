# Working rules for this repository

This is the public research-release tree for Terminus ModelTunnelv2. Read `AGENT_PROMPT.md` before changing model, data, checkpoint, or evaluation behavior. Keep implementation work in this repository; historical source repositories are provenance, not working-tree dependencies.

## Development and verification

- Preserve dense-weight FP8 and ternary QAT as distinct modes. Keep legacy tokenizer behavior covered where it remains supported.
- Run the relevant tests for each change and the CI-equivalent CPU suite before release: compile `terminus` and `tests`, run `python -m unittest discover -s tests -t . -v`, run the count-only preflight, and syntax-check `scripts/train.sh` as shown in `.github/workflows/ci.yml`.
- Do not weaken tests or treat CUDA skips as native-GPU validation. Record exactly which hardware and code paths ran.
- Keep changes reviewable and preserve existing history. Report remaining uncommitted work and validation limitations accurately.

## Data, artifacts, and public disclosure

- Never commit credentials, private endpoints/account identifiers, full corpora, fetched evaluation records, token shards, checkpoints, generated run outputs, or cache files. Commit code, tests, configurations, manifests suitable for disclosure, and small non-sensitive result receipts instead.
- Clearly distinguish committed analysis artifacts from external raw data, checkpoints, and logs. Do not claim reproducibility of a run from source alone when external inputs are required.
- Do not bundle third-party paper PDFs or relicense external tokenizer/data/evaluation assets. Link canonical sources and preserve applicable notices.

## Research claims

The completed 10B-token dense run and 2B-token QAT continuation are single-recipe regression evidence, not an optimized recipe. No matched dense continuation or downstream QAT suite has been run. Do not represent synthetic throughput, tiny smoke results, held-out token NLL, or one seed as downstream quality, causal attribution, or validated scale transfer. Expensive searches and cloud/GPU jobs require a separate explicit execution decision.
