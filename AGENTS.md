# Working rules for this repository

This repository is the standalone Terminus ModelTunnelv2 implementation. Read `AGENT_PROMPT.md` before changing code. Work here, not in the older `terminus-flagship-150m-30btokens` repository. `terminus_next/` is outside this project's scope.

## Commit progression

- Make **multiple small, meaningful commits** as the implementation advances. Do not wait until the end and create one large commit, and do not squash the milestones into one commit.
- Suggested boundaries: tokenizer integration; S3/data manifests; dense FP8 numerical path; dense→QAT transition; task evaluation/loss; experiment orchestration; end-to-end validation and documentation. Adjust boundaries when dependencies require it, but keep each commit independently reviewable.
- Run the tests relevant to each change before committing. Put the validation result and any remaining limitation in the commit message body or accompanying receipt. Use descriptive commit subjects.
- Push commits to the GitHub branch as milestones complete. Preserve history; do not force-push or rewrite published commits. At the end, report commit hashes in order and any uncommitted work.
- Never commit AWS credentials, full corpora, fetched evaluation records, token shards, checkpoints, generated run outputs, or cache files. Commit manifests, code, tests, and small non-sensitive receipts instead.

## Research boundary

Implement the complete framework and a bounded end-to-end smoke test. The owner will start the expensive search after reviewing readiness. Keep old Terminus behaviors covered by tests when making the eight-language campaign configurable. Do not represent synthetic throughput or small smoke results as an optimized full training recipe.
