# Terminus ModelTunnelv2

This is a **standalone implementation repository**, created from the maintained Terminus working tree on 29 September 2026. It currently contains the Terminus baseline, a frozen tokenizer artifact, research context, and a build brief. It does **not** yet contain the complete ModelTunnelv2-style pipeline or an optimized training recipe. No pretraining corpus, evaluation examples, S3 credentials, model checkpoint, or completed search result is bundled.

## Start here

1. Give the coding agent [`AGENT_PROMPT.md`](AGENT_PROMPT.md) and this repository URL.
2. Use [`DATA_SOURCE_MAP.md`](DATA_SOURCE_MAP.md) for the eight-language S3 scope and data inventory caveats.
3. Use [`RESEARCH_LIBRARY.md`](RESEARCH_LIBRARY.md) for the papers and public task/software sources.
4. Follow [`AGENTS.md`](AGENTS.md) for incremental, milestone-based Git commits.

## Contents

| Path | Purpose |
| --- | --- |
| `terminus/` | Maintained Terminus Python package, copied into the root so the coding agent can develop it directly. `terminus_next/` is intentionally excluded. |
| `configs/`, `tests/`, `scripts/` | Existing configurations, correctness tests, and launch/benchmark scripts. No ModelTunnel implementation is implied. |
| `data/tokenizer/` | Legacy tokenizer assets needed to understand and test the existing code. They are **not** the tokenizer chosen for this campaign. |
| `tokenizer/tokenizer-65k.json` | User-selected frozen 65,536-ID vocabulary and algorithm specification. SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. |
| `tokenizer/tokenizer-65k-freeze.md` | Provenance, usage notes, and prior checks for that artifact. It is not executable encoder code. |
| `docs/terminus/` | Existing Terminus architecture, FP8, and ModelTunnel research notes. The protocol is a plan, not a completed pipeline. |
| `docs/s3_inventory/` | Prior S3 inventory and curriculum documents, copied as **historical evidence**. Their 1.2B geometry, token totals, and suggested mixtures are not the selected Terminus recipe. Re-inventory before use. |
| `papers/` | Nine verified research-paper PDFs. No training or evaluation datasets are inside. |
| `evidence/muon_25l_readme.md` | Summary of the current 1.5B synthetic Muon/AdamW benchmark. |

## Scope and known mismatch

The S3 and tokenizer work supplied for this handoff targets **eight languages total**: English (`en`) plus Hindi (`hi`), Marathi (`mr`), Kannada (`kn`), Telugu (`te`), Tamil (`ta`), Gujarati (`gu`), and Bengali (`bn`), along with math and code domains. The maintained repository still hardcodes an older eleven-language set. The coding agent must make the campaign language set explicit and configurable without silently rewriting the old experiment's assumptions.

The current DeepGEMM route does FP8 GEMMs **with ternary weights**. It is not the dense-weight FP8 first stage used in MiniCPM4. The selected JSON tokenizer is also not loadable by `terminus/tokenizer.py` as it stands. These and the phase transition, task scorer, data pipeline, and search ledger are implementation work for the agent.

## Development boundary

The agent implements and tests in this repository and pushes multiple meaningful commits as milestones finish. The prompt stops after a bounded end-to-end smoke test; the owner will later decide when to run the expensive research campaign. The older original Terminus repository is a provenance source, not the agent's working tree.
