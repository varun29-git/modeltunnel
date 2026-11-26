# Source provenance and license scope

## Original implementation

This standalone repository was initialized on 29 September 2026 from a working-tree snapshot of the maintained Terminus project.

- Upstream source repository: <https://github.com/varun29-git/terminus-flagship-30b>
- Upstream HEAD at the snapshot: `fb4a966867656ada78dc276b25bae1fa5e0e392c`
- The snapshot also included uncommitted changes and untracked configurations, so it was not identical to that commit. Subsequent history in this repository contains the ModelTunnelv2 implementation, experiments, tests, and analyses.
- Included from the snapshot: the maintained `terminus/` package, configurations, tests, relevant scripts, legacy tokenizer test assets, project metadata, and selected research notes.
- Excluded: unrelated experimental trees, archived runs, large training data, checkpoints, generated run output, credentials, and unavailable figure dependencies.

Original Terminus source code, documentation, report generators, research reports, and subsequent original modifications in this repository are licensed under Apache-2.0; see `LICENSE` and `NOTICE`. This statement does not relicense the upstream repository or any third-party asset. The upstream repository remains a separate provenance source, not a runtime dependency.

## Third-party and external assets

- `tokenizer/tokenizer-65k.json` is a user-selected derived tokenizer artifact with SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. Its freeze note identifies the Sarvam AI Research License and modifications. It is integrated by `terminus/tokenizer65k.py` but is not an official Sarvam product and is not Apache-licensed by this release. The required license text is in `tokenizer/SARVAM_AI_RESEARCH_LICENSE.md`; its non-commercial research restrictions and explicit-permission requirements apply.
- `data/tokenizer/rwkv-7-world/rwkv_vocab_v20230424.txt` comes from the RWKV-LM source tree at commit `aeda8a06fc4803655473c7e7359b1d7166e2c604`, with SHA-256 `e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89`. It is retained for legacy compatibility under Apache-2.0; the Apache license text is included in `LICENSE`.
- Research papers are cited by canonical links in `RESEARCH_LIBRARY.md`; third-party paper PDFs are not distributed here.
- Raw corpora, token shards, evaluation records, checkpoints, full logs, and private cloud run mirrors are external. Their data/model terms and access controls are independent of this repository's code license. Dataset-license strings in manifests are provenance labels requiring review, not a grant of rights.

The optional Muon dependency is pinned to upstream commit `f98f1cacc0263b04290753e32be8d498c1efc806`; DeepGEMM, FlashAttention, Triton, PyTorch, and other installed dependencies likewise remain under their own licenses. No external dependency source is vendored or relicensed by this release.
