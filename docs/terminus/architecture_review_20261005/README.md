# Terminus architecture research package

- [Detailed architecture and tuning proposal](ARCHITECTURE_AND_TUNING_PROPOSAL.md): every model component, rationale, alternatives, optimizer groups, numerical/data issues, and a bounded tuning sequence.
- [Combined PDF](TERMINUS_ARCHITECTURE_RESEARCH.pdf): proposal plus research evidence and bibliography.
- [Primary-source evidence](RESEARCH_EVIDENCE.md) and [retrieval receipts](source_receipt.json).
- [Proposal-only specification](proposal_spec.json): explicit architecture and starting search ranges; **not loadable by the current trainer**.
- [Offline measurements](design_measurements.json) and [initialization diagnostic](initialization_diagnostic.json).
- [Validation receipt](validation_receipt.json): artifact hashes, arithmetic/link/layout checks, and unmeasured limits.

This package does not modify the training model, start a run, or make a main-run launch decision. The proposed architecture and hyperparameters are candidates; the new measurements are checkpoint-space arithmetic and CPU initialization checks, not trained-model quality results.

To reproduce the diagnostics, use Python with PyTorch. Figure generation also requires Matplotlib. From the repository root:

```bash
python3 docs/terminus/architecture_review_20261005/audit_architecture_evidence.py \
  --dense-checkpoint /path/to/verified/dense-final.pt \
  --output docs/terminus/architecture_review_20261005/design_measurements.json
python3 docs/terminus/architecture_review_20261005/probe_initial_parameterization.py
python3 docs/terminus/architecture_review_20261005/make_review_figures.py
```

From this directory, build the PDF with Pandoc and XeLaTeX:

```bash
pandoc ARCHITECTURE_AND_TUNING_PROPOSAL.md RESEARCH_EVIDENCE.md \
  -o TERMINUS_ARCHITECTURE_RESEARCH.pdf --pdf-engine=xelatex \
  -V geometry:margin=0.65in -V documentclass=article --toc \
  --include-in-header=pdf_layout.tex
```
