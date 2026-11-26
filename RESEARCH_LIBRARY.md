# Research library and reusable sources

These canonical sources inform the implementation and experimental design. They are not drop-in implementations, and their numeric hyperparameters do not automatically transfer to Terminus. Third-party paper PDFs are not bundled; use the links below and retain each source's terms.

| Research source | Relevance | Canonical link |
| --- | --- | --- |
| MiniCPM4 | Sections 2.3 and 3.3: ModelTunnelv2 task-loss signal and the FP8-to-ternary BitCPM4 pilot | [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) |
| Densing Law | ScalingBench construction and task-conditioned loss | [arXiv:2412.04315](https://arxiv.org/abs/2412.04315) |
| Continual QAT | Alternative dense/QAT transition and optimizer-state evidence | [arXiv:2502.11895](https://arxiv.org/abs/2502.11895) |
| ParetoQ | Ternary quantizer and transition alternatives; compare mechanisms rather than assuming its optimum | [arXiv:2502.02631](https://arxiv.org/abs/2502.02631) |
| Tensor Programs V / μP | Conditions and diagnostics for hyperparameter transfer across width | [arXiv:2203.03466](https://arxiv.org/abs/2203.03466) |
| Muon is Scalable | Muon shape/update scaling and optimizer context | [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) |
| Step-by-Step | Recent transfer work relevant to Muon; its MoE setting is not direct Terminus validation | [arXiv:2608.20061](https://arxiv.org/abs/2608.20061) |
| DoReMi | Proxy-model data-mixture optimization ideas | [arXiv:2305.10429](https://arxiv.org/abs/2305.10429) |
| RegMix | Regression-based mixture search and cost-effective candidate ranking | [arXiv:2407.01492](https://arxiv.org/abs/2407.01492) |

Reusable software and task sources to inspect before integration: [ScalingBench](https://huggingface.co/datasets/openbmb/DensingLaw-ScalingBench), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [DataTrove](https://github.com/huggingface/datatrove), [Microsoft μP](https://github.com/microsoft/mup), [ParetoQ reference code](https://github.com/facebookresearch/ParetoQ), and [MILU](https://huggingface.co/datasets/ai4bharat/MILU). Check each source's exact revision and license before integrating or redistributing it. The pinned ScalingBench revision used by `scripts/fetch_eval_tasks.py` is recorded in `DATA_SOURCE_MAP.md`.
