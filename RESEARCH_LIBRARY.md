# Research library and reusable sources

The PDFs below were downloaded from the cited arXiv IDs on 29 September 2026 and checked to open as PDFs with matching first-page titles. They are supporting evidence, **not a drop-in implementation** or a claim that their numeric hyperparameters transfer to Terminus.

| Local PDF | Why the agent should read it | Original |
| --- | --- | --- |
| [`minicpm4_2506.07900.pdf`](papers/minicpm4_2506.07900.pdf) | Sections 2.3 and 3.3: ModelTunnelv2 task-loss signal and the FP8-to-ternary BitCPM4 pilot. | https://arxiv.org/abs/2506.07900 |
| [`densing_law_2412.04315.pdf`](papers/densing_law_2412.04315.pdf) | ScalingBench construction and task-conditioned loss. | https://arxiv.org/abs/2412.04315 |
| [`continual_qat_2502.11895.pdf`](papers/continual_qat_2502.11895.pdf) | Alternative dense/QAT transition and optimizer-state evidence. | https://arxiv.org/abs/2502.11895 |
| [`paretoq_2502.02631.pdf`](papers/paretoq_2502.02631.pdf) | Ternary quantizer and transition alternatives; compare mechanisms, do not assume its optimum. | https://arxiv.org/abs/2502.02631 |
| [`tensor_programs_v_mup_2203.03466.pdf`](papers/tensor_programs_v_mup_2203.03466.pdf) | Conditions and diagnostics for hyperparameter transfer across width. | https://arxiv.org/abs/2203.03466 |
| [`muon_scalable_2502.16982.pdf`](papers/muon_scalable_2502.16982.pdf) | Muon shape/update scaling and optimizer context. | https://arxiv.org/abs/2502.16982 |
| [`step_by_step_2608.20061.pdf`](papers/step_by_step_2608.20061.pdf) | Recent transfer work relevant to Muon, but its MoE setting is not direct Terminus validation. | https://arxiv.org/abs/2608.20061 |
| [`doremi_2305.10429.pdf`](papers/doremi_2305.10429.pdf) | Proxy-model data-mixture optimization ideas. | https://arxiv.org/abs/2305.10429 |
| [`regmix_2407.01492.pdf`](papers/regmix_2407.01492.pdf) | Regression-based mixture search and cost-effective candidate ranking. | https://arxiv.org/abs/2407.01492 |

Potential reusable software and task sources for the coding agent to inspect later: [ScalingBench](https://huggingface.co/datasets/openbmb/DensingLaw-ScalingBench), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [DataTrove](https://github.com/huggingface/datatrove), [Microsoft μP](https://github.com/microsoft/mup), [ParetoQ reference code](https://github.com/facebookresearch/ParetoQ), and [MILU](https://huggingface.co/datasets/ai4bharat/MILU). Check each source's current revision and license before integrating it.
