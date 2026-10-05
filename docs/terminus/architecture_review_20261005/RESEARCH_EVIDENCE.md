# Terminus architecture research: evidence and applicability

5 October 2026. This review reads the maintained `terminus/` implementation at `7ae65ff`, the verified dense/QAT checkpoint audit, primary papers, and released model code. It proposes experiments and architecture choices; it does not supply a decision about launching the main run. Downloaded source hashes, resolved Hugging Face revisions, arXiv versions where available, and observed GitHub heads are in [source_receipt.json](source_receipt.json). Raw downloads remain outside git. Failed URL guesses are retained in the receipt rather than presented as sources read.

## What changes the interpretation of the completed runs

1. **MiniCPM's parameterization was not reproduced.** The released MiniCPM4-0.5B code multiplies each residual branch by `scale_depth / sqrt(num_hidden_layers)` during every forward pass, scales embedding outputs, and scales the readout. Its config specifies `scale_depth=1.4`, `scale_emb=12`, and `dim_model_base=256`. Terminus applies a small factor only to O/down initialization, has no runtime residual multiplier, and uses a different embedding/readout parameterization. Matching the printed peak LR therefore does not match the optimization process. [S1, S2]

2. **The current ternary grid is coupled to the GEMM tile.** Terminus uses one mean-absolute scale per 128×128 weight tile (16,384 values). BitCPM-CANN explicitly studies per-row groups of 32/64 and reports better losses than per-channel scaling. Its paper also uses a statistical absmean scale and identity STE successfully. Consequently, neither absmean nor identity STE is by itself a demonstrated bug. Grid size, optimization, and architecture must be separated. [S4]

3. **A lower QAT LR is not established as the remedy.** MiniCPM4 reports 0.01 dense / 0.005 QAT in its ModelTunnel experiment, whereas BitCPM-CANN's 0.5B ablation reports better loss and task scores at 0.01 than 0.005, and degradation at 0.015. These are different recipes from Terminus/Muon; they support an independent QAT sweep rather than either number as a universal rule. [S1, S4]

4. **Large changes to latent weights can be normal at ternary precision.** ParetoQ distinguishes low-bit representation reconstruction from the smaller compensation seen at 3–4 bits. The Terminus increase in latent RMS and zeros does not, alone, prove failed learning. Held-out loss, task results, and matched controls remain the functional evidence. [S5]

5. **Muon and $\mu$P are not categorically incompatible.** Published Muon scaling and the 2026 step-by-step transfer work support studying coherent parameterization and transfer rules. They do not validate mixing arbitrary $\mu$P readout multipliers with Terminus's current Muon rule. The latter paper is mainly MoE, with a dense appendix; its width and token-horizon transfer conditions matter. [S7–S9]

6. **Normalization has different jobs in different locations.** Qwen3 uses pre-RMSNorm and learned Q/K normalization. OLMo2 uses a different output-normalized design, Q/K normalization, embedding-decay changes, and z-loss. Neither implies that Terminus should stack all of those choices. Learned Q/K normalization is a targeted attention-logit control; SubLN and residual-branch normalization are separate changes. [S10–S12]

## New offline checkpoint measurements

The script [audit_architecture_evidence.py](audit_architecture_evidence.py) rehashes the dense checkpoint and evaluates all 104,448,000 projection weights. It does not run model inference or training. [design_measurements.json](design_measurements.json) records these results and proposed parameter/storage counts.

| Weight representation | Relative RMS reconstruction error | Effective RMS / dense RMS | Zero fraction |
|---|---:|---:|---:|
| Current 128×128 absmean | 0.5291 | 0.6510 | 31.70% |
| Same codes, optimal least-squares scale | 0.4777 | 0.8785 | 31.70% |
| 128×128 alternating scale/code calibration, 8 iterations | 0.4508 | 0.8926 | 47.60% |
| Row groups of 128, absmean | 0.5206 | 0.6584 | 31.23% |
| Row groups of 64, absmean | 0.5183 | 0.6610 | 31.11% |
| Row groups of 32, absmean | 0.5136 | 0.6661 | 30.85% |

The fixed-code row is a diagnostic: recomputing nearest ternary codes using its new scale would change the assignments. Alternating calibration recomputes assignments with the final scale. Smaller row groups improve this weight-space metric modestly; better language-model quality remains unmeasured. The extra FP8 encoding required for finer ternary groups is not included in this table.

## Initialization diagnostic for the proposed parameterization

[probe_initial_parameterization.py](probe_initial_parameterization.py) applies the proposed width-512 block equations with the existing CPU reference modules, including learned Q/K gains initialized to one, runtime residual scaling, and scaled tied embeddings. It uses random token IDs and FP32 arithmetic; it is not a training or task-quality experiment. Its count is exactly 118,783,744 parameters, matching the proposed proxy.

An unscaled tied readout assigned an average 99.97% probability to the current token because its embedding remains in the residual stream. Scaling the readout by `2/sqrt(width)` gave entropy 11.086 nats, near the uniform 65K entropy of 11.090. This caught an initialization issue in the draft parameterization; it does **not** identify a bug in the old trained model. [initialization_diagnostic.json](initialization_diagnostic.json) records all comparisons. The corrected readout factor is part of the parameterization to test jointly with Muon and auxiliary-LR transfer, not a proven optimum.

## Primary-source reading map

**[S1] MiniCPM4** — [paper](https://arxiv.org/abs/2506.07900), Sections 2.3 and 3.3. ModelTunnelv2 uses task-conditioned loss; MTP and FP8 are engineering components. BitCPM4 uses a trained dense parent, LR re-warmup, and weight-only ternary QAT. The 5M/2B fixed-total pilot's >40% QAT observation concerns approaching ternary-from-start loss, not a guarantee of dense parity or a universal final token budget. FP8 activations in Terminus remain a deliberate requirement. Its 350B reported QAT budget is not transferable by arithmetic alone.

**[S2] MiniCPM parameterization** — [original paper](https://arxiv.org/abs/2404.06395), Appendix A.1; [released MiniCPM4 model code](https://huggingface.co/openbmb/MiniCPM4-0.5B/blob/5253c7fcc5e29e1cf3eacb59a58adf1ba4df8630/modeling_minicpm.py); [config](https://huggingface.co/openbmb/MiniCPM4-0.5B/blob/5253c7fcc5e29e1cf3eacb59a58adf1ba4df8630/config.json). Runtime branch/embedding/readout scales are concrete implementation facts. Copying only one multiplier does not copy the complete parameterization.

**[S3] Released BitCPM4** — [model and card](https://huggingface.co/openbmb/BitCPM4-0.5B/tree/4d0e4a9e2257ceb39d99b85bb946083467e8476f). It ships `modeling_minicpm.py` and floating-point inference weights. This release is not a complete QAT training implementation or evidence of H100 ternary-kernel speed.

**[S4] BitCPM-CANN** — [technical report](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/docs/BitCPM_CANN.pdf), Sections 3.2–3.3 and 5.4, Tables 5–7; [model card](https://huggingface.co/openbmb/BitCPM-CANN-1B). Directly useful group-size, LR, and delayed-distillation ablations. Table 6: 0.005 versus 0.01 gives decay loss 1.661 versus 1.646; 0.015 degrades. Table 7: per-channel versus g64 versus g32 gives 1.661 / 1.646 / 1.635. These are authors' 0.5B measurements, not Terminus results, and the NPU throughput does not transfer to H100. The authors attribute remaining small-model reasoning gaps partly to capacity; that is an interpretation, not proof of Terminus's bottleneck.

**[S5] ParetoQ** — [paper](https://arxiv.org/abs/2502.02631), Sections 2–3; [quantizer implementation](https://github.com/facebookresearch/ParetoQ/blob/7b36b8de958aada6508e620c8dc544e6ed6c39b4/models/utils_quant.py). Learnable scales, quantization-grid geometry, scaled scale-gradients, and bit-dependent optimization matter. Its SEQ ternary representation can be expressed as scaled {-1,0,1}, but forward thresholds and backward rules must be ported together. Its published implementation is not a native DeepGEMM replacement. Check BSD license and preserve attribution if code is reused.

**[S6] Continual QAT** — [paper](https://arxiv.org/abs/2502.11895). Carrying optimizer state and gradually introducing quantization can reduce transition spikes, but longer training can compensate some differences. It supports explicit controls. A dense/ternary blend is not an exactly ternary model during the blend, and the existing native Terminus path currently requires full quantization strength.

**[S7] Muon is Scalable** — [paper](https://arxiv.org/abs/2502.16982), Section 2.2. It explicitly connects growing weight/activation RMS to precision concerns and motivates weight decay and RMS-matched updates. This makes Muon LR, weight decay, and runtime branch scale a coherent group of tuning variables.

**[S8] Tensor Programs V / $\mu$P** — [paper](https://arxiv.org/abs/2203.03466). Parameterization and width-transfer rules are a package; optimal LR transfer is not guaranteed by applying one readout factor.

**[S9] Let's Scale Step by Step** — [paper](https://arxiv.org/abs/2608.20061), methods, Appendix A and dense-model Appendix C. Uses Muon and separates width transfer from token-horizon transfer. Useful counterevidence to a blanket “Muon cannot use $\mu$P” claim. Its MoE setting and complete scaling prescription cannot be silently substituted for our dense family.

**[S10] Qwen3** — [paper](https://arxiv.org/abs/2505.09388), architecture section; [1.7B-Base config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/config.json). Supports a bias-free GQA/SwiGLU/pre-RMSNorm backbone and learned Q/K normalization. Its model/data/tokenizer and full training recipe differ.

**[S11] OLMo2** — [paper](https://arxiv.org/abs/2501.00656), Section 2.1, Table 1. Useful evidence for Q/K normalization, removal of embedding weight decay, and z-loss. Its output-norm placement is a different architecture, not an additional norm to stack onto ours without an ablation.

**[S12] Small-scale proxies for Transformer instabilities** — [paper](https://arxiv.org/abs/2309.14322), Section 3. Attention-logit growth and output-logit drift can be reproduced with small high-LR experiments. Use loss-versus-LR curves and targeted controls; finite training is insufficient to identify the best recipe.

**[S13] Gemma 3** — [paper](https://arxiv.org/abs/2503.19786), local/global and sliding-window ablations. Supports retaining hybrid attention and studying smaller windows. Its 5:1 ratio and 1024 window are not our required 4:1 and selected 512; reported negligible loss changes are specific to their tested models.

**[S14] SmolLM3** — [official article](https://huggingface.co/blog/smollm3), architecture and training sections. Useful GQA, document masking, embedding-decay, staged-data, and hybrid-NoPE experience. Only the article body is used as evidence; community comments were not treated as findings. NoPE is a separate long-context candidate, not necessary at our initial 4K context.

**[S15] DeepSeek-V3** — [paper](https://arxiv.org/abs/2412.19437), Section 3.3. E4M3 for forward and backward can work with fine-grained online scaling, adequate accumulation precision, and high-precision master weights/gradients. This supports keeping FP8 activations while auditing gradient orientations and accumulation; a sampled input-quantization error does not validate the entire training path.

**[S16] COAT** — [paper](https://arxiv.org/abs/2410.19313), [code](https://github.com/NVlabs/COAT/tree/80ec99f47aaa09231b07ace1fd04c30a1e30ec18). Useful normalization/activation/quantization fusion and activation-storage ideas. Its optimizer-state compression and measured end-to-end speedups should not be imported into Muon or assumed for Terminus.

**[S17] CODA** — [paper](https://arxiv.org/abs/2605.19269). GEMM epilogue and block-composition techniques may remove memory traffic for norms, residuals, activations, and loss. It is primarily a systems improvement, not evidence of model-quality gains; precision and algebraic rewrites must match the chosen model.

**[S18] Gram Newton-Schulz** — [code and documentation](https://github.com/Dao-AILab/gram-newton-schulz/tree/e45d0aca7083cb275c9a303220c05c4abecd9187). Reuses smaller Gram matrices and hardware-aware symmetric GEMMs. “Up to 2×” refers to the NS component. Default polynomial coefficients, precision, restart policy, and full optimizer scaling must be checked against the existing pinned Muon before claiming an equivalent optimizer.

**[S19] DeepGEMM** — [current source](https://github.com/deepseek-ai/DeepGEMM/tree/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7). The README dates dense/MoE weight-gradient support to 14 May 2025 and describes subsequent SM90/SM100 and JIT work. It does not establish that every new specialized kernel benefits this dense model. Benchmark required projection shapes at a pinned revision.

**[S20] Massive Activations** — [paper](https://arxiv.org/abs/2402.17762). Large coordinates can carry structured functions. This argues for functional ablations and token-position analysis, not automatically clipping all large activations.

**[S21] QuaRot** — [paper](https://arxiv.org/abs/2404.00456), and **[S22] SpinQuant** — [paper](https://arxiv.org/abs/2405.16406). Rotations can improve low-bit inference in their settings. They require careful handling of RMSNorm gains, residual basis, tied embeddings, and nonlinearities; their 4-bit results do not establish ternary gains.

**[S23] BitNet b1.58 2B4T** — [paper](https://arxiv.org/abs/2504.12285). A valuable native-ternary baseline, with a different training history and precision/architecture choices. Its quality does not require Terminus to reinstate all BitNet-era components.

**[S24] RegMix** — [paper](https://arxiv.org/abs/2407.01492), and **[S25] DoReMi** — [paper](https://arxiv.org/abs/2305.10429). Cost-effective data-mixture search. Refit and validate on our eight-language/domain family; neither supplies an optimal Indic/code/math split.

**[S26] DataComp-LM** — [paper](https://arxiv.org/abs/2406.11794), and **[S27] FineWeb** — [paper](https://arxiv.org/abs/2406.17557). Data filtering, deduplication, and selection have large quality effects. English findings motivate tests rather than universal rules for translated Indic text and code.

## Source-level findings that affect the tuning implementation

- `terminus/transition.py` loads the parent's optimizer state and then restores only child `lr`/`base_lr`. Changed child weight decay, momentum, betas, and eps can therefore be silently replaced by parent settings. The completed run used matching non-LR settings, so this is a prospective HPO issue rather than evidence that its applied settings were wrong.
- In the same function, an explicit RNG reset is implemented only in the world-size-mismatch branch; with equal world size, the parent RNG is reused. Separate this from the default intentional carry behavior.
- `terminus/train.py` scales loss by nominal update-token count, while `model.py` separately computes the number of non-ignored labels. The packed run has no masked training labels, so those counts agree. Adding answer masks or dropped document-boundary targets requires normalization by the global number of valid labels.
- `terminus/stability.py` samples the last query over only the last 128 keys, and up to 256 coordinates per physical parameter. It does not measure global-attention entropy over the full allowed context, all logical Q/K/V updates, or population activation tails.
- `terminus/optim.py` correctly divides weight decay by its per-matrix Muon LR multiplier. This preserves the base `lr*weight_decay`; it is **not** the double-weight-decay bug one might suspect from the saved large group LRs.
- `terminus/logical_muon.py` already orthogonalizes Q/K/V and gate/up separately within packed matrices. There is no need to unfuse their GEMMs to obtain logical Muon updates. Separate logical LR multipliers would still need implementation.

## Reproduction

Run from the repository root:

```bash
python3 docs/terminus/architecture_review_20261005/audit_architecture_evidence.py \
  --dense-checkpoint /path/to/verified/dense-final.pt \
  --output /tmp/design_measurements.json
```

The script reproduces the existing 146,423,680 parameter count exactly and reports the proposed family's counts. All proposed H100 performance, task-quality, and new-architecture claims remain experiments rather than measurements.
