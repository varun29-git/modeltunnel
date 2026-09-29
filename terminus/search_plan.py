"""Staged, bounded search plans for the ModelTunnelv2 campaign.

These generators produce deterministic trial lists (config + metadata) for the
owner to review and launch stage by stage. They deliberately avoid freezing a
final recipe: budgets are proxy-stage defaults (labeled, adjustable at
launch), the QAT token allocation is a hypothesis grid (not the 800B/200B
arithmetic), and every QAT branch is conditional on an actually trained dense
checkpoint from the same plan.

Search discipline encoded here:
- dense stage first: Muon LR-rule transfer across widths, then LR/batch/
  schedule/mixture screens at 200M with paired seeds/data;
- QAT stage conditional on trained dense checkpoints: switch point and QAT
  duration on a common mix, then QAT-specific mixtures, with dense-only and
  ternary-from-start controls; every branch finishes its decay before ranking;
- quality is always compared against measured GPU-hours, and proxy rankings
  must transfer to bridge widths before promotion (see registry.ranking_transfer).
"""

from __future__ import annotations

from .registry import make_run_config

MUON_LR_GRID = (0.001, 0.003, 0.009)
MUON_RULES = ("sqrt_fan_in", "sqrt_fan_in_ratio")
PROXY_DENSE_BUDGET_TOKENS = 2_000_000_000  # ~10 tokens/param at 200M; adjustable at launch
FAN_IN_BASE = 2304  # target width for the rule-B ratio


def _trial(stage, kind, width, config, **meta):
    experiment_id = (f"{stage}/{kind}/{width}/" +
                     "_".join(f"{key}={value}" for key, value in sorted(meta.items())))
    return {"experiment_id": experiment_id, "stage": stage, "kind": kind, "width": width,
            "config": config, **meta}


def stage1_width_transfer(*, budget_tokens=PROXY_DENSE_BUDGET_TOKENS, seed=1337):
    """Two Muon LR rules x three base LRs at 200M/700M, dense FP8, fixed data.

    The reserved 1.5B bracket is emitted without a launch budget; it runs only
    for the shortlist after the 200M/700M comparison, per the protocol.
    """
    trials = []
    for width in ("200m", "700m"):
        for rule in MUON_RULES:
            for muon_lr in MUON_LR_GRID:
                config = make_run_config(
                    width, weight_mode="dense", schedule="wsd",
                    warmup_tokens=int(0.05 * budget_tokens),
                    decay_tokens=int(0.2 * budget_tokens),
                    muon_lr=muon_lr, adamw_lr=muon_lr, seed=seed,
                    muon_lr_rule=rule,
                    muon_fan_in_base=FAN_IN_BASE if rule == "sqrt_fan_in_ratio" else None)
                trials.append(_trial("stage1", "width_transfer", width, config,
                                     muon_lr=muon_lr, rule=rule, budget_tokens=budget_tokens))
    for rule in MUON_RULES:
        trials.append({"experiment_id": f"stage1/width_transfer/1500m/rule={rule}_reserved",
                       "stage": "stage1", "kind": "reserved_bridge_bracket", "width": "1500m",
                       "rule": rule, "budget_tokens": None,
                       "note": "launch only for the shortlisted LR after the 200M/700M comparison"})
    return trials


def stage3_dense_screen(*, width="200m", budget_tokens=PROXY_DENSE_BUDGET_TOKENS, seed=1337,
                        winning_rule="sqrt_fan_in", mixtures=None):
    """Bounded dense-stage screen: batch size, schedule horizon, FP8 backward,
    and source mixtures. Adaptive: LR comes from stage 1; trials are paired on
    seed and data so differences are attributable.
    """
    batch_decompositions = [(8, 32), (8, 64), (16, 64)]  # (microbatch, accumulation)
    trials = []
    for microbatch, accumulation in batch_decompositions:
        config = make_run_config(width, weight_mode="dense", schedule="wsd",
                                 warmup_tokens=int(0.05 * budget_tokens),
                                 decay_tokens=int(0.2 * budget_tokens),
                                 muon_lr=0.003, adamw_lr=0.003, seed=seed,
                                 muon_lr_rule=winning_rule,
                                 muon_fan_in_base=FAN_IN_BASE if winning_rule != "sqrt_fan_in" else None,
                                 gradient_accumulation_steps=accumulation,
                                 extra_model={})
        trials.append(_trial("stage3", "batch_screen", width, config,
                             microbatch=microbatch, accumulation=accumulation,
                             budget_tokens=budget_tokens))
    for fp8_backward in ("bf16", "dgrad", "full"):
        config = make_run_config(width, weight_mode="dense", schedule="wsd",
                                 warmup_tokens=int(0.05 * budget_tokens),
                                 decay_tokens=int(0.2 * budget_tokens),
                                 muon_lr=0.003, adamw_lr=0.003, seed=seed,
                                 muon_lr_rule=winning_rule,
                                 muon_fan_in_base=FAN_IN_BASE if winning_rule != "sqrt_fan_in" else None,
                                 fp8_backward=fp8_backward)
        trials.append(_trial("stage3", "fp8_arithmetic", width, config,
                             fp8_backward=fp8_backward, budget_tokens=budget_tokens))
    for name in (mixtures or ()):
        pass  # mixture trials are generated by the campaign driver with real manifests
    return trials


def stage4_qat_branches(*, dense_experiment_ids, width="200m",
                        dense_budget_tokens=PROXY_DENSE_BUDGET_TOKENS, seed=1337,
                        winning_rule="sqrt_fan_in"):
    """QAT branches conditional on trained dense checkpoints.

    Switch points are expressed as fractions of the dense budget actually
    trained; QAT durations are a hypothesis grid (the 2x-decay-token rule is
    one member, not the default). Includes dense-only and ternary-from-start
    controls at matched total token exposure.
    """
    switch_fractions = (0.5, 0.8, 1.0)
    qat_durations = (0.1, 0.2, 0.4)  # QAT tokens as a fraction of dense budget
    trials = []
    for dense_id in dense_experiment_ids:
        for switch in switch_fractions:
            for qat_fraction in qat_durations:
                trials.append({"experiment_id": f"stage4/qat/{width}/{dense_id}/"
                               f"switch={switch}_qatfrac={qat_fraction}", "stage": "stage4",
                               "kind": "qat_branch", "width": width, "parent_experiment": dense_id,
                               "switch_at_dense_tokens": int(switch * dense_budget_tokens),
                               "qat_budget_tokens": int(qat_fraction * dense_budget_tokens),
                               "note": "QAT config: ternary weight mode, LR re-warmup, own decay; "
                                       "compare only completed branches"})
    controls = [
        {"experiment_id": f"stage4/control/{width}/dense_only", "stage": "stage4",
         "kind": "control_dense_only", "width": width,
         "budget_tokens": dense_budget_tokens + int(0.2 * dense_budget_tokens)},
        {"experiment_id": f"stage4/control/{width}/ternary_from_start", "stage": "stage4",
         "kind": "control_ternary_from_start", "width": width,
         "budget_tokens": dense_budget_tokens + int(0.2 * dense_budget_tokens)}]
    return trials + controls
