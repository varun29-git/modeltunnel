"""Staged, cost-capped search plan for the first ModelTunnelv2 campaign.

Replaces the old 12-run Cartesian width/LR grid with a staged plan whose jobs
carry conservative GPU-hour estimates against a hard cap. Promotion between
stages is decided from the ledger (`campaign.promotion_winners`) at runtime;
this module emits the *plan* (with assumed winners for dependent stages so the
dry run can show everything) — nothing here launches anything.

Budgets below are defaults for the dry-run example (~52 conservative H100-hours
under the labeled assumed rates); every horizon is a parameter. If measured
throughput makes the plan exceed the cap, `campaign.reprice` shortens or drops
the lowest-priority trials — the cap is never silently exceeded.
"""

from __future__ import annotations

from .campaign import Job
from .registry import make_run_config

MUON_LR_GRID = (0.001, 0.003, 0.009)      # small bracket around the archived 0.003
MUON_RULES = ("sqrt_fan_in", "sqrt_fan_in_ratio")
FAN_IN_BASE = 2304

# Real, versioned mixture overlays over the 10 campaign streams (weights in
# 0.001 increments; English takes the residual so sums are exactly 1.0).
def _mix(**weights):
    assert "en" not in weights
    weights["en"] = 1.0 - sum(weights.values())
    assert weights["en"] > 0
    return weights

MIXTURES = {
    "baseline": _mix(hi=0.15, mr=0.08, kn=0.08, te=0.08, ta=0.08, gu=0.08, bn=0.07,
                     math=0.04, code=0.04),
    "indic_heavy": _mix(hi=0.22, mr=0.12, kn=0.12, te=0.12, ta=0.12, gu=0.12, bn=0.10,
                        math=0.03, code=0.03),
    "stem_heavy": _mix(hi=0.11, mr=0.06, kn=0.06, te=0.06, ta=0.06, gu=0.06, bn=0.05,
                       math=0.14, code=0.12),
    "en_heavy": _mix(hi=0.09, mr=0.05, kn=0.05, te=0.05, ta=0.05, gu=0.05, bn=0.045,
                     math=0.03, code=0.03),
}

# Dry-run token budgets (adjustable; costs scale linearly in tokens).
STAGE1_TOKENS = 600_000_000
STAGE2_TOKENS_700M = 1_000_000_000
STAGE2_TOKENS_1500M_CHECK = 500_000_000
STAGE3_TOKENS = 600_000_000
STAGE4_PARENT_TOKENS = 600_000_000
STAGE4_SWITCH_FRACTIONS = (0.5, 0.8)
STAGE4_QAT_DURATIONS = (120_000_000, 240_000_000)  # 20% / 40% of the parent budget


def _config(width, weight_mode, tokens, *, muon_lr=0.003, rule="sqrt_fan_in", seed=1337,
            decay_fraction=0.2, accumulation=64):
    return make_run_config(width, weight_mode=weight_mode, schedule="wsd",
                           warmup_tokens=int(0.05 * tokens),
                           decay_tokens=int(decay_fraction * tokens),
                           muon_lr=muon_lr, adamw_lr=muon_lr, seed=seed,
                           muon_lr_rule=rule,
                           muon_fan_in_base=FAN_IN_BASE if rule != "sqrt_fan_in" else None,
                           gradient_accumulation_steps=accumulation)


def campaign_plan(*, seed=1337):
    """The full staged plan as a list of budgeted jobs (nothing is launched)."""
    jobs = []

    # Stage 0: native H100 gates — numerical preflight + sustained throughput
    # measurement for dense FP8 and ternary QAT. Everything downstream reprices
    # from these measurements.
    for phase in ("dense", "ternary"):
        jobs.append(Job(job_id=f"stage0/preflight_{phase}", stage="stage0",
                        kind="preflight_throughput", width="1500m", phase=phase,
                        priority=0, notes="measures real tokens/s; reprices all later jobs"))

    # Stage 1: 200M width-transfer — two Muon rules x LR bracket, dense FP8.
    # One run's checkpoints supply held-out LM loss (validation events) and
    # task loss (scored on the same checkpoints) — no duplicate campaigns.
    for rule in MUON_RULES:
        for muon_lr in MUON_LR_GRID:
            jobs.append(Job(job_id=f"stage1/200m/{rule}/lr={muon_lr}", stage="stage1",
                            kind="width_transfer", width="200m", phase="dense",
                            tokens=STAGE1_TOKENS, microbatch=8, accumulation=64,
                            priority=10, notes=f"config={_config('200m', 'dense', STAGE1_TOKENS, muon_lr=muon_lr, rule=rule, seed=seed)['training']['schedule']}"))

    # Stage 2: promote top-2 to 700M (placeholders resolved by the gate), plus
    # one short 1.5B transfer check. No broad 1.5B grid.
    for slot in (1, 2):
        jobs.append(Job(job_id=f"stage2/700m/winner_{slot}", stage="stage2",
                        kind="bridge", width="700m", phase="dense",
                        tokens=STAGE2_TOKENS_700M, microbatch=8, accumulation=64,
                        priority=20, depends_on_stage="stage1"))
    jobs.append(Job(job_id="stage2/1500m/transfer_check", stage="stage2",
                    kind="transfer_check", width="1500m", phase="dense",
                    tokens=STAGE2_TOKENS_1500M_CHECK, microbatch=8, accumulation=64,
                    priority=40, depends_on_stage="stage1",
                    notes="single short check only; not a grid"))

    # Stage 3: mixture overlays at 200M with the stage-1 winner's LR/rule.
    for name in MIXTURES:
        jobs.append(Job(job_id=f"stage3/200m/mix={name}", stage="stage3",
                        kind="mixture", width="200m", phase="dense",
                        tokens=STAGE3_TOKENS, microbatch=8, accumulation=64,
                        priority=30, depends_on_stage="stage1",
                        notes=f"manifest=manifest.{name}.json (remix overlay)"))

    # Stage 4: QAT branches from the selected dense checkpoint (shared parent),
    # fully decayed, with matched-exposure controls.
    parent = "stage4/200m/parent_dense"
    jobs.append(Job(job_id=parent, stage="stage4", kind="qat_parent", width="200m",
                    phase="dense", tokens=STAGE4_PARENT_TOKENS, microbatch=8,
                    accumulation=64, priority=15, depends_on_stage="stage3"))
    for switch in STAGE4_SWITCH_FRACTIONS:
        switch_tokens = int(switch * STAGE4_PARENT_TOKENS)
        for qat_tokens in STAGE4_QAT_DURATIONS:
            jobs.append(Job(job_id=f"stage4/200m/qat/switch={switch}/dur={qat_tokens}",
                            stage="stage4", kind="qat_branch", width="200m", phase="qat",
                            tokens=qat_tokens, microbatch=8, accumulation=64,
                            priority=25, parent_job=parent, depends_on_stage="stage3",
                            notes=f"exposure={switch_tokens + qat_tokens}; decay completes at phase end"))
    max_exposure = int(max(STAGE4_SWITCH_FRACTIONS) * STAGE4_PARENT_TOKENS) + max(STAGE4_QAT_DURATIONS)
    jobs.append(Job(job_id="stage4/200m/control_dense_continuation", stage="stage4",
                    kind="control_dense_only", width="200m", phase="dense",
                    tokens=max_exposure - STAGE4_PARENT_TOKENS, microbatch=8, accumulation=64,
                    eval_checkpoints=len(STAGE4_SWITCH_FRACTIONS) * len(STAGE4_QAT_DURATIONS),
                    priority=25, parent_job=parent, depends_on_stage="stage3",
                    notes="continues the shared parent; evaluated at each branch's matched exposure"))
    jobs.append(Job(job_id="stage4/200m/control_ternary_from_start", stage="stage4",
                    kind="control_ternary_from_start", width="200m", phase="control_ternary",
                    tokens=max_exposure, microbatch=8, accumulation=64,
                    eval_checkpoints=len(STAGE4_SWITCH_FRACTIONS) * len(STAGE4_QAT_DURATIONS),
                    priority=35, depends_on_stage="stage3",
                    notes=f"fresh ternary run to the max matched exposure {max_exposure}"))

    # Stage 5: second seeds for up to two close finalists, only if budget remains.
    for slot in (1, 2):
        jobs.append(Job(job_id=f"stage5/200m/second_seed_{slot}", stage="stage5",
                        kind="seed_replication", width="200m", phase="dense",
                        tokens=STAGE1_TOKENS, microbatch=8, accumulation=64,
                        priority=90, depends_on_stage="stage4",
                        notes="runs only if the repriced budget has room"))
    return jobs


def job_config(job, *, seed=1337):
    """The runnable training config for a plan job.

    QAT branches get ternary mode with their own re-warmup and decay window
    that completes exactly at the phase budget; phase_start_tokens is set by
    the transition tool at conversion time (never guessed here).
    """
    if job.stage == "stage0":
        return None  # preflight probes use the config templates directly
    if job.phase == "qat":
        config = _config(job.width, "ternary", job.tokens, muon_lr=0.001, seed=seed,
                         decay_fraction=0.4, accumulation=job.accumulation)
        config["training"]["warmup_tokens"] = int(0.1 * job.tokens)
        return config
    if job.phase == "control_ternary":
        return _config(job.width, "ternary", job.tokens, seed=seed,
                       accumulation=job.accumulation)
    rule = "sqrt_fan_in"
    muon_lr = 0.003
    if job.stage == "stage1":
        rule = job.job_id.split("/")[2]
        muon_lr = float(job.job_id.split("lr=")[1])
    return _config(job.width, "dense", job.tokens, muon_lr=muon_lr, rule=rule,
                   seed=seed, accumulation=job.accumulation)


def matched_exposure_check(jobs):
    """Every QAT branch must have control coverage at identical total exposure."""
    branches = [j for j in jobs if j.kind == "qat_branch"]
    parent_tokens = {j.job_id: j.tokens for j in jobs if j.kind == "qat_parent"}
    controls = {j.kind: j for j in jobs if j.kind.startswith("control_")}
    problems = []
    for branch in branches:
        switch = float(branch.job_id.split("switch=")[1].split("/")[0])
        switch_tokens = int(switch * parent_tokens[branch.parent_job])
        exposure = switch_tokens + branch.tokens
        dense_control = controls["control_dense_only"]
        ternary_control = controls["control_ternary_from_start"]
        if dense_control.eval_checkpoints < 1 or ternary_control.eval_checkpoints < 1:
            problems.append(f"{branch.job_id}: controls lack evaluation checkpoints")
        if ternary_control.tokens < exposure:
            problems.append(f"{branch.job_id}: ternary control stops below matched exposure")
        if dense_control.tokens + parent_tokens[branch.parent_job] < exposure:
            problems.append(f"{branch.job_id}: dense control stops below matched exposure")
    return problems
