"""Cost-capped campaign runner: budgeted jobs, ledger accounting, dry-run plan.

Hard cap discipline:
- Every job carries a *conservative* runtime estimate (planned against 60% of
  the assumed throughput, plus evaluation and checkpoint overhead).
- The runner refuses to launch a job whose conservative estimate exceeds the
  remaining budget. Failed jobs, evaluation, and checkpoint overhead all count
  toward the cap.
- Stage 0 measures real sustained throughput; `reprice` then re-estimates all
  remaining jobs with measured rates and drops the lowest-priority jobs if the
  projection exceeds the cap. The cap is never silently exceeded.
- One training run supplies both signals: held-out language-model loss comes
  from the training run's validation events on its checkpoints, and the same
  checkpoints are scored by the task evaluator. There are no duplicate training
  campaigns for the two losses.
- `--execute` submits jobs on the target host; a dry run (the default and the
  only mode exercised here) prints every proposed job and the total spend.

All throughput numbers below are ASSUMPTIONS for planning, labeled as such;
the only measured rate is the 1.5B synthetic H100 baseline (46,395 tok/s).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .integrity import fingerprint, write_json_atomic
from .registry import Ledger

# ASSUMED sustained tokens/s on one H100 (unmeasured except 1500m synthetic).
ASSUMED_TOKENS_PER_SECOND = {
    ("200m", "dense"): 150_000, ("200m", "ternary"): 130_000,
    ("700m", "dense"): 70_000, ("700m", "ternary"): 62_000,
    ("1500m", "dense"): 46_395, ("1500m", "ternary"): 46_395,  # measured, synthetic
}
RATE_BASIS = {"1500m": "measured synthetic (evidence/muon_25l_readme.md)",
              "200m": "ASSUMED (unmeasured)", "700m": "ASSUMED (unmeasured)"}
CONSERVATIVE_FACTOR = 0.6  # plan as if sustained throughput were 40% lower
# ASSUMED overheads per job (H100-hours).
EVAL_HOURS_PER_CHECKPOINT = {"200m": 0.05, "700m": 0.12, "1500m": 0.25}
CHECKPOINT_OVERHEAD_HOURS = 0.02
PREFLIGHT_HOURS = 0.75


class BudgetRefused(RuntimeError):
    pass


@dataclass
class Job:
    job_id: str
    stage: str
    kind: str
    width: str
    phase: str                      # dense | qat | control_dense | control_ternary | preflight
    tokens: int = 0
    microbatch: int = 1
    accumulation: int = 1
    seq_len: int = 4096
    world_size: int = 1
    eval_checkpoints: int = 1
    priority: int = 50              # higher drops first when repricing over the cap
    parent_job: str | None = None   # shared dense parent for QAT branches
    depends_on_stage: str | None = None
    assumed_rate: float | None = None  # override; default from the table
    notes: str = ""

    @property
    def global_batch_tokens(self):
        return self.microbatch * self.world_size * self.accumulation * self.seq_len

    @property
    def updates(self):
        return -(-self.tokens // self.global_batch_tokens) if self.tokens else 0

    def estimate_hours(self, *, rates=None, conservative=True):
        if self.stage == "stage0":
            return PREFLIGHT_HOURS
        rate = (rates or {}).get((self.width, self.phase_key)) or self.assumed_rate
        if rate is None:
            rate = ASSUMED_TOKENS_PER_SECOND.get((self.width, self.phase_key))
        if rate is None:
            raise ValueError(f"no assumed rate for {self.width}/{self.phase_key}")
        effective = rate * (CONSERVATIVE_FACTOR if conservative else 1.0)
        train = self.tokens / effective / 3600
        eval_hours = self.eval_checkpoints * EVAL_HOURS_PER_CHECKPOINT[self.width]
        return train + eval_hours + CHECKPOINT_OVERHEAD_HOURS

    @property
    def phase_key(self):
        return "ternary" if self.phase in ("qat", "control_ternary") else "dense"


class Campaign:
    """Budget ledger over a registry Ledger, with launch gating."""

    def __init__(self, cap_hours, ledger: Ledger):
        if cap_hours <= 0:
            raise ValueError("cap must be positive")
        self.cap_hours = cap_hours
        self.ledger = ledger

    def spent_hours(self):
        return sum(a.get("cost", {}).get("gpu_hours") or 0.0 for a in self.ledger.attempts()
                   if a.get("outcome") in {"completed", "oom", "nonfinite_loss",
                                           "data_depleted", "contract_violation",
                                           "infrastructure", "interrupted", "unknown"})

    def remaining_hours(self):
        return self.cap_hours - self.spent_hours()

    def check_launch(self, job: Job, *, rates=None):
        """Refuse a job whose conservative estimate exceeds the remaining budget."""
        estimate = job.estimate_hours(rates=rates, conservative=True)
        if estimate > self.remaining_hours():
            raise BudgetRefused(f"{job.job_id}: conservative estimate {estimate:.2f}h exceeds "
                                f"remaining budget {self.remaining_hours():.2f}h")
        return estimate


def reprice(jobs, *, campaign: Campaign, measured_rates):
    """Re-estimate pending jobs with measured rates; drop lowest priority if over cap.

    Returns (kept, dropped, projection). Projection = spent + conservative cost
    of kept jobs. Priorities: higher number drops first; stage order breaks ties
    so earlier stages are never dropped before later ones at equal priority.
    """
    pending = [job for job in jobs]
    projection = lambda js: campaign.spent_hours() + sum(
        job.estimate_hours(rates=measured_rates, conservative=True) for job in js)
    kept = sorted(pending, key=lambda job: (job.priority, job.stage))
    dropped = []
    while kept and projection(kept) > campaign.cap_hours:
        dropped.append(kept.pop())  # highest priority number, latest stage
    kept = sorted(kept, key=lambda job: jobs.index(job))
    return kept, dropped, projection(kept)


def dry_run_report(jobs, *, campaign: Campaign, rates=None):
    """The reviewable plan: every job, its estimates, gates, and total spend."""
    lines = [f"Campaign dry run — cap {campaign.cap_hours:.1f} H100-hours",
             f"Rates: {RATE_BASIS}; conservative factor {CONSERVATIVE_FACTOR}", ""]
    total_expected = total_conservative = 0.0
    rows = []
    for job in jobs:
        expected = job.estimate_hours(rates=rates, conservative=False)
        conservative = job.estimate_hours(rates=rates, conservative=True)
        total_expected += expected
        total_conservative += conservative
        rows.append({"job_id": job.job_id, "stage": job.stage, "kind": job.kind,
                     "width": job.width, "phase": job.phase, "tokens": job.tokens,
                     "updates": job.updates, "global_batch_tokens": job.global_batch_tokens,
                     "expected_hours": round(expected, 3),
                     "conservative_hours": round(conservative, 3),
                     "parent_job": job.parent_job, "priority": job.priority})
        lines.append(f"{job.stage:8s} {job.job_id:58s} {job.tokens/1e6:8.0f}M tok "
                     f"{job.updates:6d} upd  exp {expected:6.2f}h  cons {conservative:6.2f}h")
    lines.append(f"{'':8s} {'TOTAL':58s} {'':13s}{'':9s} exp {total_expected:6.2f}h  "
                 f"cons {total_conservative:6.2f}h")
    lines.append(f"Cap check: conservative total {total_conservative:.2f}h "
                 f"{'<=' if total_conservative <= campaign.cap_hours else 'EXCEEDS'} "
                 f"{campaign.cap_hours:.1f}h")
    return {"text": "\n".join(lines), "jobs": rows,
            "total_expected_hours": total_expected,
            "total_conservative_hours": total_conservative,
            "within_cap": total_conservative <= campaign.cap_hours}


def validate_job_manifest(job: Job, manifest):
    """The manifest controls the microbatch; the config controls accumulation.

    Global batch tokens must equal microbatch x data-parallel world size x
    accumulation x sequence length, and the manifest sequence length and
    microbatch must match the job. Raises on any mismatch.
    """
    config = manifest.config
    if config.batch_size != job.microbatch:
        raise ValueError(f"{job.job_id}: manifest microbatch {config.batch_size} != job "
                         f"microbatch {job.microbatch}; remix the manifest with batch_size")
    if config.seq_len != job.seq_len:
        raise ValueError(f"{job.job_id}: manifest seq_len {config.seq_len} != job {job.seq_len}")
    expected = job.microbatch * job.world_size * job.accumulation * job.seq_len
    if job.global_batch_tokens != expected:
        raise ValueError(f"{job.job_id}: inconsistent batch decomposition")
    return {"global_batch_tokens": job.global_batch_tokens, "updates": job.updates}


def promotion_winners(ledger: Ledger, *, experiment_prefix, metric="macro_task_loss",
                      top_k=2, min_rho_note=True):
    """Rank completed attempts of a stage by the metric; return the top-k ids.

    Used between stages: only winners generate bridge jobs. Ties break by
    lower cost, then by experiment id for determinism.
    """
    candidates = [a for a in ledger.attempts()
                  if a.get("experiment_id", "").startswith(experiment_prefix)
                  and a.get("outcome") == "completed" and a.get("metrics", {}).get(metric)
                  is not None]
    candidates.sort(key=lambda a: (a["metrics"][metric],
                                   a.get("cost", {}).get("gpu_hours") or 0.0,
                                   a["experiment_id"]))
    return [a["experiment_id"] for a in candidates[:top_k]]
