"""Campaign runner tests: budget, repricing, promotion, batch control, scoring."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from terminus.campaign import (ASSUMED_TOKENS_PER_SECOND, CONSERVATIVE_FACTOR, BudgetRefused,
                               Campaign, Job, dry_run_report, promotion_winners, reprice)
from terminus.registry import Ledger
from terminus.search_plan import campaign_plan
from terminus.task_eval import TaskItem, score_items
from tests.helpers import checkpoint_payload, fixture, tiny_config


def make_ledger(root):
    return Ledger(Path(root) / "ledger.jsonl")


def finish(ledger, job_id, hours, outcome="completed", metric=None):
    spec = {"experiment_id": job_id, "stage": "stageX", "kind": "k",
            "config": {"model": {}, "training": {"seed": 1}, "optimizer": {}}}
    attempt = ledger.begin_attempt(spec)
    ledger.finish_attempt(attempt, outcome=outcome,
                          metrics={"macro_task_loss": metric} if metric is not None else None,
                          failure_detail="x" if outcome != "completed" else None,
                          cost={"gpu_hours": hours})
    return attempt


class BudgetTests(unittest.TestCase):
    def test_refuses_job_exceeding_remaining_budget(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = make_ledger(root)
            campaign = Campaign(2.0, ledger)
            job = Job(job_id="big", stage="stage1", kind="width_transfer", width="200m",
                      phase="dense", tokens=600_000_000)  # ~1.9h conservative
            campaign.check_launch(job)  # fits in 2h
            finish(ledger, "prior", 1.5)  # spend down
            with self.assertRaises(BudgetRefused):
                campaign.check_launch(job)

    def test_failed_and_interrupted_jobs_count_toward_the_cap(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = make_ledger(root)
            campaign = Campaign(10.0, ledger)
            finish(ledger, "failed-job", 2.0, outcome="infrastructure")
            finish(ledger, "interrupted-job", 1.0, outcome="interrupted")
            self.assertEqual(campaign.spent_hours(), 3.0)
            self.assertEqual(campaign.remaining_hours(), 7.0)

    def test_stage0_preflight_estimate(self):
        job = Job(job_id="p", stage="stage0", kind="preflight_throughput", width="1500m",
                  phase="dense")
        self.assertGreater(job.estimate_hours(), 0.5)
        self.assertEqual(job.estimate_hours(), job.estimate_hours(rates={}))


class RepricingTests(unittest.TestCase):
    def test_slower_measured_rates_drop_low_priority_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = make_ledger(root)
            campaign = Campaign(5.0, ledger)
            finish(ledger, "stage0", 1.0)
            jobs = [Job(job_id=f"j{i}", stage="stage1", kind="width_transfer", width="200m",
                        phase="dense", tokens=100_000_000, priority=i) for i in range(6)]
            fast = {("200m", "dense"): 200_000}   # faster than assumed: all fit
            kept, dropped, projection = reprice(jobs, campaign=campaign, measured_rates=fast)
            self.assertEqual(len(kept), 6)
            self.assertLessEqual(projection, campaign.cap_hours)
            slow = {("200m", "dense"): 20_000}    # 7.5x slower: must drop
            kept, dropped, projection = reprice(jobs, campaign=campaign, measured_rates=slow)
            self.assertTrue(dropped)
            self.assertLessEqual(projection, campaign.cap_hours)
            self.assertEqual({j.job_id for j in dropped}, {"j5", "j4", "j3", "j2", "j1"})
            self.assertEqual([j.job_id for j in kept], ["j0"])  # lowest priority number survives

    def test_never_silently_exceeds_cap(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = make_ledger(root)
            campaign = Campaign(0.5, ledger)
            jobs = [Job(job_id="j", stage="stage1", kind="k", width="200m", phase="dense",
                        tokens=500_000_000)]  # ~1.6h conservative > 0.5h cap
            kept, dropped, projection = reprice(jobs, campaign=campaign, measured_rates={})
            self.assertEqual(kept, [])
            self.assertLessEqual(projection, campaign.cap_hours)


class PromotionTests(unittest.TestCase):
    def test_winners_by_task_loss_with_cost_tiebreak(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = make_ledger(root)
            finish(ledger, "stage1/a", 1.0, metric=2.0)
            finish(ledger, "stage1/b", 2.0, metric=1.5)
            finish(ledger, "stage1/c", 0.5, metric=1.5)   # same loss, cheaper -> wins tie
            finish(ledger, "stage1/failed", 3.0, outcome="nonfinite_loss")
            self.assertEqual(promotion_winners(ledger, experiment_prefix="stage1/", top_k=2),
                             ["stage1/c", "stage1/b"])


class BatchControlTests(unittest.TestCase):
    def test_remix_batch_size_changes_tokens_per_step(self):
        from terminus.remix import remix_manifest
        with tempfile.TemporaryDirectory() as root:
            config, manifest = fixture(Path(root) / "dataset")
            data_root = Path(root) / "dataset"
            model_config = tiny_config(max_seq_len=8)
            out = remix_manifest(data_root / "manifest.json", data_root, model_config,
                                 batch_size=4, name="bs4", development_mode=True)
            remixed = out["manifest_sha256"]
            from terminus.data import TokenManifest
            remixed_manifest = TokenManifest.load(out["path"])
            self.assertEqual(remixed_manifest.config.batch_size, 4)
            self.assertEqual(remixed_manifest.config.tokens_per_step, 4 * 8)
            self.assertEqual(manifest.config.tokens_per_step, 2 * 8)
            self.assertNotEqual(remixed, out.get("original"))


class DryRunTests(unittest.TestCase):
    def test_dry_run_lists_all_jobs_within_cap(self):
        with tempfile.TemporaryDirectory() as root:
            campaign = Campaign(60.0, make_ledger(root))
            jobs = campaign_plan()
            report = dry_run_report(jobs, campaign=campaign)
            self.assertTrue(report["within_cap"])
            self.assertGreaterEqual(report["total_conservative_hours"], 50.0)
            self.assertLessEqual(report["total_conservative_hours"], 60.0)
            self.assertEqual(len(report["jobs"]), len(jobs))
            self.assertIn("stage4/200m/control_ternary_from_start", report["text"])
            self.assertIn("TOTAL", report["text"])


class CheckpointScoringTests(unittest.TestCase):
    def test_trained_checkpoint_scores_on_cpu(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from evaluate_run import load_verified_model
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            root = Path(root)
            fixture(root / "dataset")
            from terminus.train import run_training
            run_training(root / "dataset/config.json", root / "dataset/manifest.json",
                         root / "dataset", root / "run")
            model, model_config, state, receipt = load_verified_model(
                root / "dataset/config.json", root / "run/checkpoints")
            self.assertEqual(state["updates"], 3)  # 9 examples, bs 2, accum 2: 2 full + 1 partial
            tokenizer = type("T", (), {"encode": lambda self, t, out_type=int:
                                       [ord(c) % 30 + 1 for c in t]})()
            items = [TaskItem(task="toy", item_id="1", prompt="ab", reference="cd")]
            records = score_items(model, tokenizer, items, device=torch.device("cpu"),
                                  max_seq_len=8)
            self.assertEqual(len(records), 1)
            self.assertGreater(records[0]["loss"], 0)
            self.assertEqual(receipt["file"], "step_00000003.pt")


if __name__ == "__main__":
    unittest.main()


class RunnablePlanTests(unittest.TestCase):
    def test_every_plan_job_config_validates(self):
        from terminus.model import BitNetLlamaConfig
        from terminus.optim import OptimizerConfig
        from terminus.search_plan import job_config
        from terminus.train import TrainConfig
        for job in campaign_plan():
            config = job_config(job)
            if job.stage == "stage0":
                self.assertIsNone(config)
                continue
            model = BitNetLlamaConfig(**config["model"])
            training = TrainConfig(**config["training"])
            OptimizerConfig(**config["optimizer"])
            if job.phase == "qat":
                self.assertEqual(model.weight_mode, "ternary")
                # fully decayed endpoint: re-warmup + decay fit inside the budget
                self.assertLess(training.warmup_tokens + training.decay_tokens, job.tokens)
            if job.phase == "dense":
                self.assertEqual(model.weight_mode, "dense")

    def test_job_manifest_binding(self):
        from terminus.campaign import validate_job_manifest
        from terminus.data import TokenManifest
        with tempfile.TemporaryDirectory() as root:
            config, manifest = fixture(Path(root) / "dataset")  # batch_size 2, seq 8
            job = Job(job_id="j", stage="stage1", kind="k", width="200m", phase="dense",
                      tokens=1000, microbatch=2, accumulation=4, seq_len=8)
            report = validate_job_manifest(job, manifest)
            self.assertEqual(report["global_batch_tokens"], 2 * 1 * 4 * 8)
            bad = Job(job_id="bad", stage="stage1", kind="k", width="200m", phase="dense",
                      tokens=1000, microbatch=4, accumulation=4, seq_len=8)
            with self.assertRaisesRegex(ValueError, "microbatch"):
                validate_job_manifest(bad, manifest)
