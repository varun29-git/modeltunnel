"""Registry, ledger, search-plan, and LR-rule tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from terminus.model import BitNetLlama, BitNetLlamaConfig
from terminus.optim import OptimizerConfig, build_optimizer, muon_update_scale_for_param
from terminus.registry import (EXPECTED_PARAM_COUNTS, Ledger, make_run_config, pareto_frontier,
                               ranking_transfer, spearman_ranks, width_model_config)
from tests.helpers import tiny_config


class WidthFamilyTests(unittest.TestCase):
    def test_param_counts_match_measured_baseline(self):
        for width, expected in EXPECTED_PARAM_COUNTS.items():
            model = BitNetLlama(BitNetLlamaConfig(
                **width_model_config(width, attention_backend="sdpa_reference")))
            self.assertEqual(model.count_parameters()["total"], expected, width)

    def test_new_campaigns_use_learned_qk_norm_in_both_phases(self):
        for mode in ("dense", "ternary"):
            config = make_run_config("200m", weight_mode=mode, schedule="wsd",
                                     warmup_tokens=100, decay_tokens=200,
                                     muon_lr=.003, adamw_lr=.003)
            self.assertEqual(config["model"]["qk_norm_mode"], "learned_rms")
            self.assertFalse(config["model"]["fused_qkv"])
            self.assertEqual(config["optimizer"]["qk_norm_lr"], .0001)
            self.assertLess(config["optimizer"]["qk_norm_lr"], config["optimizer"]["norm_lr"])

    def test_muon_lr_rules(self):
        torch = __import__("torch")
        parameter = torch.nn.Parameter(torch.empty(256, 1024))
        rule_a = muon_update_scale_for_param(parameter, rule="sqrt_fan_in")
        rule_b = muon_update_scale_for_param(parameter, rule="sqrt_fan_in_ratio",
                                             fan_in_base=2304)
        self.assertAlmostEqual(rule_a, 0.2 * 1024 ** 0.5)
        self.assertAlmostEqual(rule_b, rule_a * 2304 / 1024)
        with self.assertRaises(ValueError):
            muon_update_scale_for_param(parameter, rule="sqrt_fan_in_ratio")
        with self.assertRaises(ValueError):
            OptimizerConfig(muon_lr_rule="sqrt_fan_in_ratio")  # missing base
        with self.assertRaises(ValueError):
            OptimizerConfig(muon_fan_in_base=2304)  # base without the rule

    def test_rule_b_changes_group_lrs(self):
        model = BitNetLlama(tiny_config())
        a = build_optimizer(model, OptimizerConfig(kind="muon"))
        b = build_optimizer(model, OptimizerConfig(kind="muon", muon_lr_rule="sqrt_fan_in_ratio",
                                                   muon_fan_in_base=2304))
        muon_a = [g["lr"] for g in a.param_groups if g.get("use_muon")]
        muon_b = [g["lr"] for g in b.param_groups if g.get("use_muon")]
        self.assertEqual(len(muon_a), len(muon_b))
        self.assertTrue(all(x != y for x, y in zip(muon_a, muon_b)))


class LedgerTests(unittest.TestCase):
    def _spec(self, name="exp"):
        config = make_run_config("200m", weight_mode="dense", schedule="wsd",
                                 warmup_tokens=100, decay_tokens=200, muon_lr=0.003,
                                 adamw_lr=0.003)
        return {"experiment_id": name, "stage": "stage3", "kind": "batch_screen",
                "config": config, "data_manifest_sha256": "d" * 64,
                "tokenizer_sha256": "t" * 64}

    def test_begin_finish_and_retry(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(Path(root) / "ledger.jsonl")
            first = ledger.begin_attempt(self._spec())
            ledger.finish_attempt(first, outcome="infrastructure",
                                  failure_detail="spot instance revoked")
            second = ledger.begin_attempt({**self._spec(), "retry_of": first})
            ledger.finish_attempt(second, outcome="completed",
                                  metrics={"task_loss": 1.23}, cost={"gpu_hours": 4.5},
                                  realized={"tokens_seen": 10_000})
            attempts = {a["attempt_id"]: a for a in ledger.attempts()}
            self.assertEqual(attempts[first]["outcome"], "infrastructure")
            self.assertEqual(attempts[second]["outcome"], "completed")
            self.assertEqual(attempts[second]["retry_of"], first)
            self.assertEqual(attempts[second]["metrics"]["task_loss"], 1.23)
            self.assertEqual(len(attempts[second]["source"]) > 0, True)
            # failed attempts cannot carry quality metrics; finished attempts are sealed
            third = ledger.begin_attempt(self._spec("exp2"))
            with self.assertRaises(ValueError):
                ledger.finish_attempt(third, outcome="oom", metrics={"task_loss": 1.0})
            with self.assertRaises(ValueError):
                ledger.finish_attempt(second, outcome="completed")
            with self.assertRaises(ValueError):
                ledger.finish_attempt("nonexistent", outcome="completed")
            with self.assertRaises(ValueError):
                ledger.finish_attempt(third, outcome="exploded")
            # reload from disk
            self.assertEqual(len(Ledger(Path(root) / "ledger.jsonl").records()), 5)

    def test_pareto_frontier_and_transfer(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(Path(root) / "ledger.jsonl")
            ids = []
            for loss, hours in ((1.0, 10.0), (1.1, 4.0), (1.3, 12.0), (0.9, 20.0), (1.2, 5.0)):
                attempt = ledger.begin_attempt(self._spec())
                ledger.finish_attempt(attempt, outcome="completed",
                                      metrics={"task_loss": loss}, cost={"gpu_hours": hours})
                ids.append(attempt)
            frontier = pareto_frontier(ledger.attempts(), quality="task_loss")
            # (1.3,12) is dominated by (1.0,10); (1.2,5) by (1.1,4)
            self.assertEqual(frontier, sorted([ids[0], ids[1], ids[3]]))
        self.assertAlmostEqual(spearman_ranks([1, 2, 3], [3, 2, 1]), -1.0)
        report = ranking_transfer([1.0, 1.1, 1.2, 0.9], [2.0, 2.2, 2.4, 1.8])
        self.assertTrue(report["transfers"])
        report = ranking_transfer([1.0, 1.1, 1.2, 0.9], [1.8, 2.4, 2.2, 2.0])
        self.assertFalse(report["transfers"])
        with self.assertRaises(ValueError):
            spearman_ranks([1, 1, 1], [1, 2, 3])


class SearchPlanTests(unittest.TestCase):
    def test_staged_plan_structure(self):
        from terminus.search_plan import campaign_plan, matched_exposure_check
        jobs = campaign_plan()
        by_stage = {}
        for job in jobs:
            by_stage.setdefault(job.stage, []).append(job)
        self.assertEqual(len(by_stage["stage0"]), 2)
        self.assertEqual(len(by_stage["stage1"]), 6)   # 2 rules x 3 LRs
        self.assertEqual(len(by_stage["stage2"]), 3)   # 2 bridge + 1 short 1.5B check
        self.assertEqual(len(by_stage["stage3"]), 4)   # real mixture overlays
        self.assertEqual(len(by_stage["stage4"]), 7)   # parent + 4 branches + 2 controls
        self.assertEqual(len(by_stage["stage5"]), 2)   # conditional second seeds
        self.assertEqual(matched_exposure_check(jobs), [])
        ids = [job.job_id for job in jobs]
        self.assertEqual(len(set(ids)), len(ids))

    def test_matched_exposure_check_catches_short_control(self):
        from terminus.search_plan import campaign_plan, matched_exposure_check
        jobs = campaign_plan()
        control = [j for j in jobs if j.kind == "control_ternary_from_start"][0]
        control.tokens //= 4
        self.assertTrue(matched_exposure_check(jobs))

    def test_mixtures_are_real_and_exact(self):
        from terminus.search_plan import MIXTURES
        from terminus.integrity import CAMPAIGN_LANGUAGES
        for name, mix in MIXTURES.items():
            self.assertAlmostEqual(sum(mix.values()), 1.0, places=12)
            self.assertEqual(set(mix), set(CAMPAIGN_LANGUAGES) | {"math", "code"})
            for weight in mix.values():
                self.assertAlmostEqual(weight * 1000, round(weight * 1000), places=9)

    def test_batch_decomposition_relation(self):
        from terminus.search_plan import campaign_plan
        for job in campaign_plan():
            self.assertEqual(job.global_batch_tokens,
                             job.microbatch * job.world_size * job.accumulation * job.seq_len)

    def test_embedding_fraction_blocks_small_proxies(self):
        from terminus.registry import embedding_fraction_report
        report = embedding_fraction_report(widths=("200m",))
        self.assertAlmostEqual(report["200m"]["embedding_fraction"], 0.2513, places=3)
        self.assertGreater(report["100m_hypothetical"]["embedding_fraction"], 0.32)
        self.assertGreater(report["50m_hypothetical"]["embedding_fraction"], 0.40)


if __name__ == "__main__":
    unittest.main()
