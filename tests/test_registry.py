"""Registry, ledger, search-plan, and LR-rule tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from terminus.model import BitNetLlama, BitNetLlamaConfig
from terminus.optim import OptimizerConfig, build_optimizer, muon_update_scale_for_param
from terminus.registry import (EXPECTED_PARAM_COUNTS, Ledger, make_run_config, pareto_frontier,
                               ranking_transfer, spearman_ranks, width_model_config)
from terminus.search_plan import (stage1_width_transfer, stage3_dense_screen,
                                  stage4_qat_branches)
from tests.helpers import tiny_config


class WidthFamilyTests(unittest.TestCase):
    def test_param_counts_match_measured_baseline(self):
        for width, expected in EXPECTED_PARAM_COUNTS.items():
            model = BitNetLlama(BitNetLlamaConfig(
                **width_model_config(width, attention_backend="sdpa_reference")))
            self.assertEqual(model.count_parameters()["total"], expected, width)

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
    def test_stage1_structure(self):
        trials = stage1_width_transfer()
        launches = [t for t in trials if t["kind"] == "width_transfer"]
        reserved = [t for t in trials if t["kind"] == "reserved_bridge_bracket"]
        self.assertEqual(len(launches), 12)  # 2 widths x 2 rules x 3 LRs
        self.assertEqual(len(reserved), 2)
        ids = [t["experiment_id"] for t in trials]
        self.assertEqual(len(set(ids)), len(ids))
        for trial in launches:
            self.assertEqual(trial["config"]["model"]["weight_mode"], "dense")
            self.assertEqual(trial["config"]["training"]["schedule"], "wsd")

    def test_stage3_and_stage4_structure(self):
        stage3 = stage3_dense_screen()
        self.assertEqual(len(stage3), 6)
        stage4 = stage4_qat_branches(dense_experiment_ids=["stage3/batch_screen/200m/example"])
        branches = [t for t in stage4 if t["kind"] == "qat_branch"]
        controls = [t for t in stage4 if t["kind"].startswith("control")]
        self.assertEqual(len(branches), 9)  # 3 switch points x 3 QAT durations
        self.assertEqual({t["kind"] for t in controls},
                         {"control_dense_only", "control_ternary_from_start"})


if __name__ == "__main__":
    unittest.main()
