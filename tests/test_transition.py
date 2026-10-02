"""Dense -> ternary QAT phase transition tests (CPU, development mode)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from terminus.optim import OptimizerConfig
from terminus.train import TrainConfig, run_training
from terminus.transition import convert_checkpoint
from tests.helpers import checkpoint_payload, fixture, tiny_config


def write_config(path, *, weight_mode, schedule="cosine", warmup_tokens=0, decay_tokens=None,
                 muon_lr=0.003):
    config = tiny_config(max_seq_len=8, weight_mode=weight_mode)
    training = TrainConfig(device="cpu", dtype="fp32", development_mode=True,
                           gradient_accumulation_steps=2, log_interval=1, validation_interval=1000,
                           checkpoint_interval=1000, validation_batches_per_language=2,
                           schedule=schedule, warmup_tokens=warmup_tokens,
                           decay_tokens=decay_tokens)
    optimizer = OptimizerConfig(muon_lr=muon_lr)
    Path(path).write_text(json.dumps({"model": asdict(config), "training": asdict(training),
                                      "optimizer": asdict(optimizer)}))


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root / "dataset")  # 72 usable tokens, 18 micro steps, bs 2, accum 2
        self.manifest = self.root / "dataset/manifest.json"
        self.data_root = self.root / "dataset"
        write_config(self.root / "dense.json", weight_mode="dense")
        write_config(self.root / "qat.json", weight_mode="ternary", schedule="wsd",
                     warmup_tokens=8, decay_tokens=16, muon_lr=0.001)

    def tearDown(self):
        self.tmp.cleanup()

    def train_dense(self, updates=2):
        return run_training(self.root / "dense.json", self.manifest, self.data_root,
                            self.root / "dense_run", stop_after_updates=updates)

    def convert(self, out="qat_run", **kwargs):
        return convert_checkpoint(self.root / "dense_run", self.root / "qat.json",
                                  self.manifest, self.data_root, self.root / out, **kwargs)

    def test_transition_carries_state_and_completes_without_data_repeat(self):
        with contextlib.redirect_stdout(io.StringIO()):
            parent = self.train_dense(updates=1)  # 32 of 72 tokens, micro step 2
            self.assertEqual(parent["tokens_seen"], 32)
            receipt = self.convert()
            self.assertEqual(receipt["parent"]["state"]["tokens_seen"], 32)
            self.assertEqual(receipt["child"]["state"]["tokens_seen"], 32)
            self.assertEqual(receipt["data_policy"], "continue")
            self.assertEqual(receipt["intentional_resets"], [])
            self.assertEqual(receipt["changed_fields"]["model"]["weight_mode"],
                             {"parent": "dense", "child": "ternary"})
            # Weights carried over exactly.
            parent_payload = checkpoint_payload(self.root / "dense_run")
            child_payload = checkpoint_payload(self.root / "qat_run")
            for name, tensor in parent_payload["model"].items():
                torch.testing.assert_close(child_payload["model"][name], tensor, rtol=0, atol=0)
            # Finish the QAT phase; train.py's per-step cursor check proves the
            # stream continues exactly (no repeat, no gap).
            final = run_training(self.root / "qat_run/qat_config.json", self.manifest, self.data_root,
                                 self.root / "qat_run", resume=self.root / "qat_run/checkpoints")
            self.assertEqual(final["tokens_seen"], 72)
            self.assertEqual(final["updates"], 3)  # two QAT updates, the last one partial
            # phase-relative LR re-warmup applied: the first QAT update ran below peak scale
            events = [json.loads(line) for line in
                      (self.root / "qat_run/events.jsonl").read_text().splitlines()]
            scales = [e["lr_scale"] for e in events if e.get("event") == "train"]
            # the QAT schedule restarted at the phase boundary: the dense cosine
            # would give ~0.13 here, the fresh QAT window gives > 0.5, and the
            # QAT decay ends at the configured final fraction
            self.assertGreater(scales[0], 0.5)
            self.assertAlmostEqual(scales[-1], 0.1, places=6)

    def test_transition_matches_within_qat_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.train_dense(updates=1)  # 32 of 72 tokens consumed
            self.convert(out="qat_a")
            complete = run_training(self.root / "qat_a/qat_config.json", self.manifest, self.data_root,
                                    self.root / "qat_a", resume=self.root / "qat_a/checkpoints")
            self.convert(out="qat_b")
            # interrupt after the first QAT update, then resume exactly
            stopped = run_training(self.root / "qat_b/qat_config.json", self.manifest, self.data_root,
                                   self.root / "qat_b", resume=self.root / "qat_b/checkpoints",
                                   stop_after_updates=2)
            self.assertEqual(stopped["updates"], 2)
            self.assertLess(stopped["tokens_seen"], 72)
            resumed = run_training(self.root / "qat_b/qat_config.json", self.manifest, self.data_root,
                                   self.root / "qat_b", resume=self.root / "qat_b/checkpoints")
            self.assertEqual(complete, resumed)
            expected, actual = checkpoint_payload(self.root / "qat_a"), checkpoint_payload(self.root / "qat_b")
            for name, tensor in expected["model"].items():
                torch.testing.assert_close(actual["model"][name], tensor, rtol=0, atol=0)

    def test_dense_history_changes_final_weights(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.train_dense(updates=1)
            self.convert(out="qat_a")
            final_a = run_training(self.root / "qat_a/qat_config.json", self.manifest, self.data_root,
                                   self.root / "qat_a", resume=self.root / "qat_a/checkpoints")
            pure = run_training(self.root / "qat.json", self.manifest, self.data_root,
                                self.root / "qat_pure")
            payload_a, payload_pure = checkpoint_payload(self.root / "qat_a"), checkpoint_payload(self.root / "qat_pure")
            self.assertEqual(final_a["tokens_seen"], pure["tokens_seen"])
            differences = [float((payload_a["model"][name] - payload_pure["model"][name]).abs().max())
                           for name in payload_a["model"]]
            self.assertGreater(max(differences), 0.0)

    def test_restarts_and_rejections(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.train_dense(updates=2)
            # restart policy zeroes the data cursor and records it
            receipt = self.convert(out="qat_restart", data_policy="restart")
            self.assertEqual(receipt["child"]["state"]["tokens_seen"], 0)
            self.assertIn("data_cursor", receipt["intentional_resets"])
            self.assertIn("update_counter", receipt["intentional_resets"])
            self.assertEqual(receipt["child"]["state"]["updates"], 0)
            restarted = run_training(self.root / "qat_restart/qat_config.json", self.manifest,
                                     self.data_root, self.root / "qat_restart",
                                     resume=self.root / "qat_restart/checkpoints")
            self.assertEqual(restarted["tokens_seen"], 72)
            # non-ternary child config is refused
            write_config(self.root / "dense2.json", weight_mode="dense")
            with self.assertRaisesRegex(ValueError, "ternary"):
                convert_checkpoint(self.root / "dense_run", self.root / "dense2.json",
                                   self.manifest, self.data_root, self.root / "qat_bad1")
            # architecture mismatch is refused by strict state_dict loading
            config = json.loads((self.root / "qat.json").read_text())
            config["model"]["dim"] = 64
            (self.root / "qat_wide.json").write_text(json.dumps(config))
            with self.assertRaises(Exception):
                convert_checkpoint(self.root / "dense_run", self.root / "qat_wide.json",
                                   self.manifest, self.data_root, self.root / "qat_bad2")
            # changed accumulation without a cursor reset is refused
            config = json.loads((self.root / "qat.json").read_text())
            config["training"]["gradient_accumulation_steps"] = 3
            (self.root / "qat_accum.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                convert_checkpoint(self.root / "dense_run", self.root / "qat_accum.json",
                                   self.manifest, self.data_root, self.root / "qat_bad3")
            # fresh-directory protection
            with self.assertRaises(FileExistsError):
                self.convert(out="qat_restart")

    def test_transition_keeps_moments_but_uses_child_base_learning_rates(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.train_dense(updates=1)
            dense = checkpoint_payload(self.root / "dense_run")
            config = json.loads((self.root / "qat.json").read_text())
            config["optimizer"]["adamw_lr"] /= 2
            config["optimizer"]["norm_lr"] /= 2
            (self.root / "qat_half_lr.json").write_text(json.dumps(config))
            receipt = convert_checkpoint(self.root / "dense_run", self.root / "qat_half_lr.json",
                                         self.manifest, self.data_root, self.root / "qat_half_lr")
            child = checkpoint_payload(self.root / "qat_half_lr")
            self.assertEqual(receipt["parent"]["state"]["tokens_seen"], 32)
            for key, old_state in dense["optimizer"]["state"].items():
                for name, old_value in old_state.items():
                    new_value = child["optimizer"]["state"][key][name]
                    if isinstance(old_value, torch.Tensor):
                        torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)
                    else:
                        self.assertEqual(new_value, old_value)
            old_rates = [group["base_lr"] for group in dense["optimizer"]["param_groups"]]
            new_rates = [group["base_lr"] for group in child["optimizer"]["param_groups"]]
            self.assertEqual(new_rates, [rate / 2 for rate in old_rates])

    def test_parent_integrity_is_verified(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.train_dense(updates=2)
            latest = self.root / "dense_run/checkpoints"
            receipt = json.loads((latest / "latest.json").read_text())
            blob = bytearray((latest / receipt["file"]).read_bytes())
            blob[-1] ^= 0xFF
            (latest / receipt["file"]).write_bytes(bytes(blob))
            with self.assertRaises(ValueError):
                self.convert(out="qat_corrupt")


if __name__ == "__main__":
    unittest.main()
