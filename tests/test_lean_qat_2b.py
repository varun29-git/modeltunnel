"""The 2B QAT contract must use the verified parent and fresh source offsets."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from terminus.data import TokenDataConfig, TokenManifest, TokenShardSpec, TrainingTokenRouter
from experiments.terminus_lean_150m_qat_2b.plan import (
    MICROBATCH, PARENT_CHECKPOINT_SHA256, PARENT_TOKENS, PARENT_UPDATES,
    QAT_MIX, SEQ_LEN, TOTAL_TOKENS, render, verify,
)
from experiments.terminus_lean_150m_qat_2b.prepare_views import build_view, dense_manifest_fingerprint
from terminus.integrity import fingerprint
from experiments.terminus_lean_150m_qat_2b.finalize_blob_and_deallocate import (
    REQUIRED_FILES, validate_completed_run, verify_s3,
)
from experiments.terminus_lean_150m_qat_2b.plan import TOTAL_UPDATES, USABLE_TOKENS


class LeanQATTests(unittest.TestCase):
    def test_shutdown_s3_gate_checks_logs_checkpoints_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qat/checkpoints").mkdir(parents=True)
            (root / "qat/checkpoints/step_00000000.pt").write_bytes(b"checkpoint")
            (root / "qat/events.jsonl").write_text("training event\n")
            (root / "mirror.log").write_text("mirror complete\n")
            (root / "qat/.run.lock").write_text("not an artifact")
            calls = []

            def aws_command(args, **kwargs):
                calls.append(args)
                if args[:3] != ["aws", "s3api", "head-object"]:
                    return CompletedProcess(args, 0)
                key = args[args.index("--key") + 1]
                path = root / key.removeprefix("trial/")
                head = {"ContentLength": path.stat().st_size, "ChecksumCRC64NVME": "verified"}
                return CompletedProcess(args, 0, stdout=json.dumps(head))

            with patch("experiments.terminus_lean_150m_qat_2b.finalize_blob_and_deallocate.subprocess.run",
                       side_effect=aws_command):
                receipt = verify_s3(root, "s3://bucket/trial")
            self.assertEqual(receipt["file_count"], 3)
            self.assertEqual({item["key"] for item in receipt["files"]}, {
                "trial/qat/checkpoints/step_00000000.pt", "trial/qat/events.jsonl", "trial/mirror.log"})
            self.assertTrue((root / "s3_verified.json").is_file())
            self.assertTrue(any(args[:3] == ["aws", "s3", "cp"] and "mirror.log" in args[3]
                                for args in calls))

    def test_shutdown_s3_gate_rejects_missing_remote_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mirror.log").write_text("mirror complete\n")

            def aws_command(args, **kwargs):
                if args[:3] == ["aws", "s3api", "head-object"]:
                    return CompletedProcess(args, 0, stdout=json.dumps({
                        "ContentLength": (root / "mirror.log").stat().st_size}))
                return CompletedProcess(args, 0)

            with patch("experiments.terminus_lean_150m_qat_2b.finalize_blob_and_deallocate.subprocess.run",
                       side_effect=aws_command):
                with self.assertRaisesRegex(ValueError, "no stored checksum"):
                    verify_s3(root, "s3://bucket/trial")
            self.assertFalse((root / "s3_verified.json").exists())

    def test_model_and_optimizer_contract(self):
        config, plan = render()
        verify(config, plan)
        self.assertEqual(config["model"]["weight_mode"], "ternary")
        self.assertTrue(config["model"]["quantize_activations"])
        self.assertEqual(config["optimizer"]["muon_lr"], .005)
        self.assertEqual(config["optimizer"]["adamw_lr"], .005)
        self.assertEqual(config["optimizer"]["norm_lr"], .005)
        self.assertEqual(plan["usable_qat_tokens"], TOTAL_TOKENS // SEQ_LEN * SEQ_LEN)

    def test_fresh_view_begins_after_every_dense_source_prefix(self):
        dense_mix = {"en": .5, "code": .1, "math": .1,
                     "bn": .043, "gu": .043, "hi": .043, "kn": .042,
                     "mr": .043, "ta": .043, "te": .043}
        shards = [TokenShardSpec(f"{source}.bin", source, "train", 8_000_000_000,
                                 language=source if source not in ("code", "math") else "en")
                  for source in sorted(QAT_MIX)]
        dense = TokenManifest(TokenDataConfig(seq_len=SEQ_LEN, batch_size=MICROBATCH,
                              total_train_tokens=10_000_000_000, source_mix=dense_mix),
                              shards, {"sha256": "tokenizer"})
        legacy_payload = asdict(dense)
        del legacy_payload["config"]["source_start_examples"]
        self.assertEqual(dense_manifest_fingerprint(dense), fingerprint(legacy_payload))
        receipt = {"sha256": PARENT_CHECKPOINT_SHA256,
                   "state": {"tokens_seen": PARENT_TOKENS, "updates": PARENT_UPDATES}}
        view, result = build_view(dense, dense, receipt)
        self.assertEqual(view.config.source_mix, QAT_MIX)
        self.assertEqual(view.config.total_train_tokens, TOTAL_TOKENS)
        self.assertEqual(result["qat_usable_tokens"], TOTAL_TOKENS // SEQ_LEN * SEQ_LEN)
        dense_schedule = TrainingTokenRouter._weighted_schedule(dense_mix)
        for source in QAT_MIX:
            consumed = TrainingTokenRouter._source_count_before(
                source, PARENT_TOKENS // SEQ_LEN, dense_schedule)
            self.assertEqual(view.config.source_start_examples[source], consumed)
            self.assertEqual(result["source_start_tokens"][source], consumed * SEQ_LEN)
        self.assertEqual(sum(result["qat_source_tokens"].values()), result["qat_usable_tokens"])
        with self.assertRaisesRegex(ValueError, "verified 10B"):
            build_view(dense, dense, {**receipt, "sha256": "wrong"})

    def test_finalizer_requires_completed_ternary_run_and_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in REQUIRED_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            audit = {"phase": "final", "weight_mode": "ternary", "training_updates": TOTAL_UPDATES,
                     "tokens_seen": USABLE_TOKENS, "incidents": 0,
                     "initial_checkpoint_sha256": "initial", "final_checkpoint_sha256": "final"}
            (root / "audit_final.json").write_text(json.dumps(audit))
            (root / "qat/transition_receipt.json").write_text(json.dumps({
                "parent": {"checkpoint_sha256": PARENT_CHECKPOINT_SHA256}}))
            (root / "qat/run.json").write_text(json.dumps({"contract": {
                "model": {"weight_mode": "ternary", "quantize_activations": True}}}))
            for updates, digest in ((0, "initial"), (TOTAL_UPDATES, "final")):
                (root / f"qat/checkpoints/step_{updates:08d}.json").write_text(json.dumps({
                    "state": {"updates": updates}, "sha256": digest}))
            self.assertEqual(validate_completed_run(root), audit)
            (root / f"qat/checkpoints/step_{TOTAL_UPDATES:08d}.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_completed_run(root)


if __name__ == "__main__":
    unittest.main()
