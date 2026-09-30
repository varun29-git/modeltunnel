import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from terminus.integrity import file_sha256


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/terminus_lean_150m_3b/audit.py"


class LeanAuditTests(unittest.TestCase):
    def test_checkpoint_integrity_record_continuity_and_alerts(self):
        import sys

        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location("lean_run_audit", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "run.json").write_text(json.dumps({"contract": {"model": {"weight_mode": "dense"}}}))
            for update, tokens in ((1, 32), (2, 64)):
                checkpoint = root / "checkpoints" / f"step_{update:08d}.pt"
                checkpoint.write_bytes(f"checkpoint-{update}".encode())
                receipt = {"file": checkpoint.name, "sha256": file_sha256(checkpoint),
                           "size_bytes": checkpoint.stat().st_size,
                           "state": {"updates": update, "tokens_seen": tokens}}
                checkpoint.with_suffix(".json").write_text(json.dumps(receipt))
            events = [
                {"event": "train", "updates": 1, "loss": 3.0, "grad_clip_coefficient": 0.05},
                {"event": "validation", "updates": 1},
                {"event": "train", "updates": 2, "loss": 2.0, "grad_clip_coefficient": 1.0},
                {"event": "validation", "updates": 2},
            ]
            (root / "events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
            probes = [{"updates": 1, "probes": {"layer_00.attention_sample": {
                          "logits": {"max_abs": 40.0}, "min_entropy": 1.0}}},
                      {"updates": 2, "probes": {"layers.0.weight.parameter_sample": {
                          "update_to_weight_rms": 0.2}}}]
            (root / "stability.jsonl").write_text("".join(json.dumps(item) + "\n" for item in probes))
            with patch.multiple(module, MIDPOINT_UPDATES=1, MIDPOINT_TOKENS=32,
                                TOTAL_UPDATES=2, USABLE_TOTAL_TOKENS=64):
                report = module.audit(root, phase="final")
                self.assertEqual(report["training_updates"], 2)
                self.assertEqual(report["clip_coefficient_lt_0.1_updates"], [1])
                self.assertEqual(len(report["probe_alerts"]), 2)
                checkpoint = root / "checkpoints/step_00000001.pt"
                checkpoint.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    module.audit(root, phase="midpoint")


if __name__ == "__main__":
    unittest.main()
