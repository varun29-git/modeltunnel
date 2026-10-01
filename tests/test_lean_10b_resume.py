import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/terminus_lean_150m_10b/resume_prepare.py"


class Lean10BResumeTests(unittest.TestCase):
    def test_uncommitted_records_are_archived_before_replay(self):
        spec = importlib.util.spec_from_file_location("lean_10b_resume_prepare", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/latest.json").write_text(json.dumps({"state": {"updates": 2}}))
            records = {
                "events.jsonl": [{"event": "train", "updates": 1},
                                 {"event": "train", "updates": 2},
                                 {"event": "train", "updates": 3}],
                "stability.jsonl": [{"updates": 1}, {"updates": 3}],
                "incidents.jsonl": [{"state": {"updates": 3}, "event": "gradient_failure"}],
            }
            for name, rows in records.items():
                (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = module.reconcile(root)
            self.assertEqual(result["last_committed_update"], 2)
            self.assertEqual(result["removed_active_records"], {
                "events.jsonl": 1, "stability.jsonl": 1, "incidents.jsonl": 1})
            self.assertEqual([json.loads(x)["updates"] for x in (root / "events.jsonl").read_text().splitlines()], [1, 2])
            archive = Path(result["archive"])
            self.assertEqual(len((archive / "events.jsonl").read_text().splitlines()), 3)
            self.assertEqual(module.reconcile(root)["removed_active_records"], {})


if __name__ == "__main__":
    unittest.main()
