"""The shutdown gate must never run after an incomplete or unverified copy."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (Path(__file__).resolve().parents[1] / "experiments" /
               "terminus_lean_150m_10b" / "finalize_blob_and_deallocate.py")
spec = importlib.util.spec_from_file_location("lean_blob_finalizer", MODULE_PATH)
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def completed_run(root: Path) -> None:
    for name in finalizer.REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete run artifact\n")
    checkpoints = []
    for updates, tokens in ((finalizer.MIDPOINT_UPDATES, finalizer.MIDPOINT_TOKENS),
                            (finalizer.FINAL_UPDATES, finalizer.FINAL_TOKENS)):
        stem = f"step_{updates:08d}"
        receipt = {"file": f"{stem}.pt", "sha256": stem,
                   "state": {"updates": updates, "tokens_seen": tokens}}
        (root / "dense/checkpoints" / f"{stem}.json").write_text(json.dumps(receipt))
        checkpoints.append({"updates": updates, "tokens_seen": tokens, "sha256": stem})
    (root / "audit_final.json").write_text(json.dumps({
        "phase": "final", "training_updates": finalizer.FINAL_UPDATES,
        "checkpoints": checkpoints,
    }))


class BlobFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        completed_run(self.root)
        self.url = "https://account.blob.core.windows.net/runs/terminus-run"
        self.vm_id = "/subscriptions/example/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm"

    def test_active_training_never_uploads_or_deallocates(self):
        with (mock.patch.object(finalizer, "service_completed", return_value=False),
              mock.patch.object(finalizer, "azcopy") as upload,
              mock.patch.object(finalizer, "request_deallocation") as deallocate):
            with self.assertRaisesRegex(ValueError, "has not completed"):
                finalizer.finalize(self.root, self.url, self.vm_id, "train.service")
        upload.assert_not_called()
        deallocate.assert_not_called()

    def test_missing_final_log_blocks_shutdown(self):
        (self.root / "phase2.log").unlink()
        with self.assertRaisesRegex(ValueError, "phase2.log"):
            finalizer.validate_completed_run(self.root)

    def test_blob_verification_failure_blocks_shutdown(self):
        with (mock.patch.object(finalizer, "service_completed", return_value=True),
              mock.patch.object(finalizer, "azcopy"),
              mock.patch.object(finalizer, "managed_identity_token", return_value="token"),
              mock.patch.object(finalizer, "verify_blob", side_effect=ValueError("MD5 mismatch")),
              mock.patch.object(finalizer, "request_deallocation") as deallocate):
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                finalizer.finalize(self.root, self.url, self.vm_id, "train.service")
        deallocate.assert_not_called()

    def test_shutdown_only_after_full_verification_and_receipt(self):
        ops_temp = tempfile.TemporaryDirectory()
        self.addCleanup(ops_temp.cleanup)
        ops_log = Path(ops_temp.name) / "finalizer.log"
        ops_log.write_text("verification started\n")
        with (mock.patch.object(finalizer, "service_completed", return_value=True),
              mock.patch.object(finalizer, "azcopy") as upload,
              mock.patch.object(finalizer, "managed_identity_token", return_value="token"),
              mock.patch.object(finalizer, "verify_blob") as verify,
              mock.patch.object(finalizer, "request_deallocation", return_value=202) as deallocate):
            finalizer.finalize(self.root, self.url, self.vm_id, "train.service", ops_log)
        self.assertEqual(upload.call_count, 3)
        self.assertGreater(verify.call_count, len(finalizer.REQUIRED_FILES))
        receipt = json.loads((self.root / "blob_verified.json").read_text())
        self.assertEqual(receipt["result"], "verified")
        deallocate.assert_called_once_with(self.vm_id)

    def test_missing_finalizer_log_blocks_shutdown(self):
        with (mock.patch.object(finalizer, "service_completed", return_value=True),
              mock.patch.object(finalizer, "azcopy"),
              mock.patch.object(finalizer, "managed_identity_token", return_value="token"),
              mock.patch.object(finalizer, "verify_blob"),
              mock.patch.object(finalizer, "request_deallocation") as deallocate):
            with self.assertRaisesRegex(ValueError, "finalizer log is missing"):
                finalizer.finalize(self.root, self.url, self.vm_id, "train.service",
                                   self.root.parent / "missing-finalizer.log")
        deallocate.assert_not_called()

    def test_deallocation_waits_for_azure_operation_success(self):
        class Response:
            def __init__(self, status, body=b"{}", headers=None):
                self.status, self.body, self.headers = status, body, headers or {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, *_):
                return self.body

        responses = [Response(202, headers={"Azure-AsyncOperation": "https://management.azure.com/op"}),
                     Response(200, body=b'{"status":"Succeeded"}')]
        with (mock.patch.object(finalizer, "managed_identity_token", return_value="token"),
              mock.patch.object(finalizer.urllib.request, "urlopen", side_effect=responses) as open_url,
              mock.patch.object(finalizer.time, "sleep")):
            self.assertEqual(finalizer.request_deallocation(self.vm_id), 202)
        self.assertEqual(open_url.call_count, 2)


if __name__ == "__main__":
    unittest.main()
