"""Deallocate the H100 only after the completed run is verified in Azure Blob.

The training service owns the run directory until it exits successfully. This
independent service then copies every regular run file to Blob with Content-MD5,
checks each remote size and MD5 through Blob HEAD, writes a verification receipt,
and requests deallocation using the VM's managed identity. Any failed gate leaves
the VM running; systemd retries the finalizer later.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FINAL_UPDATES = 25432
FINAL_TOKENS = 9_999_998_976
MIDPOINT_UPDATES = 12716
MIDPOINT_TOKENS = 5_000_134_656
BLOB_API_VERSION = "2023-11-03"
ARM_API_VERSION = "2024-07-01"
GENERATED_FILES = {
    "blob_inventory.json", "blob_verified.json",
    ".blob_inventory.tmp", ".blob_verified.tmp",
}
REQUIRED_FILES = {
    "audit_midpoint.json", "audit_midpoint.log",
    "audit_final.json", "audit_final.log",
    "phase1.log", "phase2.log", "service.log", "mirror.log",
    "dense/events.jsonl", "dense/stability.jsonl", "dense/run.json",
    "dense/checkpoints/step_00012716.pt",
    "dense/checkpoints/step_00012716.json",
    "dense/checkpoints/step_00025432.pt",
    "dense/checkpoints/step_00025432.json",
}


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def service_completed(service: str) -> bool:
    result = subprocess.run(
        ["systemctl", "show", service, "-p", "ActiveState", "-p", "SubState",
         "-p", "Result", "-p", "ExecMainStatus"],
        check=True, capture_output=True, text=True,
    )
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return (state.get("ActiveState") == "inactive" and state.get("SubState") == "dead"
            and state.get("Result") == "success" and state.get("ExecMainStatus") == "0")


def validate_completed_run(root: Path) -> dict:
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"required final run files are missing: {missing}")
    audit = json.loads((root / "audit_final.json").read_text())
    if audit.get("phase") != "final" or audit.get("training_updates") != FINAL_UPDATES:
        raise ValueError("final audit does not cover all planned updates")
    audited = {item["updates"]: item for item in audit.get("checkpoints", [])}
    for updates, tokens in ((MIDPOINT_UPDATES, MIDPOINT_TOKENS),
                            (FINAL_UPDATES, FINAL_TOKENS)):
        if updates not in audited or audited[updates].get("tokens_seen") != tokens:
            raise ValueError(f"final audit does not cover checkpoint {updates}")
        stem = f"step_{updates:08d}"
        receipt = json.loads((root / "dense/checkpoints" / f"{stem}.json").read_text())
        state = receipt.get("state", {})
        if (state.get("updates") != updates or state.get("tokens_seen") != tokens
                or receipt.get("sha256") != audited[updates].get("sha256")):
            raise ValueError(f"checkpoint receipt disagrees with final audit: {stem}")
        if receipt.get("file") != f"{stem}.pt":
            raise ValueError(f"unexpected checkpoint path: {stem}")
    return audit


def run_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in run directory cannot be mirrored safely: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in GENERATED_FILES:
            files.append(path)
    return sorted(files)


def file_record(root: Path, path: Path) -> dict:
    before = path.stat()
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"run file changed while hashing: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": sha256.hexdigest(),
        "md5_base64": base64.b64encode(md5.digest()).decode("ascii"),
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name("." + path.stem + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def azcopy(*args: str) -> None:
    env = dict(os.environ, AZCOPY_AUTO_LOGIN_TYPE="MSI")
    command = ["/usr/local/bin/azcopy", *args, "--output-level=quiet"]
    subprocess.run(command, env=env, check=True)


def managed_identity_token(resource: str) -> str:
    query = urllib.parse.urlencode({"api-version": "2018-02-01", "resource": resource})
    request = urllib.request.Request(
        f"http://169.254.169.254/metadata/identity/oauth2/token?{query}",
        headers={"Metadata": "true"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)["access_token"]


def verify_blob(base_url: str, record: dict, token: str) -> None:
    url = base_url.rstrip("/") + "/" + urllib.parse.quote(record["path"], safe="/")
    request = urllib.request.Request(
        url, method="HEAD",
        headers={"Authorization": "Bearer " + token, "x-ms-version": BLOB_API_VERSION},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                size = int(response.headers.get("Content-Length", "-1"))
                md5 = response.headers.get("Content-MD5")
                if response.status != 200 or size != record["size_bytes"] or md5 != record["md5_base64"]:
                    raise ValueError(f"Blob size/MD5 mismatch for {record['path']}")
                return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def assert_snapshot_unchanged(root: Path, records: list[dict]) -> None:
    current = {path.relative_to(root).as_posix(): path for path in run_files(root)}
    expected = {item["path"] for item in records}
    if set(current) != expected:
        raise ValueError(f"run file set changed during Blob transfer: {sorted(set(current) ^ expected)}")
    for item in records:
        stat = current[item["path"]].stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
            raise ValueError(f"run file changed during Blob transfer: {item['path']}")


def request_deallocation(vm_resource_id: str) -> int:
    token = managed_identity_token("https://management.azure.com/")
    url = ("https://management.azure.com" + vm_resource_id.rstrip("/")
           + "/deallocate?api-version=" + ARM_API_VERSION)
    request = urllib.request.Request(
        url, data=b"", method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 202, 204):
            raise ValueError(f"Azure rejected VM deallocation: HTTP {response.status}")
        return response.status


def finalize(root: Path, blob_url: str, vm_resource_id: str, train_service: str) -> None:
    if not service_completed(train_service):
        raise ValueError("training service has not completed successfully")
    validate_completed_run(root)
    (root / "blob_verified.json").unlink(missing_ok=True)
    records = [file_record(root, path) for path in run_files(root)]
    inventory = {
        "schema": 1, "training_updates": FINAL_UPDATES, "tokens_seen": FINAL_TOKENS,
        "blob_url": blob_url.rstrip("/"), "files": records,
        "file_count": len(records), "total_bytes": sum(item["size_bytes"] for item in records),
    }
    inventory_path = root / "blob_inventory.json"
    write_json_atomic(inventory_path, inventory)
    inventory_record = file_record(root, inventory_path)
    all_records = records + [inventory_record]
    log(f"uploading {len(all_records)} run files ({inventory['total_bytes']} bytes) to Azure Blob")
    azcopy("copy", str(root), blob_url, "--recursive=true", "--as-subdir=false",
           "--put-md5", "--overwrite=true")
    token = managed_identity_token("https://storage.azure.com/")
    for record in all_records:
        verify_blob(blob_url, record, token)
    assert_snapshot_unchanged(root, records)
    if (inventory_path.stat().st_size, inventory_path.stat().st_mtime_ns) != (
            inventory_record["size_bytes"], inventory_record["mtime_ns"]):
        raise ValueError("Blob inventory changed during verification")
    log("every run file has matching Blob size and Content-MD5")
    receipt = {
        "schema": 1, "result": "verified", "time_utc": datetime.now(timezone.utc).isoformat(),
        "blob_url": blob_url.rstrip("/"), "file_count": len(all_records),
        "total_bytes": inventory["total_bytes"] + inventory_record["size_bytes"],
        "inventory_sha256": inventory_record["sha256"],
        "midpoint_checkpoint_sha256": next(item["sha256"] for item in records
                                              if item["path"] == "dense/checkpoints/step_00012716.pt"),
        "final_checkpoint_sha256": next(item["sha256"] for item in records
                                           if item["path"] == "dense/checkpoints/step_00025432.pt"),
    }
    receipt_path = root / "blob_verified.json"
    write_json_atomic(receipt_path, receipt)
    receipt_record = file_record(root, receipt_path)
    azcopy("copy", str(receipt_path), blob_url.rstrip("/") + "/blob_verified.json",
           "--put-md5", "--overwrite=true")
    verify_blob(blob_url, receipt_record, managed_identity_token("https://storage.azure.com/"))
    assert_snapshot_unchanged(root, records)
    if not service_completed(train_service):
        raise ValueError("training service restarted after Blob verification")
    validate_completed_run(root)
    log("final audit, checkpoints, all logs, Blob inventory and receipt verified; requesting VM deallocation")
    status = request_deallocation(vm_resource_id)
    log(f"Azure deallocation request accepted: HTTP {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--blob-url", required=True)
    parser.add_argument("--vm-resource-id", required=True)
    parser.add_argument("--train-service", default="terminus-lean-10b.service")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--status", action="store_true", help="check completion gate only")
    args = parser.parse_args()
    if not args.blob_url.startswith("https://") or ".blob.core.windows.net/" not in args.blob_url:
        parser.error("--blob-url must be an Azure Blob container path")
    if not args.vm_resource_id.startswith("/subscriptions/"):
        parser.error("--vm-resource-id must be an Azure VM resource ID")
    if args.status:
        ready = service_completed(args.train_service)
        print(json.dumps({"training_service_completed": ready,
                          "final_audit_exists": (args.run_root / "audit_final.json").is_file(),
                          "final_checkpoint_exists": (args.run_root / "dense/checkpoints/step_00025432.pt").is_file()}))
        return
    log("finalizer armed; waiting for successful training service exit")
    while True:
        if service_completed(args.train_service):
            try:
                validate_completed_run(args.run_root)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError("training service exited but final audit is incomplete") from exc
            break
        time.sleep(args.poll_seconds)
    finalize(args.run_root, args.blob_url, args.vm_resource_id, args.train_service)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"finalizer failed; VM remains allocated: {exc}")
        sys.exit(1)
