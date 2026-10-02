"""Verify the completed QAT run in S3 and Blob before deallocating the H100 VM."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.terminus_lean_150m_10b.finalize_blob_and_deallocate import (  # noqa: E402
    finalize, log, run_files, service_completed, write_json_atomic,
)
from experiments.terminus_lean_150m_qat_2b.plan import (  # noqa: E402
    PARENT_CHECKPOINT_SHA256, TOTAL_UPDATES, USABLE_TOKENS,
)

REQUIRED_FILES = {
    "audit_final.json", "audit_final.log", "data_gate.log", "preflight.json",
    "preflight.log", "transition.log", "training.log", "service.log", "mirror.log",
    "qat_initial_eval.json", "qat_initial_eval.log",
    "views/manifest.qat.json", "views/view_receipt.json",
    "qat/transition_receipt.json", "qat/events.jsonl", "qat/stability.jsonl", "qat/run.json",
    "qat/checkpoints/step_00000000.pt", "qat/checkpoints/step_00000000.json",
    f"qat/checkpoints/step_{TOTAL_UPDATES:08d}.pt",
    f"qat/checkpoints/step_{TOTAL_UPDATES:08d}.json",
}
S3_CHECKSUM_FIELDS = (
    "ChecksumSHA256", "ChecksumSHA1", "ChecksumCRC64NVME",
    "ChecksumCRC32C", "ChecksumCRC32",
)


def s3_location(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("--s3-run-uri must be a nonempty S3 prefix")
    if parsed.query or parsed.fragment:
        raise ValueError("--s3-run-uri cannot contain a query or fragment")
    return parsed.netloc, parsed.path.strip("/")


def s3_files(root: Path) -> list[Path]:
    """Mirror every completed visible run file, including training logs/checkpoints."""
    return [path for path in run_files(root)
            if not any(part.startswith(".") for part in path.relative_to(root).parts)
            and path.name != "s3_verified.json" and not path.name.endswith(".tmp")]


def s3_head(bucket: str, key: str, expected_size: int) -> dict:
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key,
         "--checksum-mode", "ENABLED", "--output", "json"],
        check=True, capture_output=True, text=True,
    )
    head = json.loads(result.stdout)
    if head.get("ContentLength") != expected_size:
        raise ValueError(f"S3 size mismatch: s3://{bucket}/{key}")
    checksums = {name: head[name] for name in S3_CHECKSUM_FIELDS if head.get(name)}
    if not checksums:
        raise ValueError(f"S3 object has no stored checksum: s3://{bucket}/{key}")
    return {"key": key, "size_bytes": expected_size, "checksums": checksums}


def verify_s3(root: Path, uri: str) -> dict:
    """Finish the mirror and verify every visible file before Blob finalization."""
    bucket, prefix = s3_location(uri)
    excluded = ["--exclude", ".*", "--exclude", "*.tmp",
                "--exclude", "mirror.log", "--exclude", "s3_verified.json",
                "--exclude", "blob_inventory.json", "--exclude", "blob_verified.json"]
    subprocess.run(["aws", "s3", "sync", str(root), uri, *excluded, "--only-show-errors"], check=True)
    mirror_log = root / "mirror.log"
    if not mirror_log.is_file():
        raise ValueError("S3 mirror log is missing")
    subprocess.run(["aws", "s3", "cp", str(mirror_log), uri.rstrip("/") + "/mirror.log",
                    "--only-show-errors"], check=True)
    paths = s3_files(root)
    snapshots = [(path.relative_to(root).as_posix(), path.stat()) for path in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(s3_head, bucket, f"{prefix}/{name}", stat.st_size)
                   for name, stat in snapshots]
        records = [future.result() for future in futures]
    for (name, before), path in zip(snapshots, paths):
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"run file changed during S3 verification: {name}")
    if [path.relative_to(root).as_posix() for path in s3_files(root)] != [name for name, _ in snapshots]:
        raise ValueError("run file set changed during S3 verification")
    receipt = {
        "schema": 1, "result": "verified", "s3_uri": uri.rstrip("/"),
        "file_count": len(records), "total_bytes": sum(item["size_bytes"] for item in records),
        "files": records,
    }
    receipt_path = root / "s3_verified.json"
    write_json_atomic(receipt_path, receipt)
    subprocess.run(["aws", "s3", "cp", str(receipt_path), uri.rstrip("/") + "/s3_verified.json",
                    "--only-show-errors"], check=True)
    s3_head(bucket, f"{prefix}/s3_verified.json", receipt_path.stat().st_size)
    log(f"S3 verified: {len(records)} run files, {receipt['total_bytes']} bytes, including all checkpoints and logs")
    return receipt


def validate_completed_run(root: Path) -> dict:
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"required QAT run files are missing: {missing}")
    audit = json.loads((root / "audit_final.json").read_text())
    if (audit.get("phase") != "final" or audit.get("weight_mode") != "ternary"
            or audit.get("training_updates") != TOTAL_UPDATES
            or audit.get("tokens_seen") != USABLE_TOKENS
            or audit.get("incidents") != 0):
        raise ValueError("QAT final audit is incomplete or reports incidents")
    transition = json.loads((root / "qat/transition_receipt.json").read_text())
    if transition["parent"]["checkpoint_sha256"] != PARENT_CHECKPOINT_SHA256:
        raise ValueError("QAT parent checkpoint does not match the verified dense finish")
    contract = json.loads((root / "qat/run.json").read_text())["contract"]
    if contract["model"]["weight_mode"] != "ternary" or not contract["model"]["quantize_activations"]:
        raise ValueError("QAT run contract is not ternary with FP8 activations")
    for updates, field in ((0, "initial_checkpoint_sha256"), (TOTAL_UPDATES, "final_checkpoint_sha256")):
        receipt = json.loads((root / f"qat/checkpoints/step_{updates:08d}.json").read_text())
        if receipt["state"]["updates"] != updates or receipt["sha256"] != audit[field]:
            raise ValueError(f"QAT checkpoint receipt disagrees with final audit: {updates}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--blob-url", required=True)
    parser.add_argument("--s3-run-uri", required=True)
    parser.add_argument("--vm-resource-id", required=True)
    parser.add_argument("--train-service", default="terminus-lean-qat-2b.service")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--ops-log", type=Path)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not args.blob_url.startswith("https://") or ".blob.core.windows.net/" not in args.blob_url:
        parser.error("--blob-url must be an Azure Blob container path")
    if not args.vm_resource_id.startswith("/subscriptions/"):
        parser.error("--vm-resource-id must be an Azure VM resource ID")
    s3_location(args.s3_run_uri)
    if args.status:
        print(json.dumps({
            "training_service_completed": service_completed(args.train_service),
            "final_audit_exists": (args.run_root / "audit_final.json").is_file(),
            "final_checkpoint_exists": (args.run_root / f"qat/checkpoints/step_{TOTAL_UPDATES:08d}.pt").is_file(),
        }))
        return
    log("QAT finalizer armed; waiting for successful training service exit")
    while not service_completed(args.train_service):
        time.sleep(args.poll_seconds)
    validate_completed_run(args.run_root)
    verify_s3(args.run_root, args.s3_run_uri)
    finalize(args.run_root, args.blob_url, args.vm_resource_id, args.train_service,
             args.ops_log, validator=validate_completed_run,
             checkpoint_paths={
                 "initial_checkpoint_sha256": "qat/checkpoints/step_00000000.pt",
                 "final_checkpoint_sha256": f"qat/checkpoints/step_{TOTAL_UPDATES:08d}.pt",
             })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"QAT finalizer failed; VM remains allocated: {exc}")
        sys.exit(1)
