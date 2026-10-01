"""Preserve and roll back telemetry emitted after the last durable checkpoint.

The trainer resumes its data cursor from the checkpoint. Replayed updates must
not leave duplicate train/stability records in the active audit stream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path


FILES = ("events.jsonl", "stability.jsonl", "incidents.jsonl")


def _update(record):
    if "updates" in record:
        return record["updates"]
    return (record.get("state") or {}).get("updates")


def reconcile(run_dir):
    run_dir = Path(run_dir)
    receipt = json.loads((run_dir / "checkpoints/latest.json").read_text())
    committed = receipt["state"]["updates"]
    archive = None
    counts = {}
    for name in FILES:
        path = run_dir / name
        if not path.exists():
            continue
        kept = []
        dropped = 0
        for line in path.read_bytes().splitlines(keepends=True):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                dropped += 1
                continue
            update = _update(record)
            if update is not None and update > committed:
                dropped += 1
            else:
                kept.append(line.rstrip(b"\r\n") + b"\n")
        if not dropped:
            continue
        if archive is None:
            archive = run_dir / "rollbacks" / f"after_{committed:08d}_{uuid.uuid4().hex}"
            archive.mkdir(parents=True)
        shutil.copy2(path, archive / name)
        temporary = path.with_name(f".{name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("xb") as handle:
            handle.writelines(kept)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        counts[name] = dropped
    if archive is not None:
        (archive / "receipt.json").write_text(json.dumps({
            "last_committed_update": committed, "removed_active_records": counts,
            "reason": "the next invocation will replay updates after the checkpoint",
        }, indent=2) + "\n")
    return {"last_committed_update": committed,
            "archive": str(archive) if archive else None, "removed_active_records": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
