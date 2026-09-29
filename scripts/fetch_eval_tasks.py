"""Fetch and version the public evaluation tasks (read-only network use).

Downloads the five ScalingBench task files at the revision pinned in
DATA_SOURCE_MAP.md, records SHA-256, bytes, license, and role
(development vs untouched-final) in a versioned evaluation manifest.
MILU (Indic) is gated on Hugging Face; if access fails the manifest records
the failure explicitly instead of substituting another task silently.

These are evaluation records only; they must never enter pretraining (the
corpus contamination guard hashes them).

Usage: .venv/bin/python scripts/fetch_eval_tasks.py --output-dir eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

SCALINGBENCH_REVISION = "c730220daf6dbd4885ea254bcaca0e1f86384098"
SCALINGBENCH_TASKS = ["bbh", "human_eval", "math", "mbpp", "mmlu"]
SCALINGBENCH_URL = ("https://huggingface.co/datasets/openbmb/DensingLaw-ScalingBench/"
                    "resolve/{revision}/{task}/data.jsonl")


def fetch(url, target):
    request = urllib.request.Request(url, headers={"User-Agent": "terminus-modeltunnelv2-eval-fetch"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            handle.write(chunk)
    return digest.hexdigest(), size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "manifest_version": 1, "created_unix": time.time(), "tasks": {}}
    for task in SCALINGBENCH_TASKS:
        target = root / "scalingbench" / task / "data.jsonl"
        entry = {"source": "openbmb/DensingLaw-ScalingBench",
                 "revision": SCALINGBENCH_REVISION, "license": "Apache-2.0",
                 "role": "development", "url": SCALINGBENCH_URL.format(
                     revision=SCALINGBENCH_REVISION, task=task),
                 "file": str(target.relative_to(root))}
        if target.exists():
            entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            entry["bytes"] = target.stat().st_size
            entry["status"] = "fetched"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                sha, size = fetch(entry["url"], target)
                entry.update(sha256=sha, bytes=size, status="fetched")
            except Exception as exc:
                entry["status"] = f"fetch_failed: {type(exc).__name__}: {exc}"
        entry["rows"] = sum(1 for _ in target.open("rb")) if target.exists() else 0
        manifest["tasks"][f"scalingbench/{task}"] = entry
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        print(task, entry["status"], entry.get("bytes"), flush=True)
    # MILU is gated; attempt and record the outcome without embedding credentials.
    milu = {"source": "ai4bharat/MILU", "revision": "unspecified-gated", "license": "CC-BY-4.0",
            "role": "development", "url": "https://huggingface.co/datasets/ai4bharat/MILU",
            "languages": ["hi", "mr", "kn", "te", "ta", "gu", "bn"]}
    try:
        probe = urllib.request.Request(
            "https://huggingface.co/api/datasets/ai4bharat/MILU",
            headers={"User-Agent": "terminus-modeltunnelv2-eval-fetch"})
        with urllib.request.urlopen(probe, timeout=30) as response:
            payload = json.loads(response.read())
        milu["status"] = "gated" if payload.get("gated") else "accessible"
        milu["card_revision"] = payload.get("sha")
    except Exception as exc:
        milu["status"] = f"probe_failed: {type(exc).__name__}: {exc}"
    manifest["tasks"]["milu"] = milu
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("milu", milu["status"])


if __name__ == "__main__":
    main()
