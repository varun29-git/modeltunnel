"""Merge per-source corpus records into token-set streams with cross-source dedup.

Streams: en (climbmix), math (4plus + 4plus_MIND), code (code-concepts +
scientific-coding), and one per Indic language (sangraha-verified + fineweb2).
Cross-source exact-text duplicates (by the records' text_sha256) are dropped;
per-source ID/near-dup already happened during the build. Writes the merged
record files, the records map JSON for the tokenizer, and licenses.json.

The corpus builder writes train/dev/test splits; the manifest's val split
takes dev. Test records flow to the untouched directory at tokenize time.

Usage:
  python scripts/merge_tokenset_records.py --build-root corpus_build --output-dir merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STREAMS = {
    "en": ["climbmix"],
    "math": ["nemotron-cc-math-4plus", "nemotron-cc-math-4plus-mind"],
    "code": ["code-concepts", "scientific-coding"],
    "hi": ["sangraha-verified-hin", "fineweb2-hin"],
    "mr": ["sangraha-verified-mar", "fineweb2-mar"],
    "kn": ["sangraha-verified-kan", "fineweb2-kan"],
    "te": ["sangraha-verified-tel", "fineweb2-tel"],
    "ta": ["sangraha-verified-tam", "fineweb2-tam"],
    "gu": ["sangraha-verified-guj", "fineweb2-guj"],
    "bn": ["sangraha-verified-ben", "fineweb2-ben"],
}
# manifest split -> corpus-builder split
SPLIT_MAP = (("train", "train"), ("val", "dev"), ("test", "test"))
SHUFFLE_ALGORITHM = "blake2b128_bucket_sort_v1"
SHUFFLE_BUCKETS = 256


def _shuffle_key(seed: str, stream: str, uid: str) -> bytes:
    return hashlib.blake2b(
        f"{seed}\0{stream}\0{uid}".encode("utf-8"), digest_size=16,
    ).digest()


def _write_shuffled_train(source_rows, target: Path, *, seed: str, stream: str) -> int:
    """Globally hash-sort documents using disk buckets and bounded memory."""
    staging = target.parent / f".shuffle.{stream}.train"
    if staging.exists():
        shutil.rmtree(staging)  # derived data left by an interrupted merge
    staging.mkdir()
    writers = {}
    count = 0
    try:
        for line in source_rows:
            uid = json.loads(line)["uid"]
            key = _shuffle_key(seed, stream, uid)
            bucket = key[0]
            writer = writers.get(bucket)
            if writer is None:
                writer = writers[bucket] = (staging / f"{bucket:02x}.jsonl").open("wb")
            writer.write(key.hex().encode("ascii") + line.encode("utf-8"))
            count += 1
        for writer in writers.values():
            writer.close()
        writers.clear()
        with target.open("wb") as output:
            for bucket in range(SHUFFLE_BUCKETS):
                path = staging / f"{bucket:02x}.jsonl"
                if not path.exists():
                    continue
                with path.open("rb") as source:
                    rows = sorted(source, key=lambda row: row[:32])
                for row in rows:
                    output.write(row[32:])
                path.unlink()
        return count
    finally:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip sources without a corpus report (tests only)")
    parser.add_argument("--shuffle-seed", help="deterministically shuffle train documents across sources")
    args = parser.parse_args()
    build_root = Path(args.build_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.allow_missing:
        missing = [source for sources in STREAMS.values() for source in sources
                   if not (build_root / source / "corpus_report.json").exists()]
        if missing:
            raise SystemExit(f"sources not built: {sorted(set(missing))}")
    seen = set()
    dropped = kept = 0
    records_map, licenses, stats = {}, {}, {}
    for stream, sources in sorted(STREAMS.items()):
        splits = {}
        for split, src_split in SPLIT_MAP:
            target = out / f"records.{stream}.{split}.jsonl"
            count = 0
            def retained_rows():
                nonlocal count, dropped, kept
                for source in sources:
                    report_path = build_root / source / "corpus_report.json"
                    if not report_path.exists():
                        continue
                    report = json.loads(report_path.read_text())
                    src_entry = (report.get("sources") or {}).get(source) or {}
                    if src_entry.get("license"):
                        licenses.setdefault(stream, set()).add(src_entry["license"])
                    stats.setdefault(stream, {})[source] = report["stats"].get(source, {})
                    path = build_root / source / f"records.{source}.{src_split}.jsonl"
                    if not path.exists():
                        continue
                    with path.open(encoding="utf-8") as src_handle:
                        for line in src_handle:
                            digest = json.loads(line)["text_sha256"]
                            if digest in seen:
                                dropped += 1
                                continue
                            seen.add(digest)
                            kept += 1
                            count += 1
                            yield line
            if split == "train" and args.shuffle_seed:
                _write_shuffled_train(retained_rows(), target, seed=args.shuffle_seed, stream=stream)
            else:
                with target.open("w", encoding="utf-8") as handle:
                    for line in retained_rows():
                        handle.write(line)
            if count:
                splits[split] = [str(target)]
            else:
                target.unlink()
        if splits:
            records_map[stream] = splits
    (out / "records_map.json").write_text(json.dumps(records_map, indent=1))
    (out / "licenses.json").write_text(json.dumps({k: sorted(v) for k, v in licenses.items()},
                                                  indent=1, sort_keys=True))
    (out / "merge_stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True))
    (out / "shuffle_receipt.json").write_text(json.dumps({
        "train": {"algorithm": SHUFFLE_ALGORITHM, "seed": args.shuffle_seed,
                  "buckets": SHUFFLE_BUCKETS} if args.shuffle_seed else None,
        "val_test": "unchanged_source_order",
    }, indent=1, sort_keys=True))
    print(json.dumps({"kept": kept, "cross_source_exact_dups_dropped": dropped,
                      "streams": {k: {s: len(v) for s, v in m.items()}
                                  for k, m in records_map.items()}}, indent=1))


if __name__ == "__main__":
    main()
