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
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip sources without a corpus report (tests only)")
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
            with target.open("w", encoding="utf-8") as handle:
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
                            handle.write(line)
                            kept += 1
                            count += 1
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
    print(json.dumps({"kept": kept, "cross_source_exact_dups_dropped": dropped,
                      "streams": {k: {s: len(v) for s, v in m.items()}
                                  for k, m in records_map.items()}}, indent=1))


if __name__ == "__main__":
    main()
