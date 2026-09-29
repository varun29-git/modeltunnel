"""Build a bounded, real-data campaign corpus sample from S3 (read-only).

Streams a small number of rows from each campaign source family through the
full cleaning path (NFC, language filter, stable-ID + exact + near-dup dedup,
translation version preference with quarantine, contamination guard against
the fetched evaluation tasks, deterministic train/dev/test split), and writes
records JSONL plus a corpus report. This is the bounded sample used by the
end-to-end smoke test — it is not the full corpus preparation.

Usage:
  .venv/bin/python scripts/prepare_campaign_sample.py \
      --output-dir sample/corpus --cache-dir sample/cache --max-rows 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminus.corpus import (CAMPAIGN_SOURCE_LANGS, FINEWEB2_SOURCE_LANGS, CorpusBuildConfig,
                             CorpusBuilder, SourceSpec)
from terminus.integrity import CAMPAIGN_SCOPE_V1
from terminus.task_eval import task_text_hashes

BUCKET = "s3://nemotron-training-data-us-east-1"
# License labels come from the dataset cards recorded in the inventory docs;
# re-verify each card before full preparation.
LICENSES = {"climbmix": "CC-BY-4.0 (NVIDIA Nemotron-CC family; verify card)",
            "nemotron-cc": "CC-BY-4.0 (NVIDIA Nemotron-CC; verify card)",
            "sangraha": "CC-BY-4.0 (AI4Bharat Sangraha)",
            "fineweb2": "ODC-By-1.0 (HuggingFace FineWeb2)",
            "cosmopedia-translation": "derived; upstream SmolLM-Corpus terms + translation pipeline",
            "code-concepts": "CC-BY-4.0 (NVIDIA Nemotron-Pretraining-Specialized; verify card)"}


def source_specs(max_rows):
    specs = [
        SourceSpec(name="climbmix", adapter="text_id", license=LICENSES["climbmix"],
                   role="language", prefix=f"{BUCKET}/datasets/nemotron-climbmix/karpathy-400b-shuffle/",
                   language="en", max_objects=1, max_rows=max_rows),
        SourceSpec(name="nemotron-cc-math-4plus", adapter="text_id", license=LICENSES["nemotron-cc"],
                   role="domain", prefix=f"{BUCKET}/datasets/nemotron-cc-math-v1/4plus/",
                   language="en", max_objects=1, max_rows=max_rows),
        SourceSpec(name="nemotron-code-concepts", adapter="text_id", license=LICENSES["code-concepts"],
                   role="domain",
                   prefix=f"{BUCKET}/datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/",
                   language="en", max_objects=1, max_rows=max_rows),
    ]
    for code3, code2 in CAMPAIGN_SOURCE_LANGS.items():
        if code2 == "en":
            continue
        specs.append(SourceSpec(name=f"sangraha-verified-{code3}", adapter="sangraha",
                                license=LICENSES["sangraha"], role="language",
                                prefix=f"{BUCKET}/datasets/sangraha/verified/{code3}/",
                                language=code2, max_objects=1, max_rows=max_rows))
    # fineweb2 is exercised by the full-prep path; its 4.6 GiB objects are out of
    # scope for the bounded sample (Hindi is covered by sangraha + cosmopedia-v4).
    v4 = "translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v4-20260905"
    v5 = "translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v5-20260906"
    specs.append(SourceSpec(name="cosmopedia-v5-mar", adapter="translation",
                            license=LICENSES["cosmopedia-translation"], role="language",
                            prefix=f"{BUCKET}/{v5}/language=mar_Deva/", language="mr",
                            max_objects=1, max_rows=max_rows // 2))
    specs.append(SourceSpec(name="cosmopedia-v5-kan", adapter="translation",
                            license=LICENSES["cosmopedia-translation"], role="language",
                            prefix=f"{BUCKET}/{v5}/language=kan_Knda/", language="kn",
                            max_objects=1, max_rows=max_rows // 2))
    # v4 is listed after v5 so skip_ids_from sees the preferred ids first.
    specs.append(SourceSpec(name="cosmopedia-v4-hin", adapter="translation",
                            license=LICENSES["cosmopedia-translation"], role="language",
                            prefix=f"{BUCKET}/{v4}/language=hin_Deva/", language="hi",
                            skip_ids_from=("cosmopedia-v5-hin",),
                            max_objects=1, max_rows=max_rows // 2))
    return specs


def stream_map():
    """Raw source name -> training stream (coverage key)."""
    streams = {"climbmix": "en", "nemotron-cc-math-4plus": "math",
               "nemotron-code-concepts": "code",
               "cosmopedia-v4-hin": "hi", "cosmopedia-v5-mar": "mr",
               "cosmopedia-v5-kan": "kn"}
    for code2 in CAMPAIGN_SOURCE_LANGS.values():
        if code2 != "en":
            streams[f"sangraha-verified-{[k for k, v in CAMPAIGN_SOURCE_LANGS.items() if v == code2][0]}"] = code2
    return streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--eval-root", default="eval", help="fetched task files for the guard")
    parser.add_argument("--guard-rows", type=int, default=2000,
                        help="task rows per file used for contamination hashing in the sample")
    args = parser.parse_args()
    guard_texts = []
    eval_root = Path(args.eval_root)
    for path in sorted(eval_root.glob("scalingbench/*/data.jsonl")):
        import itertools
        with path.open() as handle:
            rows = [json.loads(line) for line in itertools.islice(handle, args.guard_rows)]
        for row in rows:
            guard_texts.extend(v for v in (row.get("input"), row.get("output"))
                               if isinstance(v, str))
        print(f"guard: {path} ({len(rows)} rows)", flush=True)
    config = CorpusBuildConfig(scope_name=CAMPAIGN_SCOPE_V1, split_salt=CAMPAIGN_SCOPE_V1,
                               dev_per_10000=1000, test_per_10000=1000)
    builder = CorpusBuilder(config, args.output_dir, cache_dir=args.cache_dir,
                            contamination_texts=guard_texts)
    specs = source_specs(args.max_rows)
    raw_manifest = {}
    for spec in specs:
        print(f"building {spec.name}", flush=True)
        objects = builder.build_source(spec)
        raw_manifest[spec.name] = {"objects": objects, "license": spec.license,
                                   "prefix": spec.prefix, "language": spec.language,
                                   "role": spec.role}
        print(f"  -> {builder.stats.get(spec.name)}", flush=True)
    report = builder.finish(specs, raw_manifest)
    streams = stream_map()
    records = {}
    for path in sorted(Path(args.output_dir).glob("records.*.jsonl")):
        _, raw_source, split, _ = path.name.split(".")
        stream = streams[raw_source]
        records.setdefault(stream, []).append(str(path))
    (Path(args.output_dir) / "streams.json").write_text(json.dumps(records, indent=1))
    print(json.dumps({"streams": {k: len(v) for k, v in sorted(records.items())},
                      "stats": report["stats"]}, indent=1))


if __name__ == "__main__":
    main()
