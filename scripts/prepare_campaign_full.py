"""Full campaign corpus preparation — run on an in-region VM (owner-launched).

Reads every committed campaign source object (read-only), cleans, dedups,
splits deterministically, and writes record JSONL + a corpus report. Designed
for an in-region CPU worker with authorized S3 access; outputs can be synced
to S3 by the owner. Replace the source-bucket placeholder before execution.

This is the unbounded counterpart of scripts/prepare_campaign_sample.py:
same adapters and guarantees, no row/object caps. Restartable via the cursor
file in the output directory.

Usage:
  python scripts/prepare_campaign_full.py --output-dir /data/corpus_full \
      --cache-dir /data/cache --cache-gb 200 [--sources sangraha-verified-hin,...]
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

BUCKET = "s3://your-source-bucket"
V4 = "translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v4-20260905"
V5 = "translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v5-20260906"
FACT = ("translations/Nemotron-Pretraining-Specialized-v1.2/Fact-Seeking/"
        "indictrans2-en-indic-dist-200M-beam1-20260902/fact-seeking-70shard-10lang-r16-20260903")
V4_LANGS = {"hin_Deva": "hi", "guj_Gujr": "gu", "tam_Taml": "ta", "tel_Telu": "te",
            "ben_Beng": "bn"}  # pan_Guru out of campaign scope
V5_LANGS = {"kan_Knda": "kn", "mar_Deva": "mr"}  # mal_Mlym, ory_Orya out of scope
LICENSE = {"sangraha": "CC-BY-4.0 (AI4Bharat Sangraha)", "fineweb2": "ODC-By-1.0 (FineWeb2)",
           "nemotron": "CC-BY-4.0 (NVIDIA; verify card)",
           "translation": "derived; upstream source terms + translation pipeline"}


def full_specs():
    specs = [
        SourceSpec(name="climbmix", adapter="text_id", license=LICENSE["nemotron"],
                   role="language", prefix=f"{BUCKET}/datasets/nemotron-climbmix/karpathy-400b-shuffle/",
                   language="en"),
        SourceSpec(name="nemotron-cc-hq", adapter="text_id", license=LICENSE["nemotron"],
                   role="language", prefix=f"{BUCKET}/datasets/nemotron-cc-hq/", language="en"),
        SourceSpec(name="smollm-corpus", adapter="text_id", license="mixed; verify subsets",
                   role="language", prefix=f"{BUCKET}/source-snapshots/HuggingFaceTB/smollm-corpus/",
                   language="en"),
        SourceSpec(name="nemotron-cc-math-4plus", adapter="text_id", license=LICENSE["nemotron"],
                   role="domain", prefix=f"{BUCKET}/datasets/nemotron-cc-math-v1/4plus/",
                   language="en"),
        SourceSpec(name="nemotron-cc-math-4plus-mind", adapter="text_id", license=LICENSE["nemotron"],
                   role="domain", prefix=f"{BUCKET}/datasets/nemotron-cc-math-v1/4plus_MIND/",
                   language="en"),
        SourceSpec(name="nemotron-cc-math-3", adapter="text_id", license=LICENSE["nemotron"],
                   role="domain", prefix=f"{BUCKET}/datasets/nemotron-cc-math-v1/3/",
                   language="en", notes="optional score band"),
        SourceSpec(name="math-textbooks", adapter="text_id", license=LICENSE["nemotron"],
                   role="domain",
                   prefix=f"{BUCKET}/datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/",
                   language="en"),
        SourceSpec(name="code-concepts", adapter="text_id", license=LICENSE["nemotron"],
                   role="domain",
                   prefix=f"{BUCKET}/datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/",
                   language="en"),
        SourceSpec(name="scientific-coding", adapter="text_id", license=LICENSE["nemotron"],
                   role="domain",
                   prefix=f"{BUCKET}/datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding/",
                   language="en"),
    ]
    for tier in ("verified", "synthetic", "unverified"):
        names = CAMPAIGN_SOURCE_LANGS.items() if tier != "synthetic" else \
            [("hin_Deva", "hi"), ("mar_Deva", "mr"), ("kan_Knda", "kn"), ("tel_Telu", "te"),
             ("tam_Taml", "ta"), ("guj_Gujr", "gu"), ("ben_Beng", "bn")]
        for dirname, code2 in names:
            if code2 == "en":
                continue
            specs.append(SourceSpec(name=f"sangraha-{tier}-{dirname}", adapter="sangraha",
                                    license=LICENSE["sangraha"], role="language",
                                    prefix=f"{BUCKET}/datasets/sangraha/{tier}/{dirname}/",
                                    language=code2))
    for lang_script, code2 in FINEWEB2_SOURCE_LANGS.items():
        specs.append(SourceSpec(name=f"fineweb2-{lang_script}", adapter="fineweb2",
                                license=LICENSE["fineweb2"], role="language",
                                prefix=f"{BUCKET}/datasets/fineweb2-indic/{lang_script}/",
                                language=code2))
    # Preferred translation versions first so skip_ids_from sees them.
    for lang_script, code2 in V5_LANGS.items():
        specs.append(SourceSpec(name=f"cosmopedia-v5-{lang_script}", adapter="translation",
                                license=LICENSE["translation"], role="language",
                                prefix=f"{BUCKET}/{V5}/language={lang_script}/", language=code2))
    for lang_script, code2 in V4_LANGS.items():
        v5_twin = f"cosmopedia-v5-{lang_script}"
        specs.append(SourceSpec(name=f"cosmopedia-v4-{lang_script}", adapter="translation",
                                license=LICENSE["translation"], role="language",
                                prefix=f"{BUCKET}/{V4}/language={lang_script}/", language=code2,
                                skip_ids_from=(v5_twin,) if v5_twin in
                                {s.name for s in specs} else ()))
    for lang_script, code2 in {**V4_LANGS, **V5_LANGS}.items():
        specs.append(SourceSpec(name=f"fact-seeking-r16-{lang_script}", adapter="translation",
                                license=LICENSE["translation"], role="language",
                                prefix=f"{BUCKET}/{FACT}/language={lang_script}/", language=code2))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-gb", type=int, default=200)
    parser.add_argument("--sources", help="comma-separated subset for staged runs")
    parser.add_argument("--eval-root", default="eval")
    parser.add_argument("--dev-per-10000", type=int, default=50)
    parser.add_argument("--test-per-10000", type=int, default=50)
    args = parser.parse_args()
    guard_paths = sorted(Path(args.eval_root).glob("scalingbench/*/data.jsonl"))
    guard_texts = task_text_hashes(guard_paths) if guard_paths else []
    if not guard_texts:
        raise SystemExit("contamination guard needs the fetched evaluation tasks")
    config = CorpusBuildConfig(scope_name=CAMPAIGN_SCOPE_V1, split_salt=CAMPAIGN_SCOPE_V1,
                               dev_per_10000=args.dev_per_10000,
                               test_per_10000=args.test_per_10000)
    builder = CorpusBuilder(config, args.output_dir, cache_dir=args.cache_dir,
                            cache_bytes=args.cache_gb * 1024**3,
                            contamination_texts=guard_texts)
    specs = full_specs()
    if args.sources:
        wanted = set(args.sources.split(","))
        specs = [spec for spec in specs if spec.name in wanted]
        missing = wanted - {spec.name for spec in specs}
        if missing:
            raise SystemExit(f"unknown sources: {sorted(missing)}")
    raw_manifest = {}
    for spec in specs:
        print(f"[{spec.name}] start", flush=True)
        objects = builder.build_source(spec)
        raw_manifest[spec.name] = {"objects": len(objects), "license": spec.license,
                                   "prefix": spec.prefix, "language": spec.language,
                                   "role": spec.role}
        print(f"[{spec.name}] {builder.stats.get(spec.name)}", flush=True)
    report = builder.finish(specs, raw_manifest)
    print(json.dumps({"sources": len(specs), "stats": report["stats"]}, indent=1))


if __name__ == "__main__":
    main()
