"""Build corpus records for one token-set source (run one process per source).

Usage:
  python scripts/build_tokenset_corpus.py --source sangraha-verified-hin \
      --output-dir corpus_build/sangraha-verified-hin --cache-dir cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminus.corpus import CorpusBuildConfig, CorpusBuilder, SourceSpec
from terminus.integrity import CAMPAIGN_SCOPE_V1
from terminus.task_eval import task_text_hashes

BUCKET = "s3://your-source-bucket"

# (name, adapter, language, prefix, target_bytes of raw text input objects)
SOURCES = {
    "climbmix": ("text_id", "en", f"{BUCKET}/datasets/nemotron-climbmix/karpathy-400b-shuffle/",
                 30 * 2**30),
    "nemotron-cc-math-4plus": ("text_id", "en", f"{BUCKET}/datasets/nemotron-cc-math-v1/4plus/",
                               4 * 2**30),
    "nemotron-cc-math-4plus-mind": ("text_id", "en", f"{BUCKET}/datasets/nemotron-cc-math-v1/4plus_MIND/",
                                    2 * 2**30),
    "code-concepts": ("text_id", "en", f"{BUCKET}/datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/",
                      5 * 2**30),
    "scientific-coding": ("text_id", "en", f"{BUCKET}/datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding/",
                          2**30),
    "sangraha-verified-hin": ("sangraha", "hi", f"{BUCKET}/datasets/sangraha/verified/hin/", 6 * 2**30),
    "sangraha-verified-mar": ("sangraha", "mr", f"{BUCKET}/datasets/sangraha/verified/mar/", 6 * 2**30),
    "sangraha-verified-kan": ("sangraha", "kn", f"{BUCKET}/datasets/sangraha/verified/kan/", 6 * 2**30),
    "sangraha-verified-tel": ("sangraha", "te", f"{BUCKET}/datasets/sangraha/verified/tel/", 6 * 2**30),
    "sangraha-verified-tam": ("sangraha", "ta", f"{BUCKET}/datasets/sangraha/verified/tam/", 6 * 2**30),
    "sangraha-verified-guj": ("sangraha", "gu", f"{BUCKET}/datasets/sangraha/verified/guj/", 6 * 2**30),
    "sangraha-verified-ben": ("sangraha", "bn", f"{BUCKET}/datasets/sangraha/verified/ben/", 6 * 2**30),
    "fineweb2-hin": ("fineweb2", "hi", f"{BUCKET}/datasets/fineweb2-indic/hin_Deva/", 3 * 2**30),
    "fineweb2-mar": ("fineweb2", "mr", f"{BUCKET}/datasets/fineweb2-indic/mar_Deva/", 3 * 2**30),
    "fineweb2-kan": ("fineweb2", "kn", f"{BUCKET}/datasets/fineweb2-indic/kan_Knda/", 3 * 2**30),
    "fineweb2-tel": ("fineweb2", "te", f"{BUCKET}/datasets/fineweb2-indic/tel_Telu/", 3 * 2**30),
    "fineweb2-tam": ("fineweb2", "ta", f"{BUCKET}/datasets/fineweb2-indic/tam_Taml/", 3 * 2**30),
    "fineweb2-guj": ("fineweb2", "gu", f"{BUCKET}/datasets/fineweb2-indic/guj_Gujr/", 3 * 2**30),
    "fineweb2-ben": ("fineweb2", "bn", f"{BUCKET}/datasets/fineweb2-indic/ben_Beng/", 3 * 2**30),
}

# Input objects are compressed Parquet, so their byte counts cannot predict
# tokenizer output. These limits apply to *accepted decoded UTF-8 text*.
# Measured sample fertility (tokens/text byte) leaves about 15-35% headroom
# above the 10B mixture requirements; the final pack gate checks actual tokens.
TARGET_TEXT_BYTES = {
    "climbmix": 25_000_000_000,
    "nemotron-cc-math-4plus": 2_000_000_000,
    "nemotron-cc-math-4plus-mind": 2_000_000_000,
    "code-concepts": 3_000_000_000,
    "scientific-coding": 1_500_000_000,
    **{f"sangraha-verified-{lang}": 4_000_000_000
       for lang in ("hin", "mar", "kan", "tel", "tam", "guj", "ben")},
    **{f"fineweb2-{lang}": 1_500_000_000
       for lang in ("hin", "mar", "kan", "tel", "tam", "guj", "ben")},
}

LICENSES = {"text_id": "CC-BY-4.0 (NVIDIA; verify card)", "sangraha": "CC-BY-4.0 (AI4Bharat Sangraha)",
            "fineweb2": "ODC-By-1.0 (FineWeb2)"}
LICENSES_BY_SOURCE = {"climbmix": "CC-BY-4.0 (NVIDIA; verify card)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--eval-root", default="eval")
    parser.add_argument("--near-dup", action="store_true",
                        help="enable MinHash near-duplicate detection (off for token-set runs)")
    args = parser.parse_args()
    adapter, language, prefix, target = SOURCES[args.source]
    guard_paths = sorted(Path(args.eval_root).glob("scalingbench/*/data.jsonl"))
    guard_texts = task_text_hashes(guard_paths, max_rows_per_file=2000) if guard_paths else []
    config = CorpusBuildConfig(scope_name=CAMPAIGN_SCOPE_V1, split_salt=CAMPAIGN_SCOPE_V1,
                               dev_per_10000=50, test_per_10000=50,
                               near_duplicates=args.near_dup)
    builder = CorpusBuilder(config, args.output_dir,
                            cache_dir=Path(args.cache_dir) / args.source,
                            contamination_texts=guard_texts)
    spec = SourceSpec(name=args.source, adapter=adapter,
                      license=LICENSES_BY_SOURCE.get(args.source, LICENSES[adapter]),
                      role="language", prefix=prefix, language=language, max_bytes=target,
                      max_output_text_bytes=TARGET_TEXT_BYTES[args.source],
                      object_selection_seed="terminus-tokensets-20261001-v1")
    objects = builder.build_source(spec)
    report = builder.finish([spec], {args.source: {"objects": len(objects),
                                                   "license": spec.license}})
    print(json.dumps({args.source: report["stats"].get(args.source)}, indent=1))


if __name__ == "__main__":
    main()
