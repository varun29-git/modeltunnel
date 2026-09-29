"""Round-trip scan: every record of a corpus sample through the frozen tokenizer.

Usage: .venv/bin/python scripts/roundtrip_scan.py --corpus-glob 'sample/corpus/records.*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminus.tokenizer65k import Frozen65kTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-glob", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer-65k.json")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()
    tokenizer = Frozen65kTokenizer(args.tokenizer)
    fails = total = 0
    examples = []
    started = time.time()
    for path in sorted(glob.glob(args.corpus_glob)):
        for line in open(path):
            total += 1
            text = unicodedata.normalize("NFC", json.loads(line)["text"])
            try:
                if tokenizer.decode(tokenizer.encode(text)) != text:
                    fails += 1
                    if len(examples) < args.max_examples:
                        examples.append(path)
            except Exception as exc:
                fails += 1
                examples.append(f"{path}: {type(exc).__name__}: {exc}")
    print(json.dumps({"documents": total, "roundtrip_failures": fails,
                      "seconds": round(time.time() - started, 1),
                      "example_failures": examples}, indent=1))


if __name__ == "__main__":
    main()
