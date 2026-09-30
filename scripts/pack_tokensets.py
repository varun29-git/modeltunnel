"""Pack token shards into the 10B manifest and the nested 6B/3B manifests.

Reads WORK_DIR env (set by run_tokenset_vm.sh) or --work. Validates the 10B
manifest with the production gate, then writes manifest.6b.json and
manifest.3b.json as budget remixes over the same shards — exact prefixes by
construction (same shards, mixture, seq_len; only the budget differs).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminus.integrity import (CAMPAIGN_DOMAINS, CAMPAIGN_LANGUAGES, CAMPAIGN_SCOPE_V1,
                                validate_manifest)
from terminus.model import BitNetLlamaConfig
from terminus.pack_tokens import pack_token_streams
from terminus.remix import remix_manifest


def main():
    work = Path(os.environ.get("WORK_DIR") or sys.argv[sys.argv.index("--work") + 1])
    mix = {"math": 0.1, "code": 0.1}
    for lang in CAMPAIGN_LANGUAGES:
        if lang != "en":
            mix[lang] = 0.043
    mix["en"] = 1.0 - sum(mix.values())  # 0.499; English takes the residual
    scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
             "domains": list(CAMPAIGN_DOMAINS)}
    languages = {**{lang: lang for lang in CAMPAIGN_LANGUAGES}, "math": "en", "code": "en"}
    manifest = pack_token_streams(work / "packed", "tokenizer/tokenizer-65k.json",
                                  source_mix=mix, total_train_tokens=10_000_000_000,
                                  seq_len=4096, batch_size=8, scope=scope,
                                  languages_by_stream=languages)
    model_config = BitNetLlamaConfig(vocab_size=65536, max_seq_len=4096, pad_token_id=3,
                                     attention_backend="sdpa_reference")
    report = validate_manifest(manifest, work / "packed", model_config, development_mode=False)
    print("10B manifest validated:", report["manifest_sha256"])
    for billions in (6, 3):
        out = remix_manifest(work / "packed/manifest.json", work / "packed", model_config,
                             total_train_tokens=billions * 10**9, name=f"{billions}b")
        print(f"{billions}B manifest:", out["manifest_sha256"])
    (work / "packed/validation_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
