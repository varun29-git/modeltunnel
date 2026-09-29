"""Campaign data preparation tests: streams, scope, domains, split hygiene."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from terminus.data import TokenManifest
from terminus.integrity import (CAMPAIGN_DOMAINS, CAMPAIGN_LANGUAGES, CAMPAIGN_SCOPE_V1,
                                validate_manifest)
from terminus.prepare_campaign_data import prepare_campaign_data
from tests.helpers import tiny_config

TOKENIZER = str(Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer-65k.json")


def make_records(root, stream, language, texts, splits, raw_source=None, license="test-only"):
    path = Path(root) / f"records.{stream}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, (text, split) in enumerate(zip(texts, splits)):
            record = {"uid": f"u{index}", "doc_id": f"d{index}", "source": raw_source or stream,
                      "language": language, "role": "language", "license": license,
                      "split": split, "text": text,
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class PrepareCampaignTests(unittest.TestCase):
    def build(self, root, *, with_math_code=True, test_records=True):
        root = Path(root)
        streams = {}
        texts = {
            "hi": "यह एक परीक्षण वाक्य है जिसमें पर्याप्त सामग्री है। " * 20,
            "en": "This is a test sentence with enough content to tokenize well. " * 20,
        }
        for language in CAMPAIGN_LANGUAGES:
            base = texts.get(language, texts["hi"])
            text = f"{language} अद्वितीय सामग्री标记 {language}. " + base  # distinct per language
            corpus_texts = [text] * 12
            splits = ["train"] * 9 + ["dev"] * 2 + (["test"] if test_records else [])
            streams[language] = str(make_records(root, language, language, corpus_texts, splits))
        if with_math_code:
            for domain in CAMPAIGN_DOMAINS:
                text = ("def f(x): return x\n" if domain == "code"
                        else "Let x^2 + y^2 = z^2. Then z = sqrt(x^2+y^2). ") * 20
                streams[domain] = str(make_records(root, domain, "en", [text] * 12,
                                                   ["train"] * 10 + ["dev"] * 2,
                                                   raw_source=f"raw-{domain}-a"))
        return streams

    def test_scope_manifest_with_domains_validates(self):
        with tempfile.TemporaryDirectory() as root:
            streams = self.build(root)
            scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
                     "domains": list(CAMPAIGN_DOMAINS)}
            mix = {**{lang: 0.1 for lang in CAMPAIGN_LANGUAGES},
                   "math": 0.1, "code": 0.1}
            out = Path(root) / "packed"
            manifest = prepare_campaign_data(streams, TOKENIZER, out, source_mix=mix,
                                             total_train_tokens=640, seq_len=16, scope=scope)
            report = validate_manifest(manifest, out,
                                       tiny_config(vocab_size=65536, pad_token_id=3,
                                                   max_seq_len=16))
            self.assertEqual(set(report["tokens_by_language"]["train"]),
                             set(CAMPAIGN_LANGUAGES) | {"en"})
            # test-split records were never tokenized
            inventory = manifest.provenance["token_inventory"]
            self.assertFalse(any(key.endswith("/test") for key in inventory))
            self.assertTrue(all(key.endswith(("/train", "/val")) for key in inventory))
            self.assertEqual(manifest.provenance["scope"]["name"], CAMPAIGN_SCOPE_V1)
            self.assertEqual(manifest.provenance["document_separator_id"], 4)
            # <eod> terminates each packed document stream
            for spec in manifest.shards:
                tokens = np.fromfile(out / spec.path, dtype=np.uint16)
                self.assertEqual(int(tokens[-1]), 4)

    def test_missing_domain_fails_production_validation(self):
        with tempfile.TemporaryDirectory() as root:
            streams = self.build(root, with_math_code=False)
            scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
                     "domains": list(CAMPAIGN_DOMAINS)}
            mix = {lang: 0.125 for lang in CAMPAIGN_LANGUAGES}
            out = Path(root) / "packed"
            manifest = prepare_campaign_data(streams, TOKENIZER, out, source_mix=mix,
                                             total_train_tokens=640, seq_len=16, scope=scope)
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_manifest(manifest, out,
                                  tiny_config(vocab_size=65536, pad_token_id=3, max_seq_len=16))

    def test_hash_mismatch_and_bad_split_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            streams = self.build(root)
            path = Path(streams["hi"])
            lines = path.read_text().splitlines()
            record = json.loads(lines[0])
            record["text"] = record["text"] + "tampered"
            lines[0] = json.dumps(record, ensure_ascii=False)
            path.write_text("\n".join(lines) + "\n")
            scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
                     "domains": list(CAMPAIGN_DOMAINS)}
            mix = {**{lang: 0.1 for lang in CAMPAIGN_LANGUAGES}, "math": 0.1, "code": 0.1}
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare_campaign_data(streams, TOKENIZER, Path(root) / "packed",
                                      source_mix=mix, total_train_tokens=640, seq_len=16,
                                      scope=scope)


if __name__ == "__main__":
    unittest.main()


class RemixTests(unittest.TestCase):
    def test_remix_revalidates_and_preserves_shards(self):
        with tempfile.TemporaryDirectory() as root:
            streams = PrepareCampaignTests().build(root)
            scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
                     "domains": list(CAMPAIGN_DOMAINS)}
            mix = {**{lang: 0.1 for lang in CAMPAIGN_LANGUAGES}, "math": 0.1, "code": 0.1}
            out = Path(root) / "packed"
            prepare_campaign_data(streams, TOKENIZER, out, source_mix=mix,
                                  total_train_tokens=640, seq_len=16, scope=scope)
            from terminus.remix import remix_manifest
            model_config = tiny_config(vocab_size=65536, pad_token_id=3, max_seq_len=16)
            remix = remix_manifest(out / "manifest.json", out, model_config,
                                   source_mix={**{lang: 0.08 for lang in CAMPAIGN_LANGUAGES},
                                               "math": 0.18, "code": 0.18},
                                   name="mathheavy")
            self.assertTrue(Path(remix["path"]).exists())
            remixed = TokenManifest.load(remix["path"])
            self.assertEqual(remixed.config.source_mix["math"], 0.18)
            with self.assertRaises(FileExistsError):
                remix_manifest(out / "manifest.json", out, model_config,
                               source_mix=mix, name="mathheavy")
            with self.assertRaises(ValueError):
                remix_manifest(out / "manifest.json", out, model_config, name="empty")
