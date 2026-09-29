"""Gate tests for the frozen 65,536-piece tokenizer integration.

Reproduces the applicable checks from tokenizer/tokenizer-65k-freeze.md:
- Gate 1 examples (piece-string expectations stated in the freeze note; the
  700-sentence parity run against the original Sarvam model is NOT reproduced
  here because `tokenizer.model` is not in this repository).
- Gate 2 whitespace/LaTeX tokenization.
- Gate 4 broken-UTF-8 and dependent-vowel-mark counts.
- Gate 5 round-trip across the eight campaign languages, math, and code.
- Gate 7 cross-script piece count.
Gate 3 fertility is measured on real held-out text by the data pipeline
receipts (docs), not asserted here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

from terminus.integrity import (CAMPAIGN_DOMAINS, CAMPAIGN_LANGUAGES, CAMPAIGN_SCOPE_V1,
                                validate_manifest)
from terminus.prepare_data import prepare_data
from terminus.tokenizer import encode_text, load_tokenizer, tokenizer_contract
from terminus.tokenizer65k import (BOS_ID, EOD_ID, EOS_ID, PAD_ID, UNK_ID,
                                   Frozen65kTokenizer, TOKENIZER_SHA256)
from tests.helpers import tiny_config

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer-65k.json"

SAMPLES = {
    "en": "The quick brown fox jumps over the lazy dog in 2024.",
    "hi": "सूर्योदय के समय गाँव के लोग खेतों में काम करने जाते हैं।",
    "mr": "महाराष्ट्रातील ग्रामीण भागात शेती हा प्रमुख व्यवसाय आहे.",
    "kn": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಭಾರತದ ಮುಖ್ಯ ನಗರಗಳಲ್ಲಿ ಒಂದು.",
    "te": "తెలంగాణ రాష్ట్ర రాజధాని హైదరాబాద్ చారిత్రక నగరం.",
    "ta": "தமிழ்நாட்டின் தலைநகரம் சென்னை, இது ஒரு பெரிய நகரம்.",
    "gu": "ગુજરાતની સંસ્કૃતિ અને પરંપરા ખૂબ જ સમૃદ્ધ છે.",
    "bn": "পশ্চিমবঙ্গের রাজধানী কলকাতা একটি বড় শহর।",
    "math": "If $x^2 + y^2 = z^2$ then \\sqrt{x} \\leq \\frac{z}{\\pi} for x, y, z > 0.",
    "code": "def f(x):\n    return x * 2  # double\n",
}


class FrozenTokenizerGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Frozen65kTokenizer(TOKENIZER_PATH)

    def pieces_of(self, text):
        return [self.tok.pieces[i] for i in self.tok.encode(text)]

    def test_identity_and_special_ids(self):
        self.assertEqual(self.tok.sha256, TOKENIZER_SHA256)
        self.assertEqual(self.tok.get_piece_size(), 65536)
        self.assertEqual((self.tok.unk_id(), self.tok.bos_id(), self.tok.sentence_end_id(),
                          self.tok.pad_id(), self.tok.eos_id()),
                         (UNK_ID, BOS_ID, EOS_ID, PAD_ID, EOD_ID))
        self.assertEqual((UNK_ID, BOS_ID, EOS_ID, PAD_ID, EOD_ID), (0, 1, 2, 3, 4))
        # The packed-stream document separator is <eod> (4), not RWKV EOS 0.
        self.assertEqual(self.tok.eos_id(), 4)

    def test_sha256_mismatch_refused(self):
        with self.assertRaises(ValueError):
            Frozen65kTokenizer(TOKENIZER_PATH, expected_sha256="0" * 64)

    def test_spec_block_disagreement_refused(self):
        artifact = json.loads(TOKENIZER_PATH.read_text())
        for mutate in (
                lambda a: a["tokenizer_spec"]["indic"].update(dummy_prefix=False),
                lambda a: a["tokenizer_spec"]["specials"].update(ids_0_4=["<unk>"] * 5),
                lambda a: a.update(vocab_size=65537),
                lambda a: a["pieces"].__setitem__(40, "<0x00>"),
                lambda a: a.update(english_tail=25188),
                lambda a: a["tokenizer_spec"].pop("router")):
            mutated = json.loads(TOKENIZER_PATH.read_text())
            mutate(mutated)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                json.dump(mutated, handle)
                path = handle.name
            with self.assertRaises(ValueError):
                Frozen65kTokenizer(path, expected_sha256=None)

    def test_freeze_note_piece_examples(self):
        self.assertEqual(self.pieces_of("नमस्ते"), ["Ġनम", "स्ते"])
        self.assertEqual(self.pieces_of("हिन्दी"), ["Ġह", "िन्", "दी"])
        self.assertEqual(self.pieces_of("ಕನ್ನಡ"), ["Ġಕ", "ನ್ನ", "ಡ"])

    def test_mixed_routing_example_from_freeze_note(self):
        pieces = self.pieces_of("main जा रहा hoon")
        self.assertEqual(pieces[0], "main")
        self.assertEqual(pieces[1], "Ġ")  # a real space next to a Latin word
        self.assertEqual("".join(pieces[2:4]), "ĠजाĠरहा")  # one Indic span
        self.assertEqual(pieces[4], "Ġ")
        self.assertEqual("".join(pieces[5:]), "hoon")

    def test_gate2_whitespace(self):
        for text in ("\n", "\n  ", "\n    ", "\n\t"):
            ids = self.tok.encode(text)
            self.assertEqual(len(ids), 1, text)
            self.assertEqual(self.tok.decode(ids), text)
        self.assertEqual([self.tok.pieces[i] for i in self.tok.encode("\n    return")],
                         ["ĊĠĠĠĠ", "return"])

    def test_gate2_latex_single_tokens(self):
        for command in ("\\frac", "\\partial", "\\circ", "\\overrightarrow", "\\subseteq",
                        "\\setminus", "\\text", "\\pi", "\\cdot", "\\sin", "\\cos", "\\int",
                        "\\times", "\\left", "\\right", "\\quad", "\\sqrt", "\\alpha",
                        "\\equiv", "\\theta", "\\varphi", "\\beta", "\\begin", "\\end",
                        "\\binom", "\\Gamma", "\\to", "\\pmod", "\\sigma", "\\mathbb",
                        "\\epsilon", "\\Delta", "\\geq", "\\sum", "\\lim", "\\ln",
                        "\\infty", "\\pm", "\\tan", "\\Rightarrow", "\\omega", "\\approx",
                        "\\mod"):
            with self.subTest(command=command):
                self.assertEqual(len(self.tok.encode(command)), 1, command)

    def test_gate4_piece_health(self):
        # No piece is broken UTF-8: every non-special piece decodes.
        for index, piece in enumerate(self.tok.pieces):
            if index < 32 or piece.startswith("<"):
                continue
            piece.encode("utf-8")  # surrogates would raise here
        vowel_marks = sum(
            1 for index in range(544, 40347)
            if self.tok.scores[index] is not None
            and "VOWEL SIGN" in unicodedata.name(self.tok.pieces[index][0], ""))
        self.assertEqual(vowel_marks, 2702)

    def test_gate7_no_cross_script_pieces(self):
        def scripts(piece):
            found = set()
            for char in piece:
                codepoint = ord(char)
                if char in ("Ġ", "Ċ") or codepoint < 128:
                    continue
                if 0x0900 <= codepoint <= 0x0CFF:
                    found.add("indic")
                elif unicodedata.category(char).startswith("L"):
                    found.add("other-letter")
            return found
        for index in range(544, 40347):
            piece = self.tok.pieces[index]
            if not piece.startswith("<reserved"):
                self.assertLessEqual(len(scripts(piece)), 1, piece)

    def test_gate5_roundtrip_all_languages_and_domains(self):
        cases = list(SAMPLES.values()) + [
            "main जा रहा hoon", "वर्ष 2024 में 5,000 लोग", "a2क", "क a2 क",
            "क्‍क", "a‍b", "x{क", "é", "कéक", "2क", " जा", "जा ", "a  b",
            "\n\n\t  x", "यहाँ English और हिन्दी साथ में।", "😀 क 🙂",
            "3.14159 ≈ \\pi", "for i in range(10):\n\tprint(i)",
        ]
        for text in cases:
            with self.subTest(text=text):
                normalized = unicodedata.normalize("NFC", text)
                self.assertEqual(self.tok.decode(self.tok.encode(normalized)), normalized)

    def test_normal_text_never_emits_special_ids(self):
        for text in list(SAMPLES.values()) + ["<unk> <s> </s> <pad> <eod>", "reserved"]:
            ids = self.tok.encode(text)
            self.assertTrue(all(token >= 32 for token in ids), text)
        # The literal strings "<s>" etc. are text, not special tokens.
        self.assertEqual(self.tok.decode(self.tok.encode("<s> </s>")), "<s> </s>")

    def test_encode_is_deterministic(self):
        for text in SAMPLES.values():
            self.assertEqual(self.tok.encode(text), self.tok.encode(text))

    def test_decode_rejects_special_ids_unless_rendered(self):
        with self.assertRaises(ValueError):
            self.tok.decode([EOD_ID, 100])
        self.assertEqual(self.tok.decode([EOD_ID], render_specials=True), "<eod>")
        with self.assertRaises(ValueError):
            self.tok.decode([65536])

    def test_encode_text_wrapper_and_contract(self):
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        self.assertIsInstance(tokenizer, Frozen65kTokenizer)
        contract = tokenizer_contract(tokenizer, TOKENIZER_PATH)
        self.assertEqual(contract["sha256"], TOKENIZER_SHA256)
        self.assertEqual(contract["vocab_size"], 65536)
        self.assertEqual(contract["eos_token_id"], 4)
        self.assertEqual(contract["pad_token_id"], 3)
        self.assertEqual(contract["normalization"], "NFC")
        for text in SAMPLES.values():
            encode_text(tokenizer, text)  # raises unless round-trip is exact

    def test_byte_fallback_never_needed_for_campaign_text(self):
        for language, text in SAMPLES.items():
            ids = self.tok.encode(unicodedata.normalize("NFC", text))
            self.assertEqual(sum(self.tok.is_byte(i) for i in ids), 0, language)

    def test_uncovered_letter_uses_byte_fallback_and_roundtrips(self):
        # U+0990 BENGALI LETTER AI has no piece in the frozen vocabulary; it
        # must fall back to <0xHH> bytes and still round-trip exactly.
        text = "ঐতিহাসিক"
        ids = self.tok.encode(text)
        self.assertTrue(any(self.tok.is_byte(token) for token in ids))
        self.assertEqual(self.tok.decode(ids), text)


class FrozenTokenizerDataIntegrationTests(unittest.TestCase):
    def _corpus(self, root, languages):
        lines = []
        for language in languages:
            for split in ("train", "val"):
                for index in range(6):
                    text = SAMPLES[language] if split == "train" else SAMPLES[language] + "!"
                    lines.append({"language": language, "text": text * (index + 3),
                                  "split": split, "source": "synthetic_fixture",
                                  "license": "test-only"})
        corpus = Path(root) / "corpus.jsonl"
        corpus.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines))
        return corpus

    def test_prepare_data_with_frozen_tokenizer_and_campaign_scope(self):
        with tempfile.TemporaryDirectory() as root:
            corpus = self._corpus(root, CAMPAIGN_LANGUAGES)
            scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
                     "domains": []}
            output = Path(root) / "data"
            manifest = prepare_data(corpus, TOKENIZER_PATH, output,
                                    source_mix={language: 0.125 for language in CAMPAIGN_LANGUAGES},
                                    total_train_tokens=1024, seq_len=16, batch_size=1,
                                    shard_tokens=100_000, scope=scope)
            self.assertEqual(manifest.tokenizer["path"], "tokenizer.json")
            self.assertEqual(manifest.tokenizer["sha256"], TOKENIZER_SHA256)
            self.assertEqual(manifest.provenance["document_separator_id"], 4)
            self.assertEqual(manifest.provenance["scope"]["name"], CAMPAIGN_SCOPE_V1)
            # Every (language, split) stream ends with the <eod> separator.
            import numpy as np
            for spec in manifest.shards:
                tokens = np.fromfile(output / spec.path, dtype=np.uint16)
                self.assertEqual(int(tokens[-1]), 4)
            report = validate_manifest(manifest, output,
                                       tiny_config(vocab_size=65536, pad_token_id=3,
                                                   max_seq_len=16))
            self.assertEqual(set(report["tokens_by_language"]["train"]),
                             set(CAMPAIGN_LANGUAGES))

    def test_legacy_manifest_still_requires_eleven_languages(self):
        with tempfile.TemporaryDirectory() as root:
            corpus = self._corpus(root, CAMPAIGN_LANGUAGES)
            output = Path(root) / "data"
            manifest = prepare_data(corpus, TOKENIZER_PATH, output,
                                    source_mix={language: 0.125 for language in CAMPAIGN_LANGUAGES},
                                    total_train_tokens=1024, seq_len=16, batch_size=1,
                                    shard_tokens=100_000)
            with self.assertRaisesRegex(ValueError, "missing language"):
                validate_manifest(manifest, output,
                                  tiny_config(vocab_size=65536, pad_token_id=3,
                                              max_seq_len=16))

    def test_scope_with_domain_requires_domain_source(self):
        with tempfile.TemporaryDirectory() as root:
            corpus = self._corpus(root, CAMPAIGN_LANGUAGES)
            scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
                     "domains": ["math"]}
            output = Path(root) / "data"
            manifest = prepare_data(corpus, TOKENIZER_PATH, output,
                                    source_mix={language: 0.125 for language in CAMPAIGN_LANGUAGES},
                                    total_train_tokens=1024, seq_len=16, batch_size=1,
                                    shard_tokens=100_000, scope=scope)
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_manifest(manifest, output,
                                  tiny_config(vocab_size=65536, pad_token_id=3,
                                              max_seq_len=16))


if __name__ == "__main__":
    unittest.main()
