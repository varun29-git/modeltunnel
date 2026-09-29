import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from terminus.integrity import LANGUAGES, validate_manifest
from terminus.prepare_data import prepare_data
from terminus.tokenizer import encode_text, load_tokenizer, tokenizer_contract
from tests.helpers import tiny_config


class RWKVWorldTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.vocab = cls.root / "rwkv_vocab.txt"
        with cls.vocab.open("w", encoding="utf-8") as handle:
            for token_id in range(1, 65530):
                if token_id <= 256:
                    piece = bytes((token_id - 1,))
                elif token_id == 257:
                    piece = b"ab"
                elif token_id == 258:
                    piece = b"abc"
                else:
                    piece = b"\xff" + token_id.to_bytes(3, "big")
                handle.write(f"{token_id} {piece!r} {len(piece)}\n")
        cls.tokenizer = load_tokenizer(cls.vocab)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_longest_prefix_unicode_and_reserved_slots(self):
        self.assertEqual(encode_text(self.tokenizer, "abcab"), [258, 257])
        sample = "  नमस्ते 🙂\n"
        ids = encode_text(self.tokenizer, sample)
        self.assertEqual(self.tokenizer.decode(ids), sample)
        self.assertEqual(self.tokenizer.decode([0, *ids]), sample)
        self.assertEqual(self.tokenizer.get_piece_size(), 65536)
        contract = tokenizer_contract(self.tokenizer, self.vocab)
        self.assertEqual((contract["eos_token_id"], contract["pad_token_id"]), (0, None))
        with self.assertRaisesRegex(ValueError, "not a vocabulary piece"):
            self.tokenizer.decode([65535])

    def test_preparation_preserves_eos_and_uint16_identity(self):
        corpus = self.root / "corpus.jsonl"
        with corpus.open("w", encoding="utf-8") as handle:
            for language in LANGUAGES:
                for split in ("train", "val"):
                    handle.write(json.dumps({"language": language, "split": split,
                                             "text": f"{language} abcab {split} distinct text",
                                             "source": language, "license": "test-only"}) + "\n")
        mix = {language: (.1 if language == "en" else .09) for language in LANGUAGES}
        output = self.root / "prepared"
        manifest = prepare_data(corpus, self.vocab, output, source_mix=mix,
                                total_train_tokens=88, seq_len=8)
        self.assertEqual(manifest.tokenizer["path"], "tokenizer.txt")
        self.assertEqual(manifest.config.token_dtype, "uint16")
        for shard in manifest.shards:
            self.assertEqual(int(np.fromfile(output / shard.path, dtype=np.uint16)[-1]), 0)
        validate_manifest(manifest, output, tiny_config(vocab_size=65536, pad_token_id=None, max_seq_len=8))


if __name__ == "__main__":
    unittest.main()
