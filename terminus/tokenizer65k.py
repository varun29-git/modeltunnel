"""Frozen 65,536-piece JSON tokenizer (`tokenizer/tokenizer-65k.json`).

This module is the sole encoder/decoder for the user-selected frozen
vocabulary. The JSON artifact defines the pieces, the English byte-level BPE
merges, the Indic BPE scores, and the routing rules in its `tokenizer_spec`
block; the loader refuses to load a file whose spec disagrees with the rules
implemented here. The artifact is licensed by Sarvam AI under the Sarvam AI
Research License; it is not an official Sarvam product (see
`tokenizer/tokenizer-65k-freeze.md`).

Vocabulary layout (ids):
  0-31      specials and reserved: 0 <unk>, 1 <s>, 2 </s>, 3 <pad>, 4 <eod>,
            5-31 <reserved_0005>..<reserved_0031>. Encoding normal text never
            emits ids 0-31.
  32-287    the 256 byte-level base characters of the English path
            (`bytes_to_unicode` alphabet); id = 32 + byte value.
  288-543   byte-fallback pieces `<0x00>`..`<0xFF>`; id = 288 + byte value.
  544-40346 Indic pieces (each carries an `indic_scores` entry), with twelve
            `<reserved_space_*>` placeholders that are never emitted.
  40347-65535 English-tail pieces; merge i of `english_merges` produces the
            piece at id 40347 + i.

Document separation: packed pretraining streams use <eod> (id 4) as the
document separator; </s> (id 2) is the sentence/generation stop and <s> (1)
is an explicit begin token. This replaces the old RWKV EOS-0 convention.

Routing (per `tokenizer_spec.router`):
  * Protected whitespace (tab, newline, carriage return, or a run of two or
    more spaces; here: any maximal run of [ \\t\\n\\r] that is not exactly one
    ordinary space) takes the English path and splits text into stretches.
  * A stretch with Indic letters and no Latin letters takes the Indic path as
    one span, including digits, punctuation and single spaces.
  * A stretch with Latin letters and no Indic letters, or with neither, takes
    the English path as one segment.
  * A mixed stretch is split on single spaces into words. Within a word,
    Latin letters form English runs and Indic letters form Indic runs;
    digits, punctuation and other symbols attach to the run on their left (to
    the run on their right if the word starts with them; a word made only of
    them is English). A single space is pulled into the Indic path only when
    the runs on both sides are Indic; otherwise it is encoded on its own as
    the real-space token (id 64). Each English run and each non-pulled single
    space is encoded as its own English segment, so a space next to a Latin
    word stays a real space token.
  * A Latin letter is Unicode category Lu/Ll/Lt/Lm/Lo whose name starts with
    "LATIN ". An Indic letter is a character in the kept Indic blocks
    (Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada). U+200D/U+200C
    count as Indic when the nearest real letter (skipping those two
    characters, looking left first) is Indic; otherwise they attach as
    symbols.

Indic path: the span is normalized to NFC, a leading space-mark (U+0120) is
added as a dummy prefix, and each ordinary space becomes the space mark.
Characters map to their Indic-region piece, or (ASCII only) their byte-level
piece, or to `<0xHH>` byte-fallback pieces. BPE then repeatedly merges the
adjacent pair whose concatenation is a scored piece with the highest score;
equal scores merge the leftmost pair. Two canonicalizations keep the id
stream unambiguous for decoding (they are encoder choices, documented here):
  E1  if a span starts with a real space, that space is emitted as id 64
      before the span (the span body still gets its own dummy prefix), so the
      Indic path never emits a multi-space-mark piece;
  E2  if the dummy prefix fails to merge into the first piece, the bare
      leading space-mark piece (id 64) is dropped from the output.

English path: the segment is split into pretokens with the `english_scissors`
pattern on the raw text, each pretoken is mapped byte-by-byte through
`bytes_to_unicode`, and GPT-2-style BPE applies `english_merges` in rank
order (the result of merge i is id 40347 + i).

Decoding is region-based so that byte-level pieces reassemble exact UTF-8 and
Indic pieces keep their characters: byte-alphabet ids (32-287) and
byte-fallback ids (288-543) contribute one raw byte each; tail ids
(40347-65535) are mapped back through the inverse byte alphabet; Indic-region
ids contribute their piece text with the space mark decoded to an ordinary
space. The dummy prefix is removed by dropping the leading space mark of an
Indic-region piece when the preceding non-neutral token is not an Indic
token; byte-alphabet punctuation/digit pieces that both paths can emit are
neutral for that look-back. Ids 0-31 raise in `decode` unless
`render_specials=True`.
"""

from __future__ import annotations

import json
import re as std_re
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex

TOKENIZER_SHA256 = "c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29"

UNK_ID, BOS_ID, EOS_ID, PAD_ID, EOD_ID = 0, 1, 2, 3, 4
FIRST_REGULAR_ID = 32
BYTE_BASE = 32            # byte-alphabet pieces: id = BYTE_BASE + byte value
BYTE_FALLBACK_BASE = 288  # <0xHH> pieces: id = BYTE_FALLBACK_BASE + byte value
INDIC_BASE = 544          # first Indic-region id
ENGLISH_TAIL_BASE = 40347  # first English-merge-result id
SPACE_MARK = "Ġ"  # U+0120
KEPT_INDIC_BLOCKS = ((0x0900, 0x097F), (0x0980, 0x09FF), (0x0A80, 0x0AFF),
                     (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF))
_FORMAT_CHARS = {"‍", "‌"}  # U+200D ZWJ, U+200C ZWNJ
_WHITESPACE_RUN = std_re.compile(r"[ \t\n\r]+")

_LATIN_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo"}


def _is_latin_letter(char):
    return (unicodedata.category(char) in _LATIN_CATEGORIES
            and unicodedata.name(char, "").startswith("LATIN "))


def _is_indic_letter(char):
    codepoint = ord(char)
    return any(lo <= codepoint <= hi for lo, hi in KEPT_INDIC_BLOCKS)


def _translate_scissors(pattern):
    # The artifact writes astral escapes as \x{2019}; the `regex` engine
    # accepts \u2019. Translate without touching any other escape.
    return std_re.sub(r"\\x\{([0-9A-Fa-f]+)\}",
                      lambda match: "\\u%04x" % int(match.group(1), 16), pattern)


class Frozen65kTokenizer:
    """Encoder/decoder bound to the frozen 65,536-piece JSON artifact."""

    def __init__(self, path, *, expected_sha256=TOKENIZER_SHA256):
        from .integrity import file_sha256

        self.path = str(path)
        digest = file_sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"tokenizer sha256 {digest} does not match the frozen artifact")
        self.sha256 = digest
        with Path(path).open(encoding="utf-8") as handle:
            artifact = json.load(handle)
        self._validate_artifact(artifact)
        self.pieces = artifact["pieces"]
        self.scores = artifact["indic_scores"]
        self.bytes_to_unicode = artifact["bytes_to_unicode"]
        self.byte_of_char = {char: value for value, char in enumerate(self.bytes_to_unicode)}
        self.piece_to_id = {piece: index for index, piece in enumerate(self.pieces)}
        self.scored_id_by_piece = {piece: index for index, piece in enumerate(self.pieces)
                                   if self.scores[index] is not None}
        self.merges = [(left, right) for left, right in artifact["english_merges"]]
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.english_seeds = list(artifact["english_seeds"])
        # Seeds are raw strings (whitespace runs, LaTeX commands) whose
        # byte-mapped forms are single pieces. They are matched as longest
        # prefixes of scissors pretokens before rank-ordered BPE, because the
        # merge table reaches pieces such as newline-plus-four-spaces only
        # through a left-growing chain that rank order alone dead-ends.
        self.seed_ids = {}
        for seed in self.english_seeds:
            mapped = "".join(self.bytes_to_unicode[byte] for byte in seed.encode("utf-8"))
            self.seed_ids[seed] = self.piece_to_id[mapped]
        self.seed_prefixes = sorted(self.seed_ids, key=len, reverse=True)
        self.scissors = regex.compile(_translate_scissors(artifact["english_scissors"]))
        self._encode_english_word = lru_cache(maxsize=200_000)(self._encode_english_word_uncached)
        self._encode_indic_span = lru_cache(maxsize=100_000)(self._encode_indic_span_uncached)

    # ------------------------------------------------------------------ load
    def _validate_artifact(self, artifact):
        spec = artifact.get("tokenizer_spec")
        if not isinstance(spec, dict):
            raise ValueError("tokenizer artifact lacks tokenizer_spec")
        indic = spec.get("indic", {})
        expected_indic = {"normalization": "NFC", "dummy_prefix": True, "space_mark": SPACE_MARK,
                          "space_mark_codepoint": "U+0120",
                          "ordinary_space": "each ordinary space becomes the space mark",
                          "merge": "highest indic_score that exists as a piece merges first",
                          "equal_scores": "equal scores merge the leftmost pair",
                          "unknown_character": "a character with no piece becomes <0xHH> bytes"}
        if indic != expected_indic:
            raise ValueError(f"tokenizer_spec.indic disagrees with the implemented rules: {indic!r}")
        router = spec.get("router", {})
        for key in ("protected_whitespace", "indic_and_no_latin", "latin_and_no_indic",
                    "neither", "both_scripts", "latin_letter", "indic_letter", "zwj_zwnj",
                    "cross_script"):
            if key not in router:
                raise ValueError(f"tokenizer_spec.router is missing rule {key!r}")
        if "bengali, devanagari, gujarati, kannada, tamil, telugu" not in router["indic_letter"]:
            raise ValueError("tokenizer_spec.router.indic_letter names unexpected blocks")
        specials = spec.get("specials", {})
        if specials.get("ids_0_4") != ["<unk>", "<s>", "</s>", "<pad>", "<eod>"]:
            raise ValueError("tokenizer_spec.specials disagrees with the implemented ids 0-4")
        pieces = artifact["pieces"]
        scores = artifact["indic_scores"]
        if artifact.get("vocab_size") != 65536 or len(pieces) != 65536 or len(scores) != 65536:
            raise ValueError("tokenizer artifact is not the 65,536-piece vocabulary")
        if len(set(pieces)) != 65536:
            raise ValueError("tokenizer pieces are not unique")
        if pieces[0:5] != ["<unk>", "<s>", "</s>", "<pad>", "<eod>"]:
            raise ValueError("special ids 0-4 are not <unk> <s> </s> <pad> <eod>")
        if pieces[5:32] != [f"<reserved_{index:04d}>" for index in range(5, 32)]:
            raise ValueError("reserved ids 5-31 are missing")
        b2u = artifact["bytes_to_unicode"]
        if len(b2u) != 256 or len(set(b2u)) != 256:
            raise ValueError("bytes_to_unicode is not a 256-entry bijection")
        if pieces[32:288] != b2u:
            raise ValueError("byte-alphabet pieces do not sit at ids 32-287")
        if pieces[288:544] != [f"<0x{value:02X}>" for value in range(256)]:
            raise ValueError("byte-fallback pieces do not sit at ids 288-543")
        merges = artifact["english_merges"]
        tail = pieces[ENGLISH_TAIL_BASE:]
        if [left + right for left, right in merges] != tail:
            raise ValueError("english_merges results do not equal the English-tail pieces in order")
        if len(tail) != artifact.get("english_tail"):
            raise ValueError("english_tail count disagrees with the piece table")
        indic_region = pieces[INDIC_BASE:ENGLISH_TAIL_BASE]
        if len(indic_region) != artifact.get("K") - 256:
            raise ValueError("K disagrees with the piece table layout")
        reserved = [i for i in range(INDIC_BASE, ENGLISH_TAIL_BASE)
                    if pieces[i].startswith("<reserved_space_")]
        if any(scores[i] is not None for i in reserved):
            raise ValueError("reserved space pieces must keep empty scores")
        if any(scores[i] is None for i in range(INDIC_BASE, ENGLISH_TAIL_BASE)
               if not pieces[i].startswith("<reserved_space_")):
            raise ValueError("an Indic-region piece lacks a score")
        if any(scores[i] is not None and pieces[i].strip(SPACE_MARK) != ""
               for i in range(ENGLISH_TAIL_BASE, 65536)):
            raise ValueError("a non-whitespace English-tail piece carries an Indic score")
        if scores[64] is None or pieces[64] != SPACE_MARK:
            raise ValueError("the scored space-mark piece is missing")
        if artifact.get("indic_model") != "BPE":
            raise ValueError("indic_model must be BPE")
        normalizer = artifact.get("indic_normalizer", {})
        if (normalizer.get("form"), normalizer.get("dummy_prefix"),
                normalizer.get("space_mark")) != ("NFC", True, SPACE_MARK):
            raise ValueError("indic_normalizer disagrees with the implemented rules")
        b2u_set = set(b2u)
        if any(char not in b2u_set for piece in tail for char in piece):
            raise ValueError("an English-tail piece contains a non-byte-alphabet character")
        if any(char != SPACE_MARK and char in b2u_set and ord(char) >= 128
               for i in range(INDIC_BASE, ENGLISH_TAIL_BASE)
               for char in pieces[i] if not pieces[i].startswith("<reserved_space_")):
            raise ValueError("an Indic piece contains a non-ASCII byte-alphabet character")

    # ------------------------------------------------------------- specials
    def get_piece_size(self):
        return 65536

    def unk_id(self):
        return UNK_ID

    def bos_id(self):
        return BOS_ID

    def eos_id(self):
        """Packed-stream document separator: <eod> (id 4), not RWKV EOS 0."""
        return EOD_ID

    def eod_id(self):
        return EOD_ID

    def pad_id(self):
        return PAD_ID

    def sentence_end_id(self):
        return EOS_ID

    def is_byte(self, token_id):
        return BYTE_FALLBACK_BASE <= token_id < BYTE_FALLBACK_BASE + 256

    def special_id(self, name):
        mapping = {"<unk>": UNK_ID, "<s>": BOS_ID, "</s>": EOS_ID, "<pad>": PAD_ID,
                   "<eod>": EOD_ID}
        return mapping[name]

    # ------------------------------------------------------------------ router
    @staticmethod
    def _letter_classes(stretch):
        """Per-character script class: 'L' Latin, 'I' Indic, 'O' other."""
        classes = []
        for index, char in enumerate(stretch):
            if _is_latin_letter(char):
                classes.append("L")
            elif _is_indic_letter(char):
                classes.append("I")
            elif char in _FORMAT_CHARS:
                neighbor = None
                for cursor in range(index - 1, -1, -1):
                    if stretch[cursor] not in _FORMAT_CHARS:
                        neighbor = stretch[cursor]
                        break
                if neighbor is None or not _is_indic_letter(neighbor):
                    for cursor in range(index + 1, len(stretch)):
                        if stretch[cursor] not in _FORMAT_CHARS:
                            neighbor = stretch[cursor]
                            break
                classes.append("I" if neighbor is not None and _is_indic_letter(neighbor) else "O")
            else:
                classes.append("O")
        return classes

    def _route_mixed_stretch(self, stretch, classes):
        """Split a both-scripts stretch into (kind, text) units."""
        units = []
        cursor = 0
        for word_index, word in enumerate(stretch.split(" ")):
            if word_index > 0:
                units.append(("space", " "))
            cursor += word_index > 0  # account for the separating single space
            if word == "":
                continue
            word_classes = classes[cursor:cursor + len(word)]
            cursor += len(word)
            runs = []
            for char, cls in zip(word, word_classes):
                kind = "indic" if cls == "I" else ("english" if cls == "L" else None)
                if kind is None:
                    if runs:
                        runs[-1][1] += char  # digits/punct/symbols attach left
                    else:
                        runs.append(["pending", char])
                else:
                    if runs and runs[-1][0] == "pending":
                        runs[-1][0] = kind  # leading symbols attach right
                    if runs and runs[-1][0] == kind:
                        runs[-1][1] += char
                    else:
                        runs.append([kind, char])
            if runs and runs[0][0] == "pending":
                runs[0][0] = "english"  # a word made only of symbols is English
            units.extend((kind, text) for kind, text in runs)
        return units

    def _route(self, text):
        """Yield (path, segment) pairs: ('indic'|'english', str)."""
        # Split into stretches on protected whitespace runs; single spaces
        # stay inside stretches.
        items = []
        pending = None
        cursor = 0
        for match in _WHITESPACE_RUN.finditer(text):
            if match.start() > cursor:
                pending = (pending or "") + text[cursor:match.start()]
            run = match.group(0)
            if run == " ":
                pending = (pending or "") + run
            else:
                if pending is not None:
                    items.append(("stretch", pending))
                    pending = None
                items.append(("english", run))
            cursor = match.end()
        if cursor < len(text):
            pending = (pending or "") + text[cursor:]
        if pending is not None:
            items.append(("stretch", pending))

        for item_kind, item in items:
            if item_kind == "english":
                yield ("english", item)
                continue
            stretch = item
            classes = self._letter_classes(stretch)
            has_indic = "I" in classes
            has_latin = "L" in classes
            if has_indic and not has_latin:
                yield ("indic", stretch)
            elif not has_indic:
                yield ("english", stretch)
                continue
            else:
                units = self._route_mixed_stretch(stretch, classes)
                # Pull single spaces between two Indic runs into the Indic span.
                merged = []
                for index, (kind, value) in enumerate(units):
                    if (kind == "space" and merged and merged[-1][0] == "indic"
                            and index + 1 < len(units) and units[index + 1][0] == "indic"):
                        merged[-1][1] += value
                    elif kind == "indic" and merged and merged[-1][0] == "indic":
                        merged[-1][1] += value
                    elif kind == "space":
                        merged.append(["english", " "])
                    else:
                        merged.append([kind, value])
                for kind, value in merged:
                    yield ("indic" if kind == "indic" else "english", value)

    # ------------------------------------------------------------------ encode
    def encode(self, text, out_type=int):
        if out_type is not int:
            raise ValueError("tokenization returns integer IDs")
        if not isinstance(text, str):
            raise ValueError("encode expects a string")
        ids = []
        for path, segment in self._route(text):
            if path == "indic":
                if segment.startswith(" "):  # E1: leading real space stands alone
                    ids.append(BYTE_BASE + 0x20)
                    segment = segment[1:]
                    if not segment:
                        continue
                ids.extend(self._encode_indic_span(segment))
            else:
                ids.extend(self._encode_english_segment(segment))
        if any(token < FIRST_REGULAR_ID for token in ids):
            raise ValueError("normal text must not emit ids 0-31")
        return ids

    def _encode_english_segment(self, segment):
        ids = []
        for word in self.scissors.findall(segment):
            ids.extend(self._encode_english_word(word))
        return ids

    def _encode_english_word_uncached(self, word):
        symbols = []  # strings over the byte alphabet
        rest = word
        while rest and rest[0] in " \t\n\r\\":
            for seed in self.seed_prefixes:
                if rest.startswith(seed):
                    symbols.append("".join(self.bytes_to_unicode[byte]
                                           for byte in seed.encode("utf-8")))
                    rest = rest[len(seed):]
                    break
            else:
                break
        symbols.extend(self.bytes_to_unicode[byte] for byte in rest.encode("utf-8"))
        while len(symbols) > 1:
            best_rank, best_index = None, None
            for index in range(len(symbols) - 1):
                rank = self.merge_rank.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_rank is None:
                break
            symbols[best_index:best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        ids = []
        for symbol in symbols:
            byte = self.byte_of_char.get(symbol)
            if byte is not None:
                ids.append(BYTE_BASE + byte)
            else:
                ids.append(ENGLISH_TAIL_BASE + self.merge_rank_piece(symbol))
        return ids

    def merge_rank_piece(self, piece):
        # A merged English symbol's id is 40347 + the rank of the merge that
        # produced it; the piece table is exactly the merge results in order.
        index = self.piece_to_id.get(piece)
        if index is None or index < ENGLISH_TAIL_BASE:
            raise ValueError(f"English BPE produced a non-piece {piece!r}")
        return index - ENGLISH_TAIL_BASE

    def _encode_indic_span_uncached(self, span):
        transformed = SPACE_MARK + unicodedata.normalize("NFC", span).replace(" ", SPACE_MARK)
        symbols = []  # (text, id) pairs; byte-fallback text cannot merge
        for char in transformed:
            if char == SPACE_MARK:
                symbols.append((SPACE_MARK, BYTE_BASE + 0x20))
                continue
            piece_id = self.piece_to_id.get(char)
            if piece_id is not None and INDIC_BASE <= piece_id < ENGLISH_TAIL_BASE:
                symbols.append((char, piece_id))
            elif ord(char) < 128:
                symbols.append((char, BYTE_BASE + ord(char)))
            else:
                for byte in char.encode("utf-8"):
                    symbols.append((f"<0x{byte:02X}>", BYTE_FALLBACK_BASE + byte))
        while True:
            best_score, best_index = None, None
            for index in range(len(symbols) - 1):
                merged = self.scored_id_by_piece.get(symbols[index][0] + symbols[index + 1][0])
                if merged is None:
                    continue
                score = self.scores[merged]
                if best_score is None or score > best_score:
                    best_score, best_index = score, index
            if best_score is None:
                break
            left, right = symbols[best_index:best_index + 2]
            symbols[best_index:best_index + 2] = [(left[0] + right[0],
                                                   self.scored_id_by_piece[left[0] + right[0]])]
        if len(symbols) > 1 and symbols[0][1] == BYTE_BASE + 0x20:
            symbols.pop(0)  # E2: drop an unmerged dummy prefix
        return [token_id for _, token_id in symbols]

    # ------------------------------------------------------------------ decode
    def _decode_class(self, token):
        """'indic' | 'english' | 'neutral' for the dummy-prefix look-back.

        Indic-region and byte-fallback ids are only emitted by the Indic path.
        Unscored byte-alphabet ids and English-tail ids are only emitted by the
        English path (encoder canonicalizations E1/E2 keep the multi-space-mark
        tail pieces English-only). Scored byte-alphabet ids (digits,
        punctuation, the bare space mark) can be emitted by either path, so
        they are neutral and the look-back skips them.
        """
        if BYTE_FALLBACK_BASE <= token < BYTE_FALLBACK_BASE + 256:
            return "indic"
        if INDIC_BASE <= token < ENGLISH_TAIL_BASE:
            return "indic"
        if BYTE_BASE <= token < BYTE_BASE + 256 and self.scores[token] is not None:
            return "neutral"
        return "english"

    def decode(self, tokens, *, render_specials=False):
        tokens = list(tokens)
        classes = [self._decode_class(token) if FIRST_REGULAR_ID <= token < 65536 else None
                   for token in tokens]
        output = bytearray()
        for index, token in enumerate(tokens):
            if not 0 <= token < 65536:
                raise ValueError(f"token id {token} is out of range")
            if token < FIRST_REGULAR_ID:
                if not render_specials:
                    raise ValueError(f"cannot decode special/reserved id {token}")
                output.extend(self.pieces[token].encode("utf-8"))
                continue
            if BYTE_BASE <= token < BYTE_BASE + 256:
                output.append(token - BYTE_BASE)
                continue
            if BYTE_FALLBACK_BASE <= token < BYTE_FALLBACK_BASE + 256:
                output.append(token - BYTE_FALLBACK_BASE)
                continue
            piece = self.pieces[token]
            if token >= ENGLISH_TAIL_BASE:
                for char in piece:
                    output.append(self.byte_of_char[char])
                continue
            # Indic-region piece: drop the dummy prefix when the previous
            # non-neutral token is not Indic (span start); keep real spaces.
            if piece.startswith(SPACE_MARK) and len(piece) > 1:
                cursor = index - 1
                while cursor >= 0 and classes[cursor] == "neutral":
                    cursor -= 1
                if cursor < 0 or classes[cursor] != "indic":
                    piece = piece[1:]
            for char in piece:
                if char == SPACE_MARK:
                    output.append(0x20)
                else:
                    output.extend(char.encode("utf-8"))
        return output.decode("utf-8")


def load_frozen_tokenizer(path, *, expected_sha256=TOKENIZER_SHA256):
    return Frozen65kTokenizer(path, expected_sha256=expected_sha256)
