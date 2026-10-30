# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests that run offline (no downloads)."""
from __future__ import annotations

import numpy as np
import pytest

from brahmai_tokenizer_eval import metrics as M
from brahmai_tokenizer_eval import probes as P
from brahmai_tokenizer_eval.registry import CORPORA, SUITES
from brahmai_tokenizer_eval.tokenizers import load


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A small byte-level BPE tokenizer trained on a few sentences."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    text = ["the cat sat on the mat", "नमस्ते दुनिया", "x = 3.14 + 2", "the dog ate the bone"] * 50
    tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=400,
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    p = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tk.save(str(p))
    return load(str(p), name="tiny")


def test_registry_integrity():
    assert len(CORPORA) == len({c.id for c in CORPORA.values()})
    for c in CORPORA.values():
        assert c.licence, c.id
    for group in {c.parallel for c in CORPORA.values() if c.parallel in ("in22", "flores")}:
        assert any(c.parallel == group and c.lang == "eng_Latn" for c in CORPORA.values())
    for ids in SUITES.values():
        assert all(i in CORPORA for i in ids)


def test_round_trip_and_bytes(tiny):
    s = "the cat ate नमस्ते 42"
    assert tiny.decode(tiny.encode(s)) == s
    table, special = tiny.token_bytes()
    assert len({b for b in table.values() if len(b) == 1}) == 256
    assert P.missing_byte_symbols(tiny) == []


def test_offsets_cover_text(tiny):
    s = "the dog sat"
    ids, spans = tiny.encode_offsets(s)
    assert len(ids) == len(spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(s)


def _counts():
    rng = np.random.default_rng(0)
    b = rng.integers(50, 200, 300)
    return M.Counts("test", b, b, b // 5, {"a": b // 4, "b": b // 2}, {"a": [], "b": []})


def test_metric_definitions():
    c = _counts()
    s = M.summary(c, "a")
    est, lo, hi = s["tokens_per_kb"]
    assert lo <= est <= hi
    assert abs(est - 1000 * c.tokens["a"].sum() / c.bytes.sum()) < 1e-9
    assert abs(s["bytes_per_token"] * est / 1000 - 1) < 1e-9


def test_paired_ratio_and_premium():
    c = _counts()
    r, lo, hi = M.paired_ratio(c, "a", "a")
    assert r == lo == hi == 1.0
    p, lo, hi = M.premium(c, c, "b")
    assert p == 1.0
    with pytest.raises(ValueError):
        short = M.Counts("s", c.bytes[:10], c.chars[:10], c.words[:10], {"b": c.tokens["b"][:10]}, {})
        M.premium(short, c, "b")


def test_byte_and_group_premium():
    c = _counts()
    assert M.byte_premium(c, c)[0] == 1.0
    dbl = M.Counts("d", 2 * c.bytes, c.chars, c.words, {"b": 3 * c.tokens["b"]}, {})
    assert abs(M.byte_premium(dbl, c)[0] - 2.0) < 1e-12
    # token premium = tokens-per-byte ratio x byte premium on parallel text
    tpb = (dbl.tokens["b"].sum() / dbl.bytes.sum()) / (c.tokens["b"].sum() / c.bytes.sum())
    assert abs(M.premium(dbl, c, "b")[0] - tpb * M.byte_premium(dbl, c)[0]) < 1e-12
    p, lo, hi = M.group_premium([c, dbl], c, "b")
    assert abs(p - 2.0) < 1e-12 and lo <= p <= hi
    with pytest.raises(ValueError):
        M.group_premium([], c, "b")


def test_probes_run(tiny):
    r = P.reachability(tiny)
    assert r["decodable_entries"] > 0
    n = P.numbers(tiny, ["In 1947 there were 3.5 million"])
    assert 0 <= n["integers_0_9999"]["share_single_digit"] <= 1
    comp = P.composition(tiny)
    assert comp["ordinary_entries"] > 256


def test_efficiency_index_population_independent():
    c = _counts()
    one = M.efficiency_index({"g1": [c], "g2": [c]}, "a", "b")
    assert abs(one["index"] - c.tokens["b"].sum() / c.tokens["a"].sum()) < 1e-9
    c2 = M.Counts("t", c.bytes, c.chars, c.words, {**c.tokens, "z": c.tokens["a"] * 3}, {})
    two = M.efficiency_index({"g1": [c2], "g2": [c2]}, "a", "b")
    assert one["index"] == two["index"]   # adding a third tokenizer changes nothing
