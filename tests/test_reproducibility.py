# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for manifests, provenance, round-trip classification, group summaries and compare."""
from __future__ import annotations

import json

import numpy as np
import pytest

from brahmai_tokenizer_eval import manifest as MF
from brahmai_tokenizer_eval import metrics as M
from brahmai_tokenizer_eval import provenance
from brahmai_tokenizer_eval import report as R
from brahmai_tokenizer_eval.tokenizers import load

H40, H64 = "a" * 40, "b" * 64


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(["the cat sat on the mat", "नमस्ते दुनिया"] * 50,
                           trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    p = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tk.save(str(p))
    return p


def _write(tmp_path, body: str):
    p = tmp_path / "m.toml"
    p.write_text(body)
    return p


# ── manifests ─────────────────────────────────────────────────────────────────

def test_manifest_requires_pins(tmp_path):
    for bad in (f'[[tokenizer]]\nname="x"\nhub="o/r"\nrevision="main"\nsha256="{H64}"\n',      # branch, not commit
                f'[[tokenizer]]\nname="x"\nhub="o/r"\nrevision="{H40}"\n',                     # no hash
                '[[tokenizer]]\nname="x"\npath="t.json"\n',                                       # no hash
                '[[tokenizer]]\nname="x"\ntiktoken="o200k_base"\n',                              # no fingerprint
                f'[[tokenizer]]\nname="x"\nhub="o/r"\npath="t"\nrevision="{H40}"\nsha256="{H64}"\n'):
        with pytest.raises(MF.ManifestError):
            MF.load_manifest(_write(tmp_path, '[run]\nsuite="quick"\n' + bad))


def test_manifest_rejects_unknown_corpora_and_reference(tmp_path):
    t = f'[[tokenizer]]\nname="x"\nhub="o/r"\nrevision="{H40}"\nsha256="{H64}"\n'
    with pytest.raises(MF.ManifestError):
        MF.load_manifest(_write(tmp_path, '[run]\ncorpora=["nope/xx"]\n' + t))
    with pytest.raises(MF.ManifestError):
        MF.load_manifest(_write(tmp_path, '[run]\nsuite="quick"\nreference="y"\n' + t))
    with pytest.raises(MF.ManifestError):   # group member not in the run
        MF.load_manifest(_write(tmp_path, '[run]\ncorpora=["in22/hin_Deva"]\n' + t + '[groups]\ng=["in22/tam_Taml"]\n'))


def test_manifest_parses_and_verifies(tmp_path, tok_path):
    sha = provenance.sha256_file(tok_path)
    m = MF.load_manifest(_write(tmp_path, f'[run]\nsuite="quick"\nprobes="all"\n'
                                          f'[[tokenizer]]\nname="tiny"\npath="{tok_path.as_posix()}"\nsha256="{sha}"\n'
                                          f'[groups]\nIndic=["in22/hin_Deva","in22/tam_Taml"]\n'))
    assert m.reference == "tiny" and m.probes == ["tiny"] and len(m.sha256) == 64
    t = load(m.tokenizers[0].spec, name="tiny")
    MF.verify(t, m.tokenizers[0])                                       # matches
    with pytest.raises(MF.ManifestError):
        MF.verify(t, MF.TokenizerPin("tiny", str(tok_path), sha256=H64))  # file hash differs
    with pytest.raises(MF.ManifestError):
        MF.verify(t, MF.TokenizerPin("tiny", str(tok_path), fingerprint=H64))


# ── provenance ────────────────────────────────────────────────────────────────

def test_provenance_records_versions():
    r = provenance.record(["bte", "run"])
    assert r["packages"]["tokenizers"] and r["packages"]["numpy"]
    assert r["argv"] == ["bte", "run"] and "suite_commit" in r and "suite_dirty" in r


# ── round trip ────────────────────────────────────────────────────────────────

def test_roundtrip_kind():
    nfd = "é"                      # e + combining acute; NFC is "é"
    assert M.roundtrip_kind(nfd, "é") == "nfc"
    assert M.roundtrip_kind("abc", " abc") == "leading_space"
    assert M.roundtrip_kind("abc", "ab") == "lossy"
    assert M.roundtrip_kind("שלום", "") == "lossy"
    assert M.byte_diff("abc", " abc") == 0 and M.byte_diff("שלום", "") == 8 and M.byte_diff(" a", " a") == 0


def test_count_is_deterministic_and_classifies(tok_path, monkeypatch):
    t = load(str(tok_path), name="tiny")
    texts = ["the cat", "नमस्ते", "the mat sat"]
    monkeypatch.setattr(M.C, "load", lambda cid, cache=None: texts)
    a, b = M.count("x", [t]), M.count("x", [t])
    assert np.array_equal(a.tokens["tiny"], b.tokens["tiny"])
    assert a.roundtrip_failures["tiny"] == [] and a.lossy("tiny") == 0
    back = M.Counts.from_json(json.loads(json.dumps(a.to_json())))
    assert back.roundtrip_kinds == a.roundtrip_kinds and back.roundtrip_byte_diff == a.roundtrip_byte_diff


# ── group summaries ───────────────────────────────────────────────────────────

def _member(n, ra, rb, seed):
    rng = np.random.default_rng(seed)
    b = rng.integers(100, 200, n)
    return M.Counts("m", b, b, b // 5, {"a": (b * ra).astype(int), "b": (b * rb).astype(int)}, {})


def test_group_compare_pooled_vs_geomean():
    # eleven members where a == b, one where a needs 5x the tokens of b
    members = {f"l{i}": _member(200, 1.0, 1.0, i) for i in range(11)}
    members["urd"] = _member(200, 5.0, 1.0, 99)
    r = M.group_compare(members, "a", "b")
    pool = M.merge(list(members.values()))
    assert abs(r["pooled"][0] - pool.tokens["a"].sum() / pool.tokens["b"].sum()) < 1e-12
    assert abs(r["geomean"][0] - 5 ** (1 / 12)) < 1e-3
    assert r["pooled"][0] > r["geomean"][0] > r["median"]           # one member drives the pooled value
    assert r["max"][0] == "urd" and abs(r["median"] - 1.0) < 1e-9
    for k in ("pooled", "geomean"):
        assert r[k][1] <= r[k][0] <= r[k][2]


def test_group_compare_identity():
    members = {f"l{i}": _member(50, 1.3, 1.3, i) for i in range(3)}
    r = M.group_compare(members, "a", "a")
    assert r["pooled"] == (1.0, 1.0, 1.0) and r["geomean"][0] == 1.0


# ── compare ───────────────────────────────────────────────────────────────────

def test_compare_detects_differences(tmp_path):
    c = _member(30, 1.0, 2.0, 0)
    c.corpus = "in22/hin_Deva"
    M.save(c, tmp_path / "a")
    M.save(c, tmp_path / "b")
    r = R.compare(tmp_path / "a", tmp_path / "b")
    assert r["corpora"] == 1 and r["pairs"] == 2 and not r["differences"]
    c.tokens["a"][0] += 1
    M.save(c, tmp_path / "b")
    r = R.compare(tmp_path / "a", tmp_path / "b")
    assert len(r["differences"]) == 1 and r["differences"][0]["tokenizer"] == "a"


# ── throughput ────────────────────────────────────────────────────────────────

def test_throughput_interleaved(tok_path):
    from brahmai_tokenizer_eval import probes as P
    a, b = load(str(tok_path), name="a"), load(str(tok_path), name="b")
    r = P.throughput([a, b], ["the cat sat on the mat"] * 200, reference="a", rounds=3)
    assert r["rounds"] == 3 and r["reference"] == "a" and r["cpu"]
    pa, pb = r["per_tokenizer"]["a"], r["per_tokenizer"]["b"]
    assert pa["relative_to_reference"] == 1.0
    assert pb["items_bytes_per_second_min"] <= pb["items_bytes_per_second"] <= pb["items_bytes_per_second_max"]
    assert pb["relative_to_reference"] > 0
