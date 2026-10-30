# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases: empty inputs, bad tokenizer specs, damaged archives, duplicate names, CLI errors."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from brahmai_tokenizer_eval import bundle as B
from brahmai_tokenizer_eval import cli
from brahmai_tokenizer_eval import metrics as M
from brahmai_tokenizer_eval import probes as P
from brahmai_tokenizer_eval import provenance
from brahmai_tokenizer_eval.tokenizers import load


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(["the cat sat", "नमस्ते दुनिया"] * 50,
                           trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                               special_tokens=["<|eot|>"]))
    p = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tk.save(str(p))
    return p


# ── tokenizer text edge cases ─────────────────────────────────────────────────

@pytest.mark.parametrize("s", ["", " ", "\n\n", "a\x00b", "a\x1b[0mb", "  ", "﻿BOM", "é",
                               "क्‍ष", "👩‍👩‍👧", "\U0010FFFD", "x" * 100_000])
def test_round_trip_unusual_text(tok_path, s):
    t = load(str(tok_path))
    assert t.decode(t.encode(s)) == s
    ids, spans = t.encode_offsets(s)
    assert len(ids) == len(spans)


def test_special_token_text_is_reported(tok_path):
    r = P.special_collision(load(str(tok_path)))
    assert r["special_tokens"] == 1 and r["collide"] == 1


# ── bad tokenizer specs give a clear RuntimeError ─────────────────────────────

def test_bad_specs(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "empty").mkdir()
    for spec, msg in [("./missing/tokenizer.json", "no such file"), (str(tmp_path / "bad.json"), "not a valid tokenizer.json"),
                      (str(tmp_path / "empty"), "has no tokenizer.json"), ("tiktoken:nope_base", "unknown tiktoken")]:
        with pytest.raises(RuntimeError, match=msg):
            load(spec)


def test_offline_hub_uses_cache_or_explains(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import huggingface_hub.constants as hc
    monkeypatch.setattr(hc, "HF_HUB_OFFLINE", True)
    with pytest.raises(RuntimeError, match="not in the local Hugging Face cache"):
        load("no-such-org-zz9/no-such-repo-zz9")


# ── metrics and probes on empty input ─────────────────────────────────────────

def test_empty_inputs_raise_value_error(tok_path):
    t = load(str(tok_path))
    empty = M.Counts("c", np.array([]), np.array([]), np.array([]), {"x": np.array([])}, {})
    for fn in (lambda: M.summary(empty, "x"), lambda: M.merge([]), lambda: P.normalisation(t, []),
               lambda: P.normalisation(t, [""]), lambda: P.throughput([t], [], t.name, rounds=1)):
        with pytest.raises(ValueError):
            fn()


def test_single_item_interval_is_degenerate():
    c = M.Counts("c", np.array([10]), np.array([10]), np.array([2]), {"x": np.array([3])}, {})
    assert M.summary(c, "x")["tokens_per_kb"] == (300.0, 300.0, 300.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_duplicate_names_become_unique(tok_path):
    names = [t.name for t in cli._tokenizers([str(tok_path)] * 3 + [f"a={tok_path}", f"a={tok_path}"])]
    assert len(set(names)) == len(names) == 5


@pytest.mark.parametrize("argv, msg", [
    (["report", "nope-dir"], "not a results directory"),
    (["compare", "nope-a", "nope-b"], "not a results directory"),
    (["data", "install", "nope.tar.gz"], "no such archive"),
    (["run", "-t", "./missing.json", "--suite", "quick", "--out", "OUT"], "no such file"),
])
def test_cli_errors_exit_cleanly(tmp_path, monkeypatch, argv, msg):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["-q", *argv])
    assert msg in str(e.value)


def test_cli_rejects_unknown_reference(tok_path, tmp_path):
    with pytest.raises(SystemExit, match="--reference"):
        cli.main(["-q", "--offline", "run", "-t", str(tok_path), "--suite", "quick",
                  "--out", str(tmp_path / "o"), "--reference", "nope"])


# ── data archive ──────────────────────────────────────────────────────────────

def test_install_rejects_damaged_archives(tmp_path):
    a = tmp_path / "x.tar.gz"
    a.write_bytes(b"not a gzip file")
    (tmp_path / "x.tar.gz.sha256").write_text("")
    with pytest.raises(B.BundleError, match="no checksum"):
        B.install(a, cache=tmp_path / "c")
    with pytest.raises(B.BundleError, match="not a readable"):
        B.install(a, cache=tmp_path / "c", sha256=hashlib.sha256(a.read_bytes()).hexdigest())
    assert not (tmp_path / "c").exists()


# ── provenance ────────────────────────────────────────────────────────────────

def test_provenance_ignores_an_enclosing_repository(monkeypatch, tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(provenance, "PKG_ROOT", tmp_path / "site-packages")
    (tmp_path / "site-packages").mkdir()
    assert provenance.suite_commit() == {"commit": None, "dirty": None}
