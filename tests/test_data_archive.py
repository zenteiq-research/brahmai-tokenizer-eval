# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Data archive, offline mode, packaging and tokenizer loading edge cases (all offline)."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile

import pytest

from brahmai_tokenizer_eval import bundle as B
from brahmai_tokenizer_eval import corpora as C
from brahmai_tokenizer_eval.tokenizers import load, model_type

IDS = ["math/GSM8K-test", "in22/hin_Deva"]          # redistributable (MIT, CC BY)
HEAP = "code/Python"                                # per-file licences: left out of a public archive


def _write_corpus(cache, cid, texts):
    p = C.path_for(cid, cache)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n" for i, t in enumerate(texts)),
                 encoding="utf-8")
    return {"items": len(texts), "bytes": sum(len(t.encode()) for t in texts), "sha256": C._sha256_file(p)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A cache with three corpora and a lock file that matches it."""
    cache = tmp_path / "cache"
    lock = {cid: _write_corpus(cache, cid, [f"{cid} item {i} नमस्ते" for i in range(5)]) for cid in IDS + [HEAP]}
    lock_path = tmp_path / "corpora.lock.json"
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(C, "LOCK", lock_path)
    return cache, tmp_path


def test_pack_install_roundtrip(world):
    cache, tmp = world
    r = B.pack(IDS + [HEAP], tmp / "dist", cache=cache)
    assert r["left_out"] == [HEAP] and r["corpora"] == sorted(IDS)
    fresh = tmp / "fresh"
    res = B.install(r["archive"], cache=fresh)
    assert sorted(res["installed"]) == sorted(IDS)
    for cid in IDS:
        assert C.path_for(cid, fresh).read_bytes() == C.path_for(cid, cache).read_bytes()
    assert B.install(r["archive"], cache=fresh)["already_present"] == sorted(IDS)
    assert set(B.check_cache(IDS, fresh).values()) == {"ok"}


def test_pack_is_deterministic(world):
    cache, tmp = world
    a = B.pack(IDS, tmp / "a", cache=cache)
    b = B.pack(IDS, tmp / "b", cache=cache)
    assert a["sha256"] == b["sha256"]
    assert B.pack(IDS + [HEAP], tmp / "c", cache=cache, include_all=True)["corpora"] == sorted(IDS + [HEAP])


def test_install_refuses_bad_checksum(world):
    cache, tmp = world
    r = B.pack(IDS, tmp / "dist", cache=cache)
    with pytest.raises(B.BundleError, match="checksum mismatch"):
        B.install(r["archive"], cache=tmp / "x", sha256="0" * 64)
    (tmp / "dist" / (r["archive"].name + ".sha256")).unlink()
    with pytest.raises(B.BundleError, match="no checksum"):
        B.install(r["archive"], cache=tmp / "x")
    assert not (tmp / "x").exists()


def _tar(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, data, kind in members:
            ti = tarfile.TarInfo(name)
            if kind == "sym":
                ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,kind", [("../evil.jsonl", "file"), ("/abs/evil.jsonl", "file"),
                                       ("bte-corpora/corpora/../../evil.jsonl", "file"),
                                       ("bte-corpora/corpora/link.jsonl", "sym")])
def test_install_refuses_unsafe_members(world, name, kind):
    _, tmp = world
    arch = tmp / "evil.tar.gz"
    digest = _tar(arch, [(name, b"{}", kind)])
    with pytest.raises(B.BundleError, match="unsafe path|not a regular file"):
        B.install(arch, cache=tmp / "x", sha256=digest)
    assert not (tmp / "evil.jsonl").exists() and not (tmp / "x").exists()


def test_install_refuses_corpus_that_differs_from_lock(world):
    cache, tmp = world
    good = C.path_for(IDS[0], cache).read_bytes()
    arch = tmp / "tampered.tar.gz"
    digest = _tar(arch, [(f"bte-corpora/corpora/{IDS[0]}.jsonl", good, "file"),
                         (f"bte-corpora/corpora/{IDS[1]}.jsonl", b'{"id": 0, "text": "changed"}\n', "file")])
    with pytest.raises(B.BundleError, match="do not match"):
        B.install(arch, cache=tmp / "x", sha256=digest)
    assert not (tmp / "x").exists()                                # all or nothing


def test_offline_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("BTE_OFFLINE", "1")
    monkeypatch.setattr(C, "BUILDERS", {k: (lambda *a: pytest.fail("download attempted")) for k in C.BUILDERS})
    with pytest.raises(C.OfflineError, match="bte data install"):
        C.build(["math/GSM8K-test"], cache=tmp_path)


def test_package_data_resolves_inside_package():
    assert C.LOCK.exists() and C.LOCK.parent.parent.name == "brahmai_tokenizer_eval"
    assert (C.SHIPPED / "arxiv" / "abstracts.math.jsonl").exists()
    assert (C.SHIPPED / "dakshina").is_dir()


def test_wheel_contains_data(tmp_path):
    """A normal (non-editable) install must carry the lock file and shipped corpora."""
    root = C.PKG_DATA.parent.parent
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q",
                    "-w", str(tmp_path), str(root)], check=True)
    whl = next(tmp_path.glob("*.whl"))
    names = set(zipfile.ZipFile(whl).namelist())
    assert "brahmai_tokenizer_eval/data/corpora.lock.json" in names
    assert "brahmai_tokenizer_eval/data/shipped/arxiv/abstracts.math.jsonl" in names
    assert any(n.startswith("brahmai_tokenizer_eval/data/shipped/dakshina/") for n in names)


def test_tokenizer_json_without_model_type(tmp_path):
    """Older tokenizer.json files (GPT-2's) have no model "type"; loading and probing must still work."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(["the cat sat on the mat"] * 20, trainers.BpeTrainer(vocab_size=300))
    tj = json.loads(tk.to_str())
    del tj["model"]["type"]
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(tj))
    t = load(str(p), name="old")
    assert t.pipeline()["model"] == "BPE"
    assert model_type({"vocab": [["a", 0.0]]}) == "Unigram"
    assert model_type({"vocab": {"a": 0}, "continuing_subword_prefix": "##"}) == "WordPiece"
