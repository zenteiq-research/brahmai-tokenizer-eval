# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Data archive: every corpus in one versioned .tar.gz that can be verified and installed offline.

  pack      write bte-corpora-<lock digest>.tar.gz (+ .sha256) from a verified cache
  install   check the archive's SHA-256, unpack it safely to a staging directory, verify every corpus
            against the lock file shipped with this package, and only then move the files into the cache

The trust anchor is the lock file in the installed package, not anything inside the archive: a tampered
or truncated archive, a corpus that differs from the lock, or a member with an unsafe path (absolute,
"..", link, device) makes install stop before anything is written to the cache.

The archive is deterministic (sorted members, fixed timestamps and owners), so packing the same corpora
twice gives a byte-identical file and the same SHA-256.

By default only corpora whose licence allows redistribution are packed (a public archive). Corpora
whose terms must be checked per file or on a dataset card are left out and still build from their
pinned source; `include_all=True` packs them too, for an archive kept private.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from . import corpora as C
from .registry import CORPORA

ROOT = "bte-corpora"
NOT_REDISTRIBUTABLE = ("see dataset card", "per-file")  # licence strings that need a per-source check


class BundleError(RuntimeError):
    pass


def redistributable(corpus_id: str) -> bool:
    lic = CORPORA[corpus_id].licence.lower()
    return not any(m in lic for m in NOT_REDISTRIBUTABLE)


def _lock() -> dict:
    return json.loads(C.LOCK.read_text()) if C.LOCK.exists() else {}


def lock_digest() -> str:
    return hashlib.sha256(C.LOCK.read_bytes()).hexdigest()


def _add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(f"{ROOT}/{name}")
    ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    tf.addfile(ti, io.BytesIO(data))


# (corpus-id prefix, source and creators, licence key); creators are credited as CC BY and MIT require
SOURCES = [
    ("in22/", "IN22-Gen. Gala et al., IndicTrans2, AI4Bharat (2023); https://huggingface.co/datasets/mteb/IN22-Gen", "CC-BY-4.0"),
    ("flores/", "FLORES-200 devtest. NLLB Team, Meta AI (2022); https://github.com/facebookresearch/flores", "CC-BY-SA-4.0"),
    ("dakshina/", "Dakshina v1.0. Roark et al., Google Research (2020); https://github.com/google-research-datasets/dakshina", "CC-BY-SA-4.0"),
    ("milu/", "MILU-1.5, natively authored items. AI4Bharat; https://huggingface.co/datasets/ai4bharat/MILU-1.5", "CC-BY-4.0"),
    ("indiceval-gsm8k/", "Indic GSM8K. AI4Bharat, from GSM8K (Cobbe et al., OpenAI, 2021)", "Apache-2.0"),
    ("indiceval-math500/", "Indic MATH-500. AI4Bharat, from MATH (Hendrycks et al., 2021)", "Apache-2.0"),
    ("indiceval-humaneval/", "Indic HumanEval. AI4Bharat, from HumanEval (Chen et al., OpenAI, 2021)", "Apache-2.0"),
    ("indiceval-mmlupro/", "Indic MMLU-Pro. AI4Bharat, from MMLU-Pro (Wang et al., TIGER-Lab, 2024)", "MIT"),
    ("indiceval-arc/", "Indic ARC-Challenge. AI4Bharat, from ARC (Clark et al., Allen AI, 2018)", "CC-BY-SA-4.0"),
    ("math/MATH", "MATH. Hendrycks et al. (2021); https://huggingface.co/datasets/EleutherAI/hendrycks_math", "MIT"),
    ("math/GSM8K", "GSM8K. Cobbe et al., OpenAI (2021); https://huggingface.co/datasets/openai/gsm8k", "MIT"),
    ("latex/Wikipedia-formulas", "Wikipedia LaTeX formulas, English Wikipedia contributors; https://huggingface.co/datasets/piushorn/wikipedia-latex-formulas-319k", "CC-BY-SA-4.0"),
    ("science/arXiv-", "arXiv abstracts (metadata, CC0); https://info.arxiv.org/help/license/index.html", "CC0-1.0"),
    ("latex/arXiv-", "arXiv paper sources; authors, titles and licence of each paper are in the `attribution` list of MANIFEST.json in the evaluation suite's data/shipped/arxiv folder. Modified copies: comments removed, truncated, e-mail addresses redacted", "per paper"),
]
CC_URLS = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/legalcode",
           "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
           "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/legalcode"}
TEXTS = Path(__file__).parent / "data" / "licence_texts"


def _licences(ids: list[str]) -> str:
    rows = "\n".join(f"| `{cid}` | {CORPORA[cid].licence} |" for cid in ids)
    src = [(p, d, k) for p, d, k in SOURCES if any(c.startswith(p) for c in ids)]
    srows = "\n".join(f"| `{p}*` | {d} | {k} |" for p, d, k in src)
    used = {k for _, _, k in src}
    texts = "".join(f"\n\n## {k} licence text\n\n```\n{(TEXTS / (k + '.txt')).read_text()}\n```"
                    for k in ("MIT", "Apache-2.0") if k in used)
    links = "\n".join(f"* {k}: {u}" for k, u in CC_URLS.items() if k in used or k.startswith("CC-BY") and "per paper" in used)
    return ("# Licences and attribution\n\nEach corpus keeps the licence of its source and is redistributed "
            "unchanged unless a source note says otherwise. CC BY and CC BY-SA copies must credit the creators "
            "named below and link the licence; CC BY-SA copies (and adaptations) stay under CC BY-SA. "
            "MIT and Apache-2.0 sources require their licence text to travel with copies; it is below. "
            "The creators do not endorse this archive.\n\n"
            "## Sources\n\n| Corpus | Source and creators | Licence |\n|---|---|---|\n" + srows +
            "\n\n## Licence links (Creative Commons legal code)\n\n" + links +
            texts + "\n\n## Per corpus\n\n| Corpus | Licence |\n|---|---|\n" + rows + "\n")


def pack(corpus_ids: list[str], out_dir: Path, cache: Path | None = None, include_all: bool = False) -> dict:
    """Pack verified corpora; returns {"archive", "sha256", "corpora", "left_out"}."""
    cache = cache or C.default_cache()
    lock = _lock()
    ids = sorted(dict.fromkeys(corpus_ids))
    left_out = [c for c in ids if not include_all and not redistributable(c)]
    ids = [c for c in ids if c not in left_out]
    unlocked = [c for c in ids if c not in lock]
    if unlocked:
        raise BundleError(f"corpora not in the lock file: {unlocked[:5]}")
    manifest = {}
    for cid in ids:
        p = C.path_for(cid, cache)
        if not p.exists():
            raise BundleError(f"{cid} is not built; run `bte corpora` first")
        if C._sha256_file(p) != lock[cid]["sha256"]:
            raise BundleError(f"{cid} differs from the lock file; it will not be packed")
        manifest[cid] = {**lock[cid], "licence": CORPORA[cid].licence}
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"bte-corpora-{lock_digest()[:12]}{'-private' if include_all else ''}.tar.gz"
    archive = out_dir / name
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
        _add(tf, "MANIFEST.json", json.dumps({"lock_sha256": lock_digest(), "corpora": manifest},
                                             indent=1, sort_keys=True).encode())
        _add(tf, "LICENCES.md", _licences(ids).encode())
        for cid in ids:
            _add(tf, f"corpora/{cid}.jsonl", C.path_for(cid, cache).read_bytes())
    with open(archive, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
        gz.write(raw.getvalue())
    digest = C._sha256_file(archive)
    (out_dir / (name + ".sha256")).write_text(f"{digest}  {name}\n")
    return {"archive": archive, "sha256": digest, "corpora": ids, "left_out": left_out}


def _safe_member(m: tarfile.TarInfo) -> str | None:
    """Relative path under ROOT for a regular file with a safe name, else raise."""
    parts = Path(m.name).parts
    if m.name.startswith("/") or ".." in parts or not parts or parts[0] != ROOT:
        raise BundleError(f"unsafe path in archive: {m.name!r}")
    if m.isdir():
        return None
    if not m.isfile():
        raise BundleError(f"archive member is not a regular file: {m.name!r}")
    return "/".join(parts[1:])


def install(archive: Path, cache: Path | None = None, sha256: str | None = None) -> dict:
    """Verify and install an archive; returns {"installed", "already_present"}. Writes nothing on failure."""
    cache = cache or C.default_cache()
    archive = Path(archive)
    if not archive.is_file():
        raise BundleError(f"no such archive: {archive}")
    side = archive.with_name(archive.name + ".sha256")
    expected = sha256 or ((side.read_text().split() or [None])[0] if side.exists() else None)
    if expected is None:
        raise BundleError(f"no checksum: pass --sha256 or keep {side.name} next to the archive")
    actual = C._sha256_file(archive)
    if actual != expected.lower():
        raise BundleError(f"archive checksum mismatch: expected {expected}, got {actual}")
    lock = _lock()
    with tempfile.TemporaryDirectory(prefix="bte-install-") as tmp:
        stage = Path(tmp)
        try:
            tf = tarfile.open(archive, "r:gz")
        except (tarfile.TarError, OSError) as exc:
            raise BundleError(f"{archive.name} is not a readable .tar.gz archive ({exc})") from None
        with tf:
            for m in tf.getmembers():
                rel = _safe_member(m)
                if rel is None:
                    continue
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        files = sorted((stage / "corpora").rglob("*.jsonl")) if (stage / "corpora").exists() else []
        if not files:
            raise BundleError("archive contains no corpora")
        ids = {p.relative_to(stage / "corpora").as_posix()[:-len(".jsonl")]: p for p in files}
        bad = [cid for cid, p in ids.items() if cid not in lock or C._sha256_file(p) != lock[cid]["sha256"]]
        if bad:
            raise BundleError(f"{len(bad)} corpora do not match this suite's lock file (e.g. {bad[:3]}); "
                              "nothing was installed")
        installed, present = [], []
        for cid, p in ids.items():
            dest = C.path_for(cid, cache)
            if dest.exists() and C._sha256_file(dest) == lock[cid]["sha256"]:
                present.append(cid)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_dest = dest.with_suffix(".jsonl.part")
            shutil.copyfile(p, tmp_dest)
            tmp_dest.replace(dest)
            installed.append(cid)
    return {"installed": installed, "already_present": present}


def check_cache(corpus_ids: list[str], cache: Path | None = None) -> dict[str, str]:
    """Status of each corpus in the cache without building anything: ok, missing, MISMATCH, unlocked."""
    cache = cache or C.default_cache()
    lock = _lock()
    out = {}
    for cid in corpus_ids:
        p = C.path_for(cid, cache)
        if cid not in lock:
            out[cid] = "unlocked"
        elif not p.exists():
            out[cid] = "missing"
        else:
            out[cid] = "ok" if C._sha256_file(p) == lock[cid]["sha256"] else "MISMATCH"
    return out
