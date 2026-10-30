# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Held-out scientific prose and LaTeX source from arXiv, dated after the tokenizer.

The tokenizer under test was published on 2026-04-03, so its training corpus predates that day. Papers submitted between 2026-09-01 and
2026-10-15 cannot be in it. For each field group the script takes the most
recent submissions (arXiv API, 3 s between requests as arXiv asks), stores
abstracts as scientific prose, and for a subset downloads the e-print and
keeps the main .tex file (the one with \\documentclass) as LaTeX source.

Archives are untrusted input: they are unpacked with tarfile's "data" filter
into a scratch directory and only .tex files are read as text.

Usage: python tools/build_arxiv_heldout.py   (then python tools/arxiv_licences.py)
"""
from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import json as _json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "work"
RES_DIR = DATA_DIR


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(obj, indent=2))

GROUPS = {
    "math": "cat:math.AP OR cat:math.PR OR cat:math.NT OR cat:math.OC OR cat:math.AG",
    "physics": "cat:hep-th OR cat:cond-mat.str-el OR cat:quant-ph OR cat:astro-ph.CO OR cat:gr-qc",
    "cs": "cat:cs.LG OR cat:cs.CL OR cat:cs.DS OR cat:cs.IT",
    "qbio-stat": "cat:q-bio.QM OR cat:stat.ME OR cat:stat.ML OR cat:q-bio.PE",
}
WINDOW = "submittedDate:[202609010000 TO 202610152359]"
N_ABS, N_TEX, MAX_CHARS = 150, 30, 50_000
UA = {"User-Agent": "brahmai-tokenizer-eval/0.2 (research; mailto:research@zenteiq.ai)"}
NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = DATA_DIR / "corpora"
SCRATCH = DATA_DIR / "raw" / "arxiv"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    time.sleep(3.1)
    return data


def query(cat: str) -> list[dict]:
    q = urllib.parse.quote(f"({cat}) AND {WINDOW}")
    url = (f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={N_ABS}"
           "&sortBy=submittedDate&sortOrder=descending")
    root = ET.fromstring(get(url))
    out = []
    for e in root.findall("a:entry", NS):
        out.append({"id": e.find("a:id", NS).text.rsplit("/abs/", 1)[-1],
                    "published": e.find("a:published", NS).text,
                    "abstract": " ".join(e.find("a:summary", NS).text.split())})
    return out


def main_tex(arxiv_id: str) -> str | None:
    blob = get(f"https://arxiv.org/e-print/{arxiv_id}")
    texts = []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".tex") and m.size < 5_000_000:
                    f = tf.extractfile(m)
                    if f:
                        texts.append(f.read().decode("utf-8", errors="replace"))
    except tarfile.ReadError:
        try:
            texts.append(gzip.decompress(blob).decode("utf-8", errors="replace"))
        except OSError:
            return None
    mains = [t for t in texts if "\\documentclass" in t]
    if not mains:
        return None
    t = max(mains, key=len)
    t = re.sub(r"(?m)^\s*%.*\n", "", t)  # drop whole-line comments
    return t[:MAX_CHARS]


def write(corpus: str, subset: str, texts: list[str], meta: dict) -> None:
    p = OUT / corpus / f"{subset}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    meta[f"{corpus}/{subset}"] = {"items": len(texts), "bytes": sum(len(t.encode()) for t in texts)}
    print(f"  {corpus}/{subset}: {len(texts)}")


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    meta = {"window": WINDOW, "tokenizer_published": "2026-04-03", "papers": {}}
    for g, cat in GROUPS.items():
        entries = query(cat)
        write("arxiv_abs", g, [e["abstract"] for e in entries], meta)
        texs, ids = [], []
        for e in entries:
            if len(texs) >= N_TEX:
                break
            try:
                t = main_tex(e["id"])
            except Exception as exc:  # network or archive errors: skip paper
                print("   skip", e["id"], type(exc).__name__)
                continue
            if t and len(t) > 2000:
                texs.append(t)
                ids.append(e["id"])
        write("latex_src", g, texs, meta)
        meta["papers"][g] = {"abstract_ids": [e["id"] for e in entries], "latex_ids": ids,
                             "published_range": [min(e["published"] for e in entries),
                                                 max(e["published"] for e in entries)]}
    save_json(meta, RES_DIR / "manifest_arxiv.json")


if __name__ == "__main__":
    main()
