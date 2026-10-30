# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Keep only openly licensed arXiv sources and write the shipped corpus files.

arXiv metadata (titles, abstracts) is released under CC0, so all abstracts
collected by build_arxiv_heldout.py may be redistributed. Paper sources are
licensed per paper; only CC BY, CC BY-SA and CC0 papers are kept. The licence
of each paper is read from the OAI-PMH arXiv metadata record. If a field has
fewer than TARGET openly licensed papers among those already downloaded, more
recent submissions from the same window are checked and fetched.

Writes to the evaluation suite's brahmai_tokenizer_eval/data/shipped/arxiv/ directory:
  abstracts.<field>.jsonl, latex.<field>.jsonl, MANIFEST.json (id, licence)

Usage: python tools/arxiv_licences.py
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path

import json as _json

DATA_DIR = Path(__file__).resolve().parent.parent / "work"
RES_DIR = DATA_DIR


def load_json(p):
    return _json.loads(Path(p).read_text())


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(obj, indent=2))

sys.path.insert(0, str(Path(__file__).resolve().parent))
A = import_module("build_arxiv_heldout")
SHIP = Path(__file__).resolve().parent.parent / "brahmai_tokenizer_eval" / "data" / "shipped" / "arxiv"
OPEN = re.compile(r"creativecommons\.org/(licenses/by/|licenses/by-sa/|publicdomain/zero/)")
TARGET = 15
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NS = {"a": "http://www.w3.org/2005/Atom"}


def attribution(ids: list[tuple[str, str]]) -> list[dict]:
    """Title, authors, licence and link of each kept paper, as CC BY / CC BY-SA require."""
    out = []
    for aid, lic in sorted(ids):
        base = aid.split("v")[0]
        root = ET.fromstring(A.get(f"http://export.arxiv.org/api/query?id_list={base}"))
        e = root.find("a:entry", NS)
        out.append({"arxiv_id": aid, "title": " ".join(e.find("a:title", NS).text.split()),
                    "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                    "licence": lic, "url": f"https://arxiv.org/abs/{aid}"})
    return out


def licence(arxiv_id: str) -> str | None:
    base = arxiv_id.split("v")[0] if re.search(r"v\d+$", arxiv_id) else arxiv_id
    xml = A.get(f"http://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:{base}&metadataPrefix=arXiv")
    m = re.search(rb"<license>([^<]+)</license>", xml)
    return m.group(1).decode() if m else None


def main() -> None:
    meta = load_json(RES_DIR / "manifest_arxiv.json")
    SHIP.mkdir(parents=True, exist_ok=True)
    manifest = {"window": meta["window"], "tokenizer_published": meta["tokenizer_published"], "fields": {}}
    for g, rec in meta["papers"].items():
        abstracts = [json.loads(l)["text"] for l in open(DATA_DIR / "corpora" / "arxiv_abs" / f"{g}.jsonl")]
        with open(SHIP / f"abstracts.{g}.jsonl", "w", encoding="utf-8") as f:
            for i, (aid, t) in enumerate(zip(rec["abstract_ids"], abstracts)):
                f.write(json.dumps({"id": i, "arxiv_id": aid, "text": t}, ensure_ascii=False) + "\n")
        texs = [json.loads(l)["text"] for l in open(DATA_DIR / "corpora" / "latex_src" / f"{g}.jsonl")]
        kept, checked = [], {}
        for aid, t in zip(rec["latex_ids"], texs):
            lic = licence(aid)
            checked[aid] = lic
            if lic and OPEN.search(lic):
                kept.append((aid, lic, t))
        for aid in rec["abstract_ids"]:
            if len(kept) >= TARGET:
                break
            if aid in checked:
                continue
            lic = licence(aid)
            checked[aid] = lic
            if lic and OPEN.search(lic):
                try:
                    t = A.main_tex(aid)
                except Exception:
                    t = None
                if t and len(t) > 2000:
                    kept.append((aid, lic, t))
        with open(SHIP / f"latex.{g}.jsonl", "w", encoding="utf-8") as f:
            for i, (aid, lic, t) in enumerate(kept):
                t = EMAIL.sub("author@example.org", t)   # author contact details are not redistributed
                f.write(json.dumps({"id": i, "arxiv_id": aid, "licence": lic, "text": t}, ensure_ascii=False) + "\n")
        manifest["fields"][g] = {"abstracts": len(abstracts), "latex_kept": [[a, l] for a, l, _ in kept],
                                 "licences_checked": checked, "published_range": rec["published_range"]}
        print(g, "abstracts", len(abstracts), "latex kept", len(kept), "checked", len(checked))
    manifest["attribution"] = attribution([(a, l) for f in manifest["fields"].values() for a, l in f["latex_kept"]])
    save_json(manifest, SHIP / "MANIFEST.json")


if __name__ == "__main__":
    main()
