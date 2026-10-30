# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tables from a results directory produced by `brahmai-tokenizer-eval run`."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import metrics as M
from .registry import CORPORA, lang_name


def _fmt(ci: tuple[float, float, float], nd: int = 1) -> str:
    return f"{ci[0]:.{nd}f} [{ci[1]:.{nd}f}, {ci[2]:.{nd}f}]"


def corpus_table(out: Path, corpus_ids: list[str], names: list[str]) -> list[dict]:
    rows = []
    for cid in corpus_ids:
        c = M.load_counts(out, cid)
        for n in names:
            if n not in c.tokens:
                continue
            s = M.summary(c, n)
            rows.append({"corpus": cid, "domain": CORPORA[cid].domain if cid in CORPORA else "",
                         "language": lang_name(CORPORA[cid].lang) if cid in CORPORA and CORPORA[cid].lang else "",
                         "tokenizer": n, "items": s["items"], "bytes": s["bytes"], "tokens": s["tokens"],
                         "tokens_per_kb": round(s["tokens_per_kb"][0], 2),
                         "tokens_per_kb_lo": round(s["tokens_per_kb"][1], 2),
                         "tokens_per_kb_hi": round(s["tokens_per_kb"][2], 2),
                         "bytes_per_token": round(s["bytes_per_token"], 3),
                         "fertility": round(s["fertility"][0], 3) if "fertility" in s else "",
                         "p95_tokens_per_item": round(s["p95_tokens_per_item"], 1),
                         "roundtrip_failures": s["roundtrip_failures"],
                         "roundtrip_lossy": s["roundtrip_lossy"]})
    return rows


def ratio_table(out: Path, groups: dict[str, list[str]], names: list[str], reference: str) -> list[dict]:
    """Paired ratio T_peer / T_reference per group of pooled corpora."""
    rows = []
    for g, cids in groups.items():
        c = M.merge([M.load_counts(out, cid) for cid in cids])
        for n in names:
            if n == reference or n not in c.tokens:
                continue
            r = M.paired_ratio(c, n, reference)
            rows.append({"group": g, "tokenizer": n, "reference": reference,
                         "ratio": round(r[0], 4), "lo": round(r[1], 4), "hi": round(r[2], 4)})
    return rows


def group_table(out: Path, groups: dict[str, list[str]], names: list[str], reference: str) -> list[dict]:
    """Every tokenizer against the reference on each named group: pooled and geometric-mean
    ratios with intervals, and the spread of per-corpus ratios (see metrics.group_compare)."""
    rows = []
    for g, cids in groups.items():
        members = {cid: M.load_counts(out, cid) for cid in cids}
        for n in names:
            if n == reference or any(n not in c.tokens for c in members.values()):
                continue
            r = M.group_compare(members, n, reference)
            rows.append({"group": g, "tokenizer": n, "reference": reference, "members": r["members"],
                         "pooled": round(r["pooled"][0], 4), "pooled_lo": round(r["pooled"][1], 4),
                         "pooled_hi": round(r["pooled"][2], 4),
                         "geomean": round(r["geomean"][0], 4), "geomean_lo": round(r["geomean"][1], 4),
                         "geomean_hi": round(r["geomean"][2], 4), "median": round(r["median"], 4),
                         "min": round(r["min"][1], 4), "min_member": r["min"][0],
                         "max": round(r["max"][1], 4), "max_member": r["max"][0]})
    return rows


def premium_table(out: Path, parallel: str, names: list[str], run_ids: list[str],
                  english: str = "eng_Latn") -> list[dict]:
    members = [CORPORA[c] for c in run_ids if CORPORA[c].parallel == parallel]
    eng = next(c for c in members if c.lang == english)
    ce = M.load_counts(out, eng.id)
    rows = []
    for c in members:
        if c.id == eng.id:
            continue
        cl = M.load_counts(out, c.id)
        bp = M.byte_premium(cl, ce)[0]
        for n in names:
            if n in cl.tokens:
                p = M.premium(cl, ce, n)
                rows.append({"parallel": parallel, "language": lang_name(c.lang), "lang": c.lang, "tokenizer": n,
                             "premium": round(p[0], 3), "lo": round(p[1], 3), "hi": round(p[2], 3),
                             "byte_premium": round(bp, 3)})
    return rows


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = list(rows[0])
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")


def build_report(out: Path, reference: str | None = None) -> None:
    meta = json.load(open(out / "run.json"))
    names, cids = meta["tokenizers"], meta["corpora"]
    if reference and reference not in names:
        raise ValueError(f"reference {reference!r} is not one of the run's tokenizers {names}")
    write(corpus_table(out, cids, names), out / "tables" / "per_corpus")
    domains: dict[str, list[str]] = {}
    for cid in cids:
        domains.setdefault(CORPORA[cid].domain + ":" + cid.split("/")[0], []).append(cid)
    if reference:
        write(ratio_table(out, domains, names, reference), out / "tables" / f"ratio_vs_{reference}")
        if meta.get("groups"):
            write(group_table(out, meta["groups"], names, reference), out / "tables" / f"groups_vs_{reference}")
    for par in sorted({CORPORA[c].parallel for c in cids if CORPORA[c].parallel}):
        members = [c for c in cids if CORPORA[c].parallel == par]
        if len(members) > 1 and any(CORPORA[c].lang == "eng_Latn" for c in members):
            write(premium_table(out, par, names, cids), out / "tables" / f"premium_{par}")


def compare(a: Path, b: Path) -> dict:
    """Compare per-item counts of two results directories.

    For every corpus present in both and every tokenizer present in both, the
    per-item byte and token arrays must be identical. Returns the numbers of
    corpora and (corpus, tokenizer) pairs compared and a list of differences.
    """
    import numpy as np
    ca = {p.relative_to(a / "counts").with_suffix("").as_posix() for p in (a / "counts").rglob("*.json")}
    cb = {p.relative_to(b / "counts").with_suffix("").as_posix() for p in (b / "counts").rglob("*.json")}
    diffs, pairs = [], 0
    for cid in sorted(ca & cb):
        x, y = M.load_counts(a, cid), M.load_counts(b, cid)
        if not np.array_equal(x.bytes, y.bytes):
            diffs.append({"corpus": cid, "tokenizer": None, "what": "item bytes differ (corpus changed)"})
            continue
        for n in sorted(set(x.tokens) & set(y.tokens)):
            pairs += 1
            if not np.array_equal(x.tokens[n], y.tokens[n]):
                k = int((x.tokens[n] != y.tokens[n]).sum())
                diffs.append({"corpus": cid, "tokenizer": n, "what": f"{k} items differ in token count"})
    return {"corpora": len(ca & cb), "only_a": sorted(ca - cb), "only_b": sorted(cb - ca),
            "pairs": pairs, "differences": diffs}
