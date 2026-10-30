# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BrahmAI Tokenizer Eval — command-line interface.

  brahmai-tokenizer-eval list                              suites and corpora with licences
  brahmai-tokenizer-eval corpora --suite core [--lock]     build (download/sample) and verify corpora
  brahmai-tokenizer-eval run -t NAME=SPEC [-t ...] --suite core --out results/ [--reference NAME]
  brahmai-tokenizer-eval probes -t NAME=SPEC [...] --out results/
  brahmai-tokenizer-eval report results/ --reference NAME
  brahmai-tokenizer-eval reproduce manifest.toml --out results/   pinned run + probes from a manifest
  brahmai-tokenizer-eval compare results_a/ results_b/              are per-item counts identical?
  brahmai-tokenizer-eval data install bte-corpora-<id>.tar.gz        verify and install the data archive
  brahmai-tokenizer-eval data pack --suite full --out dist/         build the data archive (maintainers)
  brahmai-tokenizer-eval data check --suite full                    cache status against the lock file

`bte` is a short alias for `brahmai-tokenizer-eval`.
A tokenizer is a Hugging Face id (org/repo[@revision]), a local tokenizer.json or directory,
or tiktoken:<encoding>; prefix NAME= to choose the label used in tables.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from . import bundle as B
from . import corpora as C
from . import manifest as MF
from . import metrics as M
from . import probes as P
from . import provenance
from . import report as R
from . import ui
from .registry import CORPORA, SUITES
from .tokenizers import load


TRUST_REMOTE_CODE = False
ALLOW_DATA_MISMATCH = False


def _tokenizers(specs: list[str]) -> list:
    """Accepts NAME=SPEC or a bare SPEC (a Hugging Face id, local path or tiktoken:<enc>)."""
    from .tokenizers import default_name
    pairs = []
    for s in specs:
        name, sep, spec = s.partition("=")
        if not sep or "/" in name or name.startswith(("tiktoken:", "hf:", ".")):
            name, spec = default_name(s), s
        pairs.append((name, spec))
    seen: set[str] = set()
    for i, (name, spec) in enumerate(pairs):  # make names unique: name, name-2, name-3, ...
        new, k = name, 1
        while new in seen:
            k += 1
            new = f"{name}-{k}"
        if new != name:
            ui.warn(f"tokenizer name {name!r} is used twice; the second is called {new!r}")
        pairs[i] = (new, spec)
        seen.add(new)
    out = []
    with ui.progress() as p:
        task = p.add_task("loading tokenizers", total=len(pairs), detail="")
        for name, spec in pairs:
            p.update(task, detail=name)
            out.append(load(spec, name=name, trust_remote_code=TRUST_REMOTE_CODE))
            p.advance(task)
        p.update(task, detail="done")
    return out


def _corpus_ids(args) -> list[str]:
    ids = list(args.corpus or [])
    if args.suite:
        ids += SUITES[args.suite]
    if not ids:
        sys.exit("choose corpora with --suite and/or --corpus (see `brahmai-tokenizer-eval list`)")
    unknown = [c for c in ids if c not in CORPORA]
    if unknown:
        sys.exit(f"unknown corpus ids: {unknown}")
    return list(dict.fromkeys(ids))


def _build(ids: list[str], force: bool = False, strict: bool = True) -> tuple[dict, dict]:
    with ui.progress() as p:
        task = p.add_task("building corpora", total=len(ids), detail="")
        info = C.build(ids, force=force,
                       on_step=lambda cid, cached: p.update(task, advance=1, detail=f"{cid}{' (cached)' if cached else ''}"))
        p.update(task, detail="done")
    status = C.verify(info)
    bad = [c for c, s in status.items() if s == "MISMATCH"]
    if bad:
        msg = (f"{len(bad)} corpora differ from the lock file ({C.LOCK.name}), so results would not be "
               f"comparable with published numbers: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
        if strict and not ALLOW_DATA_MISMATCH:
            sys.exit(f"error: {msg}. Reinstall the data archive, or pass --allow-data-mismatch "
                     "for an exploratory run.")
        ui.warn(msg)
    return info, status


def cmd_list(args) -> None:
    ui.table("Suites", ["suite", "corpora"], [[s, len(ids)] for s, ids in SUITES.items()])
    rows = [[c.id, c.domain, c.lang or "", c.parallel or "", c.licence] for c in CORPORA.values()]
    ui.table("Corpora", ["id", "domain", "language", "parallel group", "licence"], rows, justify_right_from=99)


def cmd_corpora(args) -> None:
    ids = _corpus_ids(args)
    ui.header("BrahmAI Tokenizer Eval · corpora", {"version": __version__, "corpora": str(len(ids)),
                                                   "cache": str(C.default_cache())})
    info, status = _build(ids, force=args.force, strict=False)  # reports; --lock may re-record
    counts = {s: sum(1 for v in status.values() if v == s) for s in ("ok", "unlocked", "MISMATCH")}
    ui.table("Verification against the lock file", ["status", "corpora"], [[k, v] for k, v in counts.items()])
    ui.info(f"{len(ids)} corpora, {sum(i['items'] for i in info.values()):,} items, "
            f"{sum(i['bytes'] for i in info.values()) / 1e6:.1f} MB")
    if args.lock:
        C.write_lock(info)
        ui.info(f"wrote {C.LOCK}")
    if counts["MISMATCH"]:
        sys.exit(1)


def cmd_run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = _corpus_ids(args)
    ui.header("BrahmAI Tokenizer Eval · run", {
        "version": __version__, "tokenizers": str(len(args.tokenizer)), "corpora": str(len(ids)),
        "output": str(out), "cache": str(C.default_cache())})
    prov = provenance.record()
    toks = _tokenizers(args.tokenizer)
    if args.reference and args.reference not in [t.name for t in toks]:
        sys.exit(f"error: --reference {args.reference!r} is not one of {[t.name for t in toks]}")
    _run(out, ids, toks, args.reference, prov=prov)


def _end_check(prov: dict) -> dict:
    """Provenance is taken when a command starts; flag a suite tree that changed while it ran."""
    end = provenance.suite_commit()
    changed = (end["commit"], end["dirty"]) != (prov.get("suite_commit"), prov.get("suite_dirty"))
    if changed:
        ui.warn(f"the suite tree changed during the run (start {str(prov.get('suite_commit'))[:12]}, "
                f"end {str(end['commit'])[:12]}, dirty {end['dirty']})")
    return {"suite_commit_at_end": end["commit"], "suite_dirty_at_end": end["dirty"], "suite_changed_during_run": changed}


def _run(out: Path, ids: list[str], toks: list, reference: str | None, groups: dict | None = None,
         extra: dict | None = None, prov: dict | None = None) -> None:
    prov = prov or provenance.record()
    info, status = _build(ids)
    used = {t.name: set() for t in toks}
    t0 = time.perf_counter()
    with ui.progress() as p:
        total = p.add_task("tokenizing", total=len(ids) * len(toks), detail="")
        for cid in ids:
            c = M.count(cid, toks, on_step=lambda name, cid=cid: p.update(total, advance=1, detail=f"{cid} · {name}"))
            for n, u in c.used_ids.items():
                used[n] |= u
            M.save(c, out)
        p.update(total, detail="done")
    utilisation = {}
    for t in toks:
        table, special = t.token_bytes()
        ordinary = set(table) - special
        utilisation[t.name] = {"ordinary_entries": len(ordinary), "used": len(used[t.name] & ordinary),
                               "share": len(used[t.name] & ordinary) / max(len(ordinary), 1)}
    meta = {**prov, **_end_check(prov), **(extra or {}), "tokenizers": [t.name for t in toks], "corpora": ids,
            "groups": groups or {}, "corpus_info": info, "corpus_lock_status": status,
            "vocabulary_utilisation": utilisation, "tokenize_seconds": round(time.perf_counter() - t0, 1),
            "tokenizer_sources": {t.name: {**t.source, "spec": t.spec, "vocab_size": t.vocab_size,
                                           "vocab_fingerprint": t.fingerprint()} for t in toks}}
    with open(out / "run.json", "w") as f:
        json.dump(meta, f, indent=1)
    ref = reference or toks[0].name
    R.build_report(out, reference=ref)
    _summary(out, ids, toks, ref)


def _summary(out: Path, ids: list[str], toks: list, ref: str) -> None:
    """Pooled tokens per 1,000 bytes by corpus group, and round-trip failures."""
    groups: dict[str, list[str]] = {}
    for cid in ids:
        groups.setdefault(cid.split("/")[0], []).append(cid)
    names = [t.name for t in toks]
    rows = []
    for g, cids in groups.items():
        c = M.merge([M.load_counts(out, cid) for cid in cids])
        rows.append([g] + [f"{1000 * c.tokens[n].sum() / c.bytes.sum():.1f}" for n in names])
    show = names[:8]
    ui.table(f"Tokens per 1,000 bytes, pooled by corpus group (lower is better; first {len(show)} tokenizers)",
             ["group"] + show, [r[:1 + len(show)] for r in rows])
    fails = {n: 0 for n in names}
    for cid in ids:
        c = M.load_counts(out, cid)
        for n in names:
            fails[n] += len(c.roundtrip_failures.get(n, []))
    bad = [[n, v] for n, v in fails.items() if v]
    if bad:
        ui.table("Items that do not round-trip exactly", ["tokenizer", "items"], bad)
    ui.info(f"[bold green]done[/bold green] · tables in {out / 'tables'} · reference tokenizer {ref}")


def cmd_probes(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ui.header("BrahmAI Tokenizer Eval · probes", {"version": __version__, "tokenizers": str(len(args.tokenizer)),
                                                  "output": str(out)})
    prov = provenance.record()
    _probes(out, _tokenizers(args.tokenizer), prov=prov)


def _probes(out: Path, toks: list, extra: dict | None = None, prov: dict | None = None,
            reference: str | None = None) -> None:
    prov = prov or provenance.record()
    norm_ids = ["in22/hin_Deva", "in22/mal_Mlym", "in22/ben_Beng", "flores/ben_Beng"]
    boundary_ids = norm_ids + ["in22/eng_Latn", "in22/tam_Taml"]
    _build(["math/GSM8K-test"] + boundary_ids)
    running = C.load("math/GSM8K-test")
    probes = [
        ("pipeline", lambda t: t.pipeline()),
        ("missing byte symbols", P.missing_byte_symbols),
        ("unicode coverage", P.unicode_coverage),
        ("reachability", P.reachability),
        ("numbers", lambda t: P.numbers(t, running)),
        ("normalisation", lambda t: {cid: P.normalisation(t, C.load(cid)) for cid in norm_ids}),
        ("composition", P.composition),
        ("special-token collision", P.special_collision),
        ("utf-8 boundaries", lambda t: {cid: P.utf8_boundaries(t, C.load(cid)) for cid in boundary_ids}),
    ]
    res = {}
    with ui.progress() as p:
        task = p.add_task("running probes", total=len(toks) * len(probes), detail="")
        for t in toks:
            res[t.name] = {"vocab_size": t.vocab_size}
            for label, fn in probes:
                p.update(task, detail=f"{t.name} · {label}")
                res[t.name][label.replace(" ", "_").replace("-", "_")] = fn(t)
                p.advance(task)
        p.update(task, detail="throughput (interleaved rounds)")
        tp = P.throughput(toks, C.load("in22/hin_Deva") + C.load("in22/eng_Latn"), reference or toks[0].name)
        for t in toks:
            res[t.name]["throughput"] = tp["per_tokenizer"][t.name]
        p.update(task, detail="done")
    with open(out / "probes.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    with open(out / "probes.run.json", "w") as f:
        json.dump({**prov, **_end_check(prov), **(extra or {}), "tokenizers": [t.name for t in toks],
                   "throughput": {k: v for k, v in tp.items() if k != "per_tokenizer"},
                   "tokenizer_sources": {t.name: {**t.source, "spec": t.spec,
                                                  "vocab_fingerprint": t.fingerprint()} for t in toks}},
                  f, indent=1)
    rows = [[n, r["unicode_coverage"]["code_points_lost"], len(r["missing_byte_symbols"]),
             r["reachability"]["not_self_encoding"], f"{r['numbers']['integers_0_9999']['share_single_digit']:.2f}",
             r["special_token_collision"].get("collide", 0)] for n, r in res.items()]
    ui.table("Probe summary", ["tokenizer", "lost code points", "missing bytes", "not self-encoding",
                               "single-digit share", "special collisions"], rows)
    ui.info(f"[bold green]done[/bold green] · {out / 'probes.json'}")


def cmd_reproduce(args) -> None:
    """Run and probe exactly what a manifest pins; refuse anything that differs from it."""
    m = MF.load_manifest(args.manifest)
    out = Path(args.out)
    prov = provenance.record()
    commit = {"commit": prov["suite_commit"], "dirty": prov["suite_dirty"]}
    if commit["dirty"] and not args.allow_dirty:
        sys.exit("the suite's source tree has uncommitted changes; commit them or pass --allow-dirty")
    ui.header(f"BrahmAI Tokenizer Eval · reproduce {m.name}", {
        "version": __version__, "commit": (commit["commit"] or "unknown")[:12], "manifest": str(m.path),
        "tokenizers": str(len(m.tokenizers)), "corpora": str(len(m.corpora)), "output": str(out)})
    toks = _tokenizers(m.specs())
    for t, pin in zip(toks, m.tokenizers):
        MF.verify(t, pin)
    ui.info(f"all {len(toks)} tokenizers match their pins")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(m.path, out / "manifest.toml")
    extra = {"manifest": {"name": m.name, "sha256": m.sha256, "path": str(m.path)}}
    if not args.probes_only:
        _run(out, m.corpora, toks, m.reference, m.groups, extra, prov=prov)
    if m.probes and not args.skip_probes:
        _probes(out, [t for t in toks if t.name in m.probes], extra, prov=prov, reference=m.reference)


def cmd_compare(args) -> None:
    for d in (args.a, args.b):
        if not (Path(d) / "counts").is_dir():
            sys.exit(f"error: {d} is not a results directory (no counts/ folder)")
    r = R.compare(Path(args.a), Path(args.b))
    if not r["corpora"]:
        sys.exit("error: the two results directories have no corpus in common")
    ui.info(f"{r['corpora']} corpora and {r['pairs']} (corpus, tokenizer) pairs compared")
    if r["only_a"] or r["only_b"]:
        ui.warn(f"corpora only in {args.a}: {len(r['only_a'])}; only in {args.b}: {len(r['only_b'])}")
    if r["differences"]:
        ui.table("Differences", ["corpus", "tokenizer", "what"],
                 [[d["corpus"], d["tokenizer"] or "", d["what"]] for d in r["differences"][:50]])
        sys.exit(1)
    ui.info("[bold green]identical[/bold green] per-item counts")


def cmd_data(args) -> None:
    if args.action == "install":
        r = B.install(Path(args.archive), sha256=args.sha256)
        ui.info(f"[bold green]verified[/bold green] archive and every corpus against the lock file; "
                f"installed {len(r['installed'])}, already present {len(r['already_present'])} "
                f"in {C.default_cache() / 'corpora'}")
        return
    ids = _corpus_ids(args)
    if args.action == "check":
        st = B.check_cache(ids)
        counts = {k: sum(1 for v in st.values() if v == k) for k in ("ok", "missing", "MISMATCH", "unlocked")}
        ui.table("Cache against the lock file", ["status", "corpora"], [[k, v] for k, v in counts.items()])
        for k in ("missing", "MISMATCH"):
            if counts[k]:
                ui.warn(f"{k}: {[c for c, v in st.items() if v == k][:8]}")
        if counts["missing"] or counts["MISMATCH"]:
            sys.exit(1)
        return
    r = B.pack(ids, Path(args.out), include_all=args.include_all)
    ui.info(f"wrote {r['archive']} ({r['archive'].stat().st_size / 1e6:.1f} MB, {len(r['corpora'])} corpora)")
    ui.info(f"sha256 {r['sha256']}")
    if r["left_out"]:
        ui.warn(f"left out {len(r['left_out'])} corpora whose licence needs a per-source check "
                f"(they build from their pinned source): {r['left_out']}")


def cmd_report(args) -> None:
    if not (Path(args.results) / "run.json").is_file():
        sys.exit(f"error: {args.results} is not a results directory (no run.json)")
    R.build_report(Path(args.results), reference=args.reference)
    ui.info(f"tables written to {Path(args.results) / 'tables'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="brahmai-tokenizer-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"brahmai-tokenizer-eval {__version__}")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress bars or tables")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="allow tokenizers implemented as custom code in their Hub repository")
    ap.add_argument("--offline", action="store_true",
                    help="never download corpora (same as BTE_OFFLINE=1); use the installed data archive")
    ap.add_argument("--allow-data-mismatch", action="store_true",
                    help="continue when a corpus differs from the lock file (results are then not comparable)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list suites and corpora")
    p.set_defaults(fn=cmd_list)
    for name, fn, hlp in (("corpora", cmd_corpora, "build and verify corpora"),
                          ("run", cmd_run, "tokenize corpora and write tables")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--suite", choices=sorted(SUITES))
        p.add_argument("--corpus", action="append", help="corpus id (repeatable)")
        if name == "corpora":
            p.add_argument("--force", action="store_true", help="rebuild even if cached")
            p.add_argument("--lock", action="store_true", help="record checksums in the lock file")
        else:
            p.add_argument("-t", "--tokenizer", action="append", required=True,
                           help="Hugging Face id (e.g. meta-llama/Llama-3.2-1B), local path, tiktoken:<enc>, "
                                "or NAME=SPEC; repeatable")
            p.add_argument("--out", required=True)
            p.add_argument("--reference", help="tokenizer used as denominator in ratio tables (default: first)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("probes", help="diagnostic probes (no corpus needed)")
    p.add_argument("-t", "--tokenizer", action="append", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_probes)
    p = sub.add_parser("reproduce", help="pinned run and probes from a manifest (see manifest.py)")
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--skip-probes", action="store_true")
    p.add_argument("--probes-only", action="store_true")
    p.add_argument("--allow-dirty", action="store_true", help="run from a tree with uncommitted changes")
    p.set_defaults(fn=cmd_reproduce)
    p = sub.add_parser("compare", help="check that two results directories have identical per-item counts")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("data", help="install, check or build the data archive")
    dsub = p.add_subparsers(dest="action", required=True)
    q = dsub.add_parser("install", help="verify an archive and install its corpora into the cache")
    q.add_argument("archive")
    q.add_argument("--sha256", help="expected SHA-256 of the archive (default: the .sha256 file next to it)")
    for action, hlp in (("check", "status of cached corpora against the lock file"),
                        ("pack", "write the data archive from verified cached corpora")):
        q = dsub.add_parser(action, help=hlp)
        q.add_argument("--suite", choices=sorted(SUITES))
        q.add_argument("--corpus", action="append")
        if action == "pack":
            q.add_argument("--out", required=True)
            q.add_argument("--include-all", action="store_true",
                           help="also pack corpora whose licence needs a per-source check (keep that archive private)")
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("report", help="rebuild tables from a results directory")
    p.add_argument("results")
    p.add_argument("--reference")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    ui.set_quiet(args.quiet)
    global TRUST_REMOTE_CODE, ALLOW_DATA_MISMATCH
    TRUST_REMOTE_CODE = args.trust_remote_code
    ALLOW_DATA_MISMATCH = args.allow_data_mismatch
    if args.offline:
        os.environ["BTE_OFFLINE"] = "1"
    try:
        args.fn(args)
    except (MF.ManifestError, RuntimeError, ValueError, OSError) as exc:
        if os.environ.get("BTE_DEBUG"):
            raise
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
