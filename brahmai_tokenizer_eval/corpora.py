# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Materialise registry corpora to JSONL and verify them against the lock file.

Each builder returns a list of texts. Sampling always uses a generator seeded
from the global SEED and the corpus id, so a corpus never depends on which
other corpora were built before it.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tarfile
from pathlib import Path

from .registry import CORPORA, SEED, Corpus

PKG_DATA = Path(__file__).resolve().parent / "data"   # installed with the package
SHIPPED = PKG_DATA / "shipped"
LOCK = PKG_DATA / "corpora.lock.json"


class OfflineError(RuntimeError):
    """A corpus is missing from the cache and downloads are disabled."""


def offline() -> bool:
    return os.environ.get("BTE_OFFLINE", "") not in ("", "0")


def default_cache() -> Path:
    return Path(os.environ.get("BTE_CACHE", Path.home() / ".cache" / "brahmai-tokenizer-eval"))


def _rng(corpus_id: str) -> random.Random:
    return random.Random(f"{SEED}:{corpus_id}")


def _hf(repo: str, path: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision,
                           token=os.environ.get("HF_TOKEN"))


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── builders ──────────────────────────────────────────────────────────────────

def b_hf_parquet_column(c: Corpus, cache: Path) -> list[str]:
    import pandas as pd
    a = c.args
    return pd.read_parquet(_hf(a["repo"], a["file"], a["revision"]))[a["column"]].tolist()


def b_url_tar_lines(c: Corpus, cache: Path) -> list[str]:
    a = c.args
    raw = cache / "_raw" / Path(a["url"]).name
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        part = raw.with_name(raw.name + ".part")
        try:
            subprocess.run(["curl", "-fsSL", "-o", str(part), a["url"]], check=True)
        except FileNotFoundError:
            raise RuntimeError("downloading this corpus needs `curl`; or install the data archive") from None
        except subprocess.CalledProcessError as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"download failed: {a['url']} (curl exit {exc.returncode})") from None
        part.replace(raw)
    if _sha256_file(raw) != a["sha256"]:
        raw.unlink()   # a truncated or changed download must not be reused
        raise RuntimeError(f"checksum mismatch for {a['url']}; the file was removed, rerun to download it again")
    with tarfile.open(raw) as tf:
        names = set(tf.getnames())
        member = a["member"] if a["member"] in names else "./" + a["member"]
        f = tf.extractfile(member)
        return [l.rstrip("\n") for l in f.read().decode("utf-8").split("\n") if l.strip()]


def b_shipped(c: Corpus, cache: Path) -> list[str]:
    p = SHIPPED / c.args["file"]
    if p.suffix == ".jsonl":
        return [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]


def b_heap(c: Corpus, cache: Path) -> list[str]:
    import pandas as pd
    from huggingface_hub import HfApi
    a = c.args
    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(a["repo"], repo_type="dataset",
                                                                      revision=a["revision"])
    shard = sorted(f for f in files if f.startswith(f"data/{a['language']}/"))[0]
    df = pd.read_parquet(_hf(a["repo"], shard, a["revision"]))
    df = df[~(df.exact_duplicates_stackv1 | df.exact_duplicates_stackv2
              | df.near_duplicates_stackv1 | df.near_duplicates_stackv2)]
    df = df[df.content.str.len().between(a["min_chars"], a["max_chars"])]
    idx = sorted(_rng(c.id).sample(range(len(df)), min(a["n"], len(df))))
    return df.iloc[idx].content.tolist()


def b_hf_csv_sample(c: Corpus, cache: Path) -> list[str]:
    import pandas as pd
    a = c.args
    col = pd.read_csv(_hf(a["repo"], a["file"], a["revision"]))[a["column"]].dropna().astype(str)
    col = col[col.str.len().between(a["min_chars"], a["max_chars"])].tolist()
    idx = sorted(_rng(c.id).sample(range(len(col)), min(a["n"], len(col))))
    return [col[i] for i in idx]


def b_hf_parquet_concat(c: Corpus, cache: Path) -> list[str]:
    import pandas as pd
    from huggingface_hub import HfApi
    a = c.args
    files = a.get("files") or sorted(
        f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            a["repo"], repo_type="dataset", revision=a["revision"]) if f.endswith(a["suffix"]))
    out = []
    for f in files:
        df = pd.read_parquet(_hf(a["repo"], f, a["revision"]))
        out += [a["template"].format(**r) for r in df.to_dict("records")]
    return out


def b_hf_parquet_sample(c: Corpus, cache: Path) -> list[str]:
    import pandas as pd
    a = c.args
    col = [t for t in pd.read_parquet(_hf(a["repo"], a["file"], a["revision"]), columns=[a["column"]])[a["column"]].tolist()
           if isinstance(t, str) and len(t) > a["min_chars"]]
    idx = sorted(_rng(c.id).sample(range(len(col)), min(a["n"], len(col))))
    return [col[i] for i in idx]


def _flat(v) -> str:
    """Join list-valued fields (answer options) one per line."""
    if isinstance(v, dict) and "text" in v:
        v = v["text"]
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "\n".join(str(x) for x in v)
    return "" if v is None else str(v)


def b_hf_wide_translation(c: Corpus, cache: Path) -> list[str]:
    """English original or one translation from a wide table, row-aligned across languages.

    Rows are never dropped per language (an empty translation yields an empty
    item, which is then skipped everywhere), and optional sampling is seeded by
    the parallel group so every language keeps the same rows.
    """
    import pandas as pd
    a = c.args
    df = pd.read_parquet(_hf(a["repo"], a["file"], a["revision"]))
    rows = range(len(df))
    if a.get("sample"):
        rows = sorted(random.Random(f"{SEED}:{c.parallel}").sample(range(len(df)), a["sample"]))
    langs_cols = [(en if a["language"] == "English" else tpl.format(L=a["language"])) for en, tpl in a["fields"]]
    all_cols = [[en] + [tpl.format(L=L) for L in _indic_eval_langs()] for en, tpl in a["fields"]]
    out = []
    for r in rows:
        rec = df.iloc[r]
        complete = all(_flat(rec[col]).strip() for group in all_cols for col in group if col in df.columns)
        out.append("\n".join(_flat(rec[col]) for col in langs_cols) if complete else "")
    return out


def _indic_eval_langs() -> list[str]:
    from .registry import INDIC_EVAL_LANGS
    return list(INDIC_EVAL_LANGS)


def b_milu_native(c: Corpus, cache: Path) -> list[str]:
    import pandas as pd
    a = c.args
    df = pd.read_parquet(_hf(a["repo"], f"{a['language']}/test-00000-of-00001.parquet", a["revision"]))
    df = df[~df.is_translated.astype(bool)]
    return [f"{r.question}\n{r.option1}\n{r.option2}\n{r.option3}\n{r.option4}" for r in df.itertuples()]


BUILDERS = {k[2:]: v for k, v in globals().items() if k.startswith("b_")}


# ── public API ────────────────────────────────────────────────────────────────

def path_for(corpus_id: str, cache: Path | None = None) -> Path:
    return (cache or default_cache()) / "corpora" / f"{corpus_id}.jsonl"


def build(corpus_ids: list[str], cache: Path | None = None, force: bool = False, on_step=None) -> dict:
    """Materialise corpora; returns {id: {"items", "bytes", "sha256"}}.

    `on_step(corpus_id, cached)` is called after each corpus.
    """
    cache = cache or default_cache()
    info = {}
    for cid in corpus_ids:
        cached = path_for(cid, cache).exists() and not force
        c = CORPORA[cid]
        p = path_for(cid, cache)
        if force or not p.exists():
            if offline() and c.source != "shipped":
                raise OfflineError(f"corpus {cid} is not in {cache / 'corpora'} and downloads are disabled "
                                   "(BTE_OFFLINE / --offline); install the data archive with `bte data install`")
            texts = [t for t in BUILDERS[c.source](c, cache) if isinstance(t, str) and t.strip()]
            if not texts:
                raise RuntimeError(f"corpus {cid} built no items; its source may have changed")
            p.parent.mkdir(parents=True, exist_ok=True)
            part = p.with_name(p.name + ".part")   # an interrupted build must not look cached
            with open(part, "w", encoding="utf-8") as f:
                for i, t in enumerate(texts):
                    f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
            part.replace(p)
        texts = load(cid, cache)
        info[cid] = {"items": len(texts), "bytes": sum(len(t.encode("utf-8")) for t in texts),
                     "sha256": _sha256_file(p)}
        if on_step:
            on_step(cid, cached)
    return info


def load(corpus_id: str, cache: Path | None = None) -> list[str]:
    with open(path_for(corpus_id, cache), encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def verify(info: dict) -> dict[str, str]:
    """Compare built corpora with the lock file; returns {id: status}."""
    lock = json.load(open(LOCK)) if LOCK.exists() else {}
    status = {}
    for cid, i in info.items():
        if cid not in lock:
            status[cid] = "unlocked"
        elif lock[cid]["sha256"] == i["sha256"]:
            status[cid] = "ok"
        else:
            status[cid] = "MISMATCH"
    return status


def write_lock(info: dict) -> None:
    lock = json.load(open(LOCK)) if LOCK.exists() else {}
    lock.update(info)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as f:
        json.dump(dict(sorted(lock.items())), f, indent=1)
