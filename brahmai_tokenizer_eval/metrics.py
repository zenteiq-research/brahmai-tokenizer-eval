# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Efficiency metrics with item-level bootstrap intervals.

All metrics are computed from per-item counts (one item = one sentence,
document or source file), so every aggregate can carry a confidence interval
and two tokenizers can be compared on exactly the same items.

Definitions (T = tokens, B = UTF-8 bytes, C = Unicode code points, W = whitespace
words, summed over the items of a corpus):

  tokens per 1,000 bytes  1000 * T / B      lower is better; the primary metric
  bytes per token         B / T             higher is better; reciprocal of the above
  characters per token    C / T
  fertility               T / W             only for scripts that separate words with spaces
  paired ratio            T_a / T_b         same items, tokenizer a vs tokenizer b
  token premium           T_lang / T_eng    parallel corpora only; cost relative to English
  byte premium            B_lang / B_eng    parallel corpora only; no tokenizer involved
  efficiency index        geometric mean over declared domain groups of T_ref / T_tok,
                          against one fixed reference tokenizer (secondary summary only)

The primary metric is normalised by bytes because bytes are the one length
unit that is identical for every tokenizer and every script; word counts
depend on orthography and character counts on Unicode encoding choices.
Bytes are not identical across languages, though: the same content takes more
UTF-8 bytes in some scripts, so tokens per byte compares tokenizers on one
corpus, never languages with each other. Across languages, use the token
premium on parallel text, and the byte premium to show how much of it comes
from the encoding.

Intervals are percentile bootstrap intervals over items (B = 1,000
resamples, fixed seed). For paired ratios and premiums the same resampled
item indices are used in numerator and denominator.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import corpora as C
from .registry import CORPORA

N_BOOT = 1000
BOOT_SEED = 12345


@dataclass
class Counts:
    corpus: str
    bytes: np.ndarray
    chars: np.ndarray
    words: np.ndarray
    tokens: dict[str, np.ndarray]
    roundtrip_failures: dict[str, list[int]]
    used_ids: dict[str, set] | None = None
    # per tokenizer, one entry per item of roundtrip_failures: its kind (see roundtrip_kind) and the
    # number of UTF-8 bytes by which decoded and input text differ (after removing one added leading
    # space), so losses can be quantified without re-tokenizing
    roundtrip_kinds: dict[str, list[str]] = field(default_factory=dict)
    roundtrip_byte_diff: dict[str, list[int]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"corpus": self.corpus, "bytes": self.bytes.tolist(), "chars": self.chars.tolist(),
                "words": self.words.tolist(), "tokens": {k: v.tolist() for k, v in self.tokens.items()},
                "roundtrip_failures": self.roundtrip_failures, "roundtrip_kinds": self.roundtrip_kinds,
                "roundtrip_byte_diff": self.roundtrip_byte_diff}

    @classmethod
    def from_json(cls, d: dict) -> "Counts":
        return cls(d["corpus"], np.array(d["bytes"]), np.array(d["chars"]), np.array(d["words"]),
                   {k: np.array(v) for k, v in d["tokens"].items()}, d["roundtrip_failures"],
                   roundtrip_kinds=d.get("roundtrip_kinds", {}), roundtrip_byte_diff=d.get("roundtrip_byte_diff", {}))

    def lossy(self, name: str) -> int:
        """Items whose decoded text differs from the input other than by NFC or a leading space."""
        return sum(k == "lossy" for k in self.roundtrip_kinds.get(name, []))


def roundtrip_kind(text: str, decoded: str) -> str:
    """Classify a round-trip failure.

    "nfc"            decoded == NFC(text): the tokenizer normalises; counts measure the NFC form
    "leading_space"  decoded == " " + text: the decoder adds one space; nothing is lost
    "lossy"          anything else: characters were dropped or changed, so the count is too low
    """
    if decoded == unicodedata.normalize("NFC", text):
        return "nfc"
    if decoded == " " + text or (decoded.startswith(" ") and decoded[1:] == unicodedata.normalize("NFC", text)):
        return "leading_space"
    return "lossy"


def byte_diff(text: str, decoded: str) -> int:
    """|bytes(text) - bytes(decoded)|, ignoring one leading space the decoder added."""
    if decoded.startswith(" ") and not text.startswith(" "):
        decoded = decoded[1:]
    return abs(len(text.encode("utf-8")) - len(decoded.encode("utf-8")))


def count(corpus_id: str, tokenizers: list, cache: Path | None = None, on_step=None) -> Counts:
    """Per-item counts for every tokenizer; `on_step(name)` is called after each tokenizer."""
    texts = C.load(corpus_id, cache)
    tok_counts, fails, kinds, diffs, used = {}, {}, {}, {}, {}
    for tk in tokenizers:
        ids = tk.encode_batch(texts)
        tok_counts[tk.name] = np.array([len(x) for x in ids])
        decoded = [tk.decode(x) for x in ids]
        fails[tk.name] = [i for i, (t, d) in enumerate(zip(texts, decoded)) if d != t]
        kinds[tk.name] = [roundtrip_kind(texts[i], decoded[i]) for i in fails[tk.name]]
        diffs[tk.name] = [byte_diff(texts[i], decoded[i]) for i in fails[tk.name]]
        used[tk.name] = {i for x in ids for i in x}
        if on_step:
            on_step(tk.name)
    return Counts(corpus_id,
                  np.array([len(t.encode("utf-8")) for t in texts]),
                  np.array([len(t) for t in texts]),
                  np.array([len(t.split()) for t in texts]),
                  tok_counts, fails, used, kinds, diffs)


def merge(counts: list[Counts]) -> Counts:
    """Concatenate item arrays of several Counts (same tokenizers) into one pool."""
    if not counts:
        raise ValueError("merge needs at least one Counts")
    names = set.intersection(*(set(c.tokens) for c in counts))
    return Counts("+".join(c.corpus for c in counts),
                  np.concatenate([c.bytes for c in counts]), np.concatenate([c.chars for c in counts]),
                  np.concatenate([c.words for c in counts]),
                  {n: np.concatenate([c.tokens[n] for c in counts]) for n in names},
                  {n: [] for n in names})


def _boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))


def _ci(num: np.ndarray, den: np.ndarray, scale: float = 1.0) -> tuple[float, float, float]:
    if len(num) == 0:
        raise ValueError("no items: the corpus is empty")
    idx = _boot_idx(len(num))
    est = scale * num.sum() / den.sum()
    bs = scale * num[idx].sum(1) / den[idx].sum(1)
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summary(c: Counts, name: str) -> dict:
    t = c.tokens[name]
    out = {"items": int(len(t)), "tokens": int(t.sum()), "bytes": int(c.bytes.sum()),
           "tokens_per_kb": _ci(t, c.bytes, 1000.0),
           "bytes_per_token": float(c.bytes.sum() / t.sum()),
           "chars_per_token": float(c.chars.sum() / t.sum()),
           "p50_tokens_per_item": float(np.percentile(t, 50)),
           "p95_tokens_per_item": float(np.percentile(t, 95)),
           "roundtrip_failures": len(c.roundtrip_failures.get(name, [])),
           "roundtrip_lossy": c.lossy(name)}
    corp = CORPORA.get(c.corpus)
    if corp is None or corp.whitespace_words:
        out["fertility"] = _ci(t, c.words)
    return out


def paired_ratio(c: Counts, a: str, b: str) -> tuple[float, float, float]:
    """T_a / T_b on the same items, with a paired bootstrap interval."""
    return _ci(c.tokens[a], c.tokens[b])


def premium(c_lang: Counts, c_eng: Counts, name: str) -> tuple[float, float, float]:
    """Token premium of a language over English on a parallel corpus."""
    if len(c_lang.bytes) != len(c_eng.bytes):
        raise ValueError("premium needs sentence-aligned corpora of equal length")
    return _ci(c_lang.tokens[name], c_eng.tokens[name])


def byte_premium(c_lang: Counts, c_eng: Counts) -> tuple[float, float, float]:
    """UTF-8 bytes of a language over English on a parallel corpus (Arnett et al. 2024); no tokenizer.

    Tokens per byte cannot be compared across languages: the same sentence takes about one byte per
    character in Latin script and about three in the Indian scripts. On parallel text,
    tokens-per-byte ratio x byte premium = token premium, so this separates encoding from tokenization.
    """
    if len(c_lang.bytes) != len(c_eng.bytes):
        raise ValueError("byte_premium needs sentence-aligned corpora of equal length")
    return _ci(c_lang.bytes, c_eng.bytes)


def group_premium(members: list[Counts], c_eng: Counts, name: str) -> tuple[float, float, float]:
    """Pooled token premium of a group of languages over English on one parallel corpus.

    Sum of the group's tokens over the tokens of the English sentences counted once per member, so
    the value is the average premium weighted by each language's tokens (Petrov et al. 2023 report
    per-language premiums; this pools them as the tables pool tokens per byte). Bootstrap resamples
    sentences, with the same sentence indices in every language and in English.
    """
    if not members:
        raise ValueError("group_premium needs at least one member")
    if any(len(c.bytes) != len(c_eng.bytes) for c in members):
        raise ValueError("group_premium needs sentence-aligned corpora of equal length")
    num = np.sum([c.tokens[name] for c in members], axis=0)
    return _ci(num, len(members) * c_eng.tokens[name])


def group_compare(members: dict[str, Counts], a: str, b: str) -> dict:
    """How many tokens tokenizer a needs relative to b over a group of corpora (e.g. 12 languages).

    Two summaries, because they answer different questions and can differ a lot:

      pooled     sum T_a / sum T_b over all items of all members. Members weigh by their token
                 count, so one member where a is very poor can dominate. This is the ratio of the
                 pooled tokens-per-byte values shown in group tables.
      geomean    geometric mean over members of (sum T_a / sum T_b) within each member. Every
                 member weighs equally.

    Both carry 95% bootstrap intervals (pooled: items resampled across the pool, as in
    paired_ratio; geomean: items resampled within each member, same indices for a and b).
    `median`, `min` and `max` give the spread of the per-member ratios, with the member named,
    so a reader can see when one member drives the average.
    """
    if not members:
        raise ValueError("group_compare needs at least one member")
    pool = merge(list(members.values()))
    per = {m: float(c.tokens[a].sum() / c.tokens[b].sum()) for m, c in members.items()}
    logs = []
    for c in members.values():
        idx = _boot_idx(len(c.bytes))
        logs.append(np.log(c.tokens[a][idx].sum(1) / c.tokens[b][idx].sum(1)))
    boot = np.exp(np.mean(np.vstack(logs), axis=0))
    geo = float(np.exp(np.mean(np.log(list(per.values())))))
    lo_m, hi_m = min(per, key=per.get), max(per, key=per.get)
    return {"a": a, "b": b, "members": len(per),
            "pooled": paired_ratio(pool, a, b),
            "geomean": (geo, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "median": float(np.median(list(per.values()))),
            "min": (lo_m, per[lo_m]), "max": (hi_m, per[hi_m]),
            "per_member": per}


def efficiency_index(groups: dict[str, list[Counts]], name: str, reference: str) -> dict:
    """Relative efficiency index: geometric mean over domain groups of T_ref / T_name.

    Each group (e.g. "English", "Indic", "code") pools its corpora; the group
    ratio is reference tokens over tokenizer tokens on the same items, so a
    value of 1.12 means 12% fewer tokens than the reference. Groups are
    weighted equally; the caller decides the group list and must report it.
    The index depends only on the two tokenizers and the corpora, never on
    which other tokenizers are evaluated. The interval resamples items within
    every group with the same indices for both tokenizers.
    """
    logs, boots, per = [], [], {}
    for g, cs in groups.items():
        c = merge(cs)
        a, b = c.tokens[reference], c.tokens[name]
        idx = _boot_idx(len(a))
        r = a.sum() / b.sum()
        per[g] = float(r)
        logs.append(np.log(r))
        boots.append(np.log(a[idx].sum(1) / b[idx].sum(1)))
    est = float(np.exp(np.mean(logs)))
    bs = np.exp(np.mean(np.vstack(boots), axis=0))
    return {"index": est, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "reference": reference, "groups": per}


def save(counts: Counts, out_dir: Path) -> None:
    p = out_dir / "counts" / f"{counts.corpus}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(counts.to_json(), f)


def load_counts(out_dir: Path, corpus_id: str) -> Counts:
    with open(out_dir / "counts" / f"{corpus_id}.json") as f:
        return Counts.from_json(json.load(f))
