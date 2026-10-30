# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic probes that do not need an evaluation corpus.

  unicode_coverage   round trip of every assigned code point (incl. private use)
  reachability       vocabulary entries that cannot be produced from their own text
  numbers            segmentation of integers, decimals, separators, Indic digits
  normalisation      sensitivity to NFC/NFD and to ZWJ/ZWNJ removal
  composition        vocabulary entries by dominant script; cross-word entries
  special_collision  plain-text occurrences of special-token strings parsed as control tokens
  utf8_boundaries    token boundaries that fall inside a multi-byte UTF-8 character
  throughput         encoding speed (bytes/s, interleaved rounds, relative to a reference) and its
                     scaling with input length

Each probe returns plain JSON-serialisable dicts.
"""
from __future__ import annotations

import numpy as np

import collections
import random
import unicodedata

import regex

from .registry import SEED

SCRIPTS = ["Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada",
           "Malayalam", "Sinhala", "Ol_Chiki", "Meetei_Mayek", "Arabic", "Hebrew", "Armenian", "Syriac",
           "Thaana", "Nko", "Myanmar", "Thai", "Latin", "Cyrillic", "Greek", "Han", "Hiragana",
           "Katakana", "Hangul"]
_SCRIPT_RE = {s: regex.compile(rf"\p{{Script={s}}}") for s in SCRIPTS}
INDIC = {"Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada",
         "Malayalam", "Ol_Chiki", "Meetei_Mayek"}


def script_of(ch: str) -> str:
    for s in SCRIPTS:
        if _SCRIPT_RE[s].match(ch):
            return s
    return "Common/other"


# ── Unicode coverage ──────────────────────────────────────────────────────────

def unicode_coverage(tok) -> dict:
    """Encode each assigned code point inside 'a?b' and check the round trip.

    Surrogates and unassigned code points are skipped; private-use code points
    are included because real text (fonts, legacy encodings) contains them.
    """
    cps = [cp for cp in range(0x20, 0x110000)
           if not (0xD800 <= cp <= 0xDFFF) and unicodedata.category(chr(cp)) not in ("Cn", "Cs")]
    texts = [f"a{chr(cp)}b" for cp in cps]
    lost = []
    for i in range(0, len(texts), 20000):
        batch = texts[i:i + 20000]
        for t, ids in zip(batch, tok.encode_batch(batch)):
            if tok.decode(ids) != t:
                lost.append(ord(t[1]))
    by_lead = collections.Counter(chr(cp).encode("utf-8")[0] for cp in lost)
    by_script = collections.Counter(script_of(chr(cp)) for cp in lost)
    by_cat = collections.Counter(unicodedata.category(chr(cp)) for cp in lost)
    controls = {name: (tok.decode(tok.encode(s)) == s) for name, s in
                [("LF", "a\nb"), ("CR", "a\rb"), ("CRLF", "a\r\nb"), ("TAB", "a\tb"), ("NUL", "a\x00b")]}
    return {"code_points_tested": len(cps), "code_points_lost": len(lost),
            "lost_by_utf8_lead_byte": {f"0x{k:02X}": v for k, v in sorted(by_lead.items())},
            "lost_by_script": dict(by_script.most_common()), "lost_by_category": dict(by_cat.most_common()),
            "control_round_trips": controls,
            "examples_lost": [f"U+{cp:04X} {unicodedata.name(chr(cp), '?')}" for cp in lost[:: max(1, len(lost) // 15)][:15]]}


def missing_byte_symbols(tok) -> list[str]:
    """Single-byte entries absent from a byte-level vocabulary (empty if not byte-level)."""
    if tok.kind != "hf" or "ByteLevel" not in str(tok._json.get("pre_tokenizer")):
        return []
    table, _ = tok.token_bytes()
    present = {b[0] for b in table.values() if len(b) == 1}
    return [f"0x{b:02X}" for b in range(256) if b not in present]


# ── reachability ──────────────────────────────────────────────────────────────

def reachability(tok) -> dict:
    """Entries whose own decoded text does not encode back to exactly that entry."""
    table, _ = tok.token_bytes()
    ids, strs = [], []
    for i, b in table.items():
        try:
            strs.append(b.decode("utf-8"))
            ids.append(i)
        except UnicodeDecodeError:
            pass  # partial UTF-8 pieces cannot be supplied as text
    enc = tok.encode_batch(strs)
    bad = [i for i, e in zip(ids, enc) if e != [i]]
    return {"decodable_entries": len(ids), "not_self_encoding": len(bad),
            "share": len(bad) / max(len(ids), 1),
            "examples": [table[i].decode("utf-8", "replace") for i in bad[:: max(1, len(bad) // 20)][:20]]}


# ── numbers ───────────────────────────────────────────────────────────────────

INDIC_DIGITS = {"Devanagari": "०१२३४५६७८९", "Bengali": "০১২৩৪৫৬৭৮৯", "Gujarati": "૦૧૨૩૪૫૬૭૮૯",
                "Gurmukhi": "੦੧੨੩੪੫੬੭੮੯", "Oriya": "୦୧୨୩୪୫୬୭୮୯", "Tamil": "௦௧௨௩௪௫௬௭௮௯",
                "Telugu": "౦౧౨౩౪౫౬౭౮౯", "Kannada": "೦೧೨೩೪೫೬೭೮೯", "Malayalam": "൦൧൨൩൪൫൬൭൮൯"}
_NUM = regex.compile(r"\d+(?:[.,]\d+)*")


def numbers(tok, running_text: list[str] | None = None) -> dict:
    texts = [f" {n}" for n in range(10000)]
    enc = tok.encode_batch(texts)
    single = 0
    for n, (t, ids) in enumerate(zip(texts, enc)):
        _, spans = tok.encode_offsets(t)
        pieces = [t[a:b].strip() for a, b in spans if t[a:b].strip()]
        single += all(len(p) == 1 and p.isdigit() for p in pieces) and len(pieces) == len(str(n))
    out = {"integers_0_9999": {"mean_tokens": sum(map(len, enc)) / 10000,
                               "share_single_digit": single / 10000,
                               "token_count_distribution": dict(sorted(collections.Counter(map(len, enc)).items()))}}
    rng = random.Random(SEED)
    probes = {
        "decimal": [f" {rng.randint(0, 999)}.{rng.randint(0, 999):03d}" for _ in range(500)],
        "scientific": [f" {rng.randint(1, 9)}.{rng.randint(0, 99):02d}e-{rng.randint(1, 30)}" for _ in range(500)],
        "thousands_separator": [f" {rng.randint(1, 999)},{rng.randint(0, 999):03d},{rng.randint(0, 999):03d}" for _ in range(500)],
        "years_1900_2030": [f" {y}" for y in range(1900, 2031)],
    }
    for s, digs in INDIC_DIGITS.items():
        probes[f"digits_{s}"] = [" " + "".join(digs[int(c)] for c in str(rng.randint(0, 99999))) for _ in range(500)]
    out["tokens_per_char"] = {k: sum(map(len, tok.encode_batch(v))) / sum(len(x.strip()) for x in v)
                              for k, v in probes.items()}
    if running_text:
        fused = digit_tokens = differ = total = 0
        for t in running_text:
            _, spans = tok.encode_offsets(t)
            for a, b in spans:
                p = t[a:b]
                if regex.search(r"\d", p):
                    digit_tokens += 1
                    fused += bool(regex.search(r"[^\d]", p.strip(" ")))
            for m in _NUM.finditer(t):
                s, e = m.span()
                ctx = [regex.sub(r"[^\d.,]", "", t[a:b]) for a, b in spans if a < e and b > s]
                iso_t = " " + m.group()
                iso = [regex.sub(r"[^\d.,]", "", iso_t[a:b]) for a, b in tok.encode_offsets(iso_t)[1]]
                total += 1
                differ += [x for x in ctx if x] != [x for x in iso if x]
        out["running_text"] = {"digit_tokens": digit_tokens, "share_fused_with_non_digit": fused / max(digit_tokens, 1),
                               "numbers": total, "share_context_dependent": differ / max(total, 1)}
    return out


# ── normalisation ─────────────────────────────────────────────────────────────

def normalisation(tok, texts: list[str]) -> dict:
    """Token-count change and identity under NFC/NFD and ZWJ/ZWNJ removal."""
    def total(xs):
        return sum(map(len, tok.encode_batch(xs)))
    if not any(texts):
        raise ValueError("normalisation needs at least one non-empty text")
    nfc = [unicodedata.normalize("NFC", t) for t in texts]
    nfd = [unicodedata.normalize("NFD", t) for t in texts]
    nozw = [t.replace("‍", "").replace("‌", "") for t in texts]
    base = total(texts)
    same_nfc_nfd = sum(a == b for a, b in zip(tok.encode_batch(nfc), tok.encode_batch(nfd)))
    return {"items": len(texts), "changed_by_nfd": sum(a != b for a, b in zip(texts, nfd)),
            "contains_zw": sum(("‍" in t) or ("‌" in t) for t in texts),
            "nfd_over_original": total(nfd) / base, "nfc_over_original": total(nfc) / base,
            "no_zw_over_original": total(nozw) / base,
            "share_identical_ids_nfc_vs_nfd": same_nfc_nfd / len(texts)}


# ── composition ───────────────────────────────────────────────────────────────

def _classify(b: bytes) -> str:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return "partial-UTF-8"
    core = s.strip()
    if not core:
        return "whitespace"
    counts = collections.Counter()
    for ch in core:
        if ch.isascii() and ch.isdigit():
            counts["digit"] += 1
        elif ch.isalpha():
            counts[script_of(ch)] += 1
        else:
            counts["symbol"] += 1
    letters = {k: v for k, v in counts.items() if k not in ("digit", "symbol")}
    if not letters:
        return "digits/symbols"
    top = max(letters, key=letters.get)
    return "Kana" if top in ("Hiragana", "Katakana") else top


def composition(tok) -> dict:
    table, special = tok.token_bytes()
    comp, cross = collections.Counter(), collections.Counter()
    for b in table.values():
        c = _classify(b)
        comp[c] += 1
        s = b.decode("utf-8", errors="ignore")
        if regex.search(r"\S\s+\S", s):
            cross[c] += 1
    n = sum(comp.values())
    return {"ordinary_entries": n, "special_entries": len(special), "by_script": dict(comp.most_common()),
            "indic_share": sum(v for k, v in comp.items() if k in INDIC) / n,
            "cross_word_entries": sum(cross.values()), "cross_word_by_script": dict(cross.most_common())}


# ── special-token collision ───────────────────────────────────────────────────

def special_collision(tok) -> dict:
    """Does a special-token string typed as ordinary text become a control token?

    Chat templates rely on special tokens to separate roles. If user-supplied
    text containing "<|end_of_turn|>" is encoded to the real control id, a user
    can forge turn boundaries. Measured with special tokens *not* added, as
    when raw user text is encoded.
    """
    if tok.kind == "tiktoken":
        return {"special_tokens": len(tok._tt._special_tokens), "collide": 0, "note": "encode_ordinary never parses specials"}
    if tok.kind == "slow":
        specials = [(s, tok._tf.convert_tokens_to_ids(s)) for s in tok._tf.all_special_tokens]
    else:
        specials = [(a["content"], a["id"]) for a in tok._json.get("added_tokens", []) if a.get("special")]
    collide = [c for c, i in specials if i in tok.encode(f"user text {c} more text")]
    return {"special_tokens": len(specials), "collide": len(collide), "share": len(collide) / max(len(specials), 1),
            "examples": collide[:10]}


# ── UTF-8 boundary integrity ─────────────────────────────────────────────────

def utf8_boundaries(tok, texts: list[str]) -> dict:
    """Share of internal token boundaries that split a multi-byte character."""
    table, _ = tok.token_bytes()
    inside = total = 0
    for t, ids in zip(texts, tok.encode_batch(texts)):
        raw = t.encode("utf-8")
        starts = set()
        pos = 0
        for ch in t:
            starts.add(pos)
            pos += len(ch.encode("utf-8"))
        pos = 0
        for i in ids[:-1]:
            b = table.get(i)
            if b is None:
                break
            pos += len(b)
            total += 1
            inside += pos not in starts
    return {"internal_boundaries": total, "inside_character": inside, "share": inside / max(total, 1)}


# ── throughput ───────────────────────────────────────────────────────────────

ROUNDS = 7
MIN_SECONDS = 0.25   # each timing repeats the items until it lasts at least this long


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine()


def throughput(toks: list, texts: list[str], reference: str, rounds: int = ROUNDS) -> dict:
    """Single-thread encode speed of several tokenizers, measured so that results are comparable.

    Wall-clock speed depends on the machine and on its load at the moment of measurement, so a
    single timing per tokenizer is not reproducible. Here every round times every tokenizer on the
    same items, in an order rotated each round, so load affects all tokenizers alike. Reported per
    tokenizer: the median and range of bytes per second over rounds, and the median over rounds
    of its speed divided by the reference tokenizer's speed in the same round (the comparable
    number). Each timing repeats the items until it lasts MIN_SECONDS, so very fast tokenizers
    are not dominated by timer noise. Absolute bytes per second still move with machine load
    (about 10% between runs on a shared machine); relative speeds held to about 1-3%. `single_string_seconds` times one long string of growing length (median of three):
    a tokenizer without a pre-tokenizer runs BPE over the whole input, so cost can grow faster
    than linearly with length.
    """
    import os
    import time
    if not toks or not any(texts):
        raise ValueError("throughput needs at least one tokenizer and one non-empty text")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")   # encode() of one string is single-threaded
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    names = [t.name for t in toks]
    reps = {}
    for tk in toks:   # warm-up pass, also sets how many passes make one timing last MIN_SECONDS
        t0 = time.perf_counter()
        for t in texts:
            tk.encode(t)
        reps[tk.name] = max(1, int(np.ceil(MIN_SECONDS / max(time.perf_counter() - t0, 1e-6))))
    bps = {n: [] for n in names}
    for r in range(rounds):
        k = r % len(toks)
        for tk in toks[k:] + toks[:k]:
            t0 = time.perf_counter()
            for _ in range(reps[tk.name]):
                for t in texts:
                    tk.encode(t)
            bps[tk.name].append(reps[tk.name] * nbytes / (time.perf_counter() - t0))
    data = " ".join(texts).encode("utf-8")
    out = {}
    for tk in toks:
        scaling = {}
        for kb in (1, 10, 100, 1000):
            s = data[: kb * 1024].decode("utf-8", errors="ignore")
            if len(s.encode()) < kb * 900:
                break
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                tk.encode(s)
                ts.append(time.perf_counter() - t0)
            scaling[f"{kb}KB"] = round(float(np.median(ts)), 5)
        rel = [a / b for a, b in zip(bps[tk.name], bps[reference])] if reference in bps else []
        out[tk.name] = {"items_bytes_per_second": float(np.median(bps[tk.name])),
                        "items_bytes_per_second_min": float(min(bps[tk.name])),
                        "items_bytes_per_second_max": float(max(bps[tk.name])),
                        "relative_to_reference": float(np.median(rel)) if rel else None,
                        "passes_per_timing": reps[tk.name],
                        "single_string_seconds": scaling}
    return {"reference": reference, "rounds": rounds, "bytes_per_pass": nbytes, "min_seconds": MIN_SECONDS,
            "cpu": cpu_model(),
            "per_tokenizer": out}
