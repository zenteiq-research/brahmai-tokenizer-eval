# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Uniform loading and encoding for Hugging Face `tokenizers` files and tiktoken.

A tokenizer is named by a spec string:

    <org>/<repo>[@<revision>]  any Hugging Face model repository (the "hf:" prefix is optional)
    <path>/tokenizer.json      a local tokenizer.json file
    <path>/                    a local directory that contains tokenizer.json
    tiktoken:<encoding>        an OpenAI tiktoken encoding, e.g. tiktoken:o200k_base

For a Hub repository the loader uses, in order: tokenizer.json at the root; a
tokenizer.json in a subfolder; otherwise transformers.AutoTokenizer (converted to
a fast tokenizer when possible, or wrapped as a slow tokenizer). Repositories
whose tokenizer is custom code load only with trust_remote_code=True.

Only tokenizer files are downloaded, never model weights. Encoding never adds
special tokens: the suite measures how text is segmented, not how a model's
prompt template wraps it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import regex


def _gpt2_byte_decoder() -> dict[str, int]:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DEC = _gpt2_byte_decoder()
_SPM_BYTE = regex.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@dataclass
class Tokenizer:
    """A loaded tokenizer with a uniform interface."""

    name: str
    spec: str
    kind: str                      # "hf", "tiktoken" or "slow" (transformers Python tokenizer)
    source: dict = field(default_factory=dict)
    _hf: object = None
    _tt: object = None
    _json: dict | None = None
    _tf: object = None

    # ── encoding ──────────────────────────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        if self.kind == "tiktoken":
            return self._tt.encode_ordinary(text)
        if self.kind == "slow":
            return self._tf.encode(text, add_special_tokens=False)
        return self._hf.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self.kind == "tiktoken":
            return self._tt.encode_ordinary_batch(texts)
        if self.kind == "slow":
            return [self._tf.encode(t, add_special_tokens=False) for t in texts]
        return [e.ids for e in self._hf.encode_batch(texts, add_special_tokens=False)]

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids with character offsets into `text`."""
        if self.kind == "tiktoken":
            ids = self._tt.encode_ordinary(text)
            raw, spans, pos = text.encode("utf-8"), [], 0
            for i in ids:
                start = len(raw[:pos].decode("utf-8", errors="ignore"))
                pos += len(self._tt.decode_single_token_bytes(i))
                spans.append((start, len(raw[:pos].decode("utf-8", errors="ignore"))))
            return ids, spans
        if self.kind == "slow":
            ids = self.encode(text)
            spans, prev = [], 0
            for k in range(1, len(ids) + 1):  # prefix decoding; slow but exact for lossless tokenizers
                cur = len(self.decode(ids[:k]))
                spans.append((prev, cur))
                prev = cur
            return ids, spans
        e = self._hf.encode(text, add_special_tokens=False)
        return e.ids, list(e.offsets)

    def decode(self, ids: list[int]) -> str:
        if self.kind == "tiktoken":
            return self._tt.decode(ids)
        if self.kind == "slow":
            return self._tf.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return self._hf.decode(ids, skip_special_tokens=False)

    # ── vocabulary ────────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        if self.kind == "tiktoken":
            return self._tt.n_vocab
        if self.kind == "slow":
            return len(self._tf)
        return self._hf.get_vocab_size(with_added_tokens=True)

    def token_bytes(self) -> tuple[dict[int, bytes], set[int]]:
        """Raw bytes of every ordinary token, and the set of special-token ids.

        Byte-level BPE entries are decoded through the GPT-2 byte map;
        SentencePiece-style entries map "▁" to a space and <0xXX> to a byte.
        """
        if self.kind == "tiktoken":
            table = {}
            for i in range(self._tt.n_vocab):
                try:
                    table[i] = self._tt.decode_single_token_bytes(i)
                except KeyError:
                    pass
            return table, set(self._tt._special_tokens.values())
        if self.kind == "slow":
            vocab = self._tf.get_vocab()
            special = set(self._tf.all_special_ids)
            table = {}
            for tok, i in vocab.items():
                if i in special:
                    continue
                if isinstance(tok, bytes):
                    table[i] = tok
                    continue
                try:
                    table[i] = bytes(_BYTE_DEC[c] for c in tok)
                except KeyError:
                    m = _SPM_BYTE.match(tok)
                    table[i] = bytes([int(m.group(1), 16)]) if m else tok.replace("▁", " ").encode("utf-8")
            return table, special
        tj = self._json
        special = {a["id"] for a in tj.get("added_tokens", []) if a.get("special")}
        added = {a["id"]: a["content"] for a in tj.get("added_tokens", [])}
        vocab = tj["model"]["vocab"]
        if isinstance(vocab, list):  # Unigram
            vocab = {p[0]: i for i, p in enumerate(vocab)}
        bytelevel = ("ByteLevel" in json.dumps(tj.get("pre_tokenizer"))
                     or "ByteLevel" in json.dumps(tj.get("decoder")))
        table: dict[int, bytes] = {}
        for tok, i in vocab.items():
            if i in special:
                continue
            if bytelevel and i not in added:
                try:
                    table[i] = bytes(_BYTE_DEC[c] for c in tok)
                    continue
                except KeyError:
                    pass
            m = _SPM_BYTE.match(tok)
            table[i] = bytes([int(m.group(1), 16)]) if m else tok.replace("▁", " ").encode("utf-8")
        for i, c in added.items():
            if i not in special:
                table[i] = c.encode("utf-8")
        return table, special

    def fingerprint(self) -> str:
        """SHA-256 over the sorted token -> id map (ignores special-token metadata)."""
        if self.kind in ("tiktoken", "slow"):
            items = sorted((b.hex(), i) for i, b in self.token_bytes()[0].items())
        else:
            v = self._json["model"]["vocab"]
            items = sorted(v.items()) if isinstance(v, dict) else [(p[0], i) for i, p in enumerate(v)]
        return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest()

    def pipeline(self) -> dict:
        """Normaliser, pre-tokenizer, model type and byte fallback, for reporting."""
        if self.kind == "tiktoken":
            return {"model": "tiktoken BPE", "normalizer": None, "pre_tokenizer": "tiktoken regex"}
        if self.kind == "slow":
            return {"model": type(self._tf).__name__, "normalizer": "n/a (slow tokenizer)", "pre_tokenizer": "n/a"}
        tj = self._json
        return {"model": model_type(tj["model"]), "byte_fallback": tj["model"].get("byte_fallback"),
                "normalizer": tj.get("normalizer"), "pre_tokenizer": tj.get("pre_tokenizer")}


def model_type(model: dict) -> str:
    """Model type of a tokenizer.json; older files (e.g. GPT-2's) omit "type", so infer it."""
    if model.get("type"):
        return model["type"]
    if "merges" in model:
        return "BPE"
    if isinstance(model.get("vocab"), list):
        return "Unigram"
    if "continuing_subword_prefix" in model or "max_input_chars_per_word" in model:
        return "WordPiece"
    return "unknown"


def default_name(spec: str) -> str:
    """Readable name for a spec: the repository name, file stem or encoding."""
    if spec.startswith("tiktoken:"):
        return spec.split(":", 1)[1]
    p = Path(spec)
    if p.exists():
        return (p if p.is_dir() else p.parent).name
    return spec.removeprefix("hf:").split("@")[0].split("/")[-1]


def _is_local(spec: str) -> bool:
    return Path(spec).exists()


def _looks_like_path(spec: str) -> bool:
    """A spec that cannot be a Hub id: starts like a path or names a file."""
    return (spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json")
            or spec.count("/") > 1 and not spec.startswith("hf:"))


def load(spec: str, name: str | None = None, token: str | None = None,
         trust_remote_code: bool = False) -> Tokenizer:
    """Load a tokenizer from a spec string (see module docstring)."""
    token = token or os.environ.get("HF_TOKEN")
    name = name or default_name(spec)
    if spec.startswith("tiktoken:"):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(f"{spec} needs tiktoken: pip install 'brahmai-tokenizer-eval[tiktoken]'") from exc
        enc = spec.split(":", 1)[1]
        try:
            tt = tiktoken.get_encoding(enc)
        except ValueError as exc:
            raise RuntimeError(f"unknown tiktoken encoding {enc!r}; known: {tiktoken.list_encoding_names()}") from None
        return Tokenizer(name, spec, "tiktoken", {"encoding": enc}, _tt=tt)
    if _is_local(spec):
        p = Path(spec)
        path = p / "tokenizer.json" if p.is_dir() else p
        if not path.is_file():
            raise RuntimeError(f"{spec}: directory has no tokenizer.json")
        return _from_file(name, spec, str(path), {"path": str(path)})
    if _looks_like_path(spec):
        raise RuntimeError(f"{spec}: no such file or directory")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import (GatedRepoError, HFValidationError, LocalEntryNotFoundError,
                                        RepositoryNotFoundError, RevisionNotFoundError)
    repo, _, rev = spec.removeprefix("hf:").partition("@")
    source = {"repo": repo}
    try:
        info = HfApi(token=token).model_info(repo, revision=rev or None)
        rev = info.sha
        files = [s.rfilename for s in info.siblings]
        candidates = sorted((f for f in files if f == "tokenizer.json" or f.endswith("/tokenizer.json")),
                            key=lambda f: (f.count("/"), f))
        source["revision"] = rev
        if candidates:
            path = hf_hub_download(repo, candidates[0], revision=rev, token=token)
            source["file"] = candidates[0]
            return _from_file(name, spec, path, source)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"{name}: {repo} is gated and HF_TOKEN has no access. Request access at "
            f"https://huggingface.co/{repo}, or place the pinned tokenizer.json in the Hugging Face "
            f"cache (the manifest's SHA-256 check makes any copy of the right file equivalent).") from None
    except RepositoryNotFoundError:
        raise RuntimeError(f"{spec}: not a local file and not a Hugging Face model repository you can access "
                           "(check the name, or set HF_TOKEN for a private repository)") from None
    except RevisionNotFoundError:
        raise RuntimeError(f"{spec}: revision {rev!r} not found in {repo}") from None
    except HFValidationError:
        raise RuntimeError(f"{spec}: not a local file and not a valid Hugging Face repository id") from None
    except (OSError, LocalEntryNotFoundError) as exc:   # offline or network error: use the local cache
        path, rev = _cached_tokenizer_json(repo, rev or None)
        if path is None:
            raise RuntimeError(f"{spec}: the Hub is unreachable ({type(exc).__name__}) and tokenizer.json is not "
                               "in the local Hugging Face cache (or several revisions are cached: add @<commit>)") from None
        source.update(revision=rev, file="tokenizer.json", offline_cache=True)
        return _from_file(name, spec, path, source)

    # no tokenizer.json: fall back to transformers (SentencePiece models, vocab/merges, custom code)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"{repo} has no tokenizer.json; install transformers (pip install "
                           f"'brahmai-tokenizer-eval[transformers]') to load it") from exc
    try:
        tf = AutoTokenizer.from_pretrained(repo, revision=rev, token=token, trust_remote_code=trust_remote_code)
    except Exception as exc:
        hint = ("" if trust_remote_code else
                " If the repository implements its tokenizer as custom code and you trust it, "
                "rerun with --trust-remote-code.")
        raise RuntimeError(f"could not load a tokenizer for {repo}@{(rev or '')[:10]}: {type(exc).__name__}: {exc}.{hint}") from exc
    source["loader"] = f"transformers.{type(tf).__name__}"
    if getattr(tf, "is_fast", False):
        tj = json.loads(tf.backend_tokenizer.to_str())
        source["converted_to_fast"] = True
        return Tokenizer(name, spec, "hf", source, _hf=HFTokenizer.from_str(json.dumps(tj)), _json=tj)
    return Tokenizer(name, spec, "slow", source, _tf=tf)


def _cached_tokenizer_json(repo: str, rev: str | None) -> tuple[str | None, str | None]:
    """tokenizer.json of `repo` in the local Hugging Face cache, and its commit.

    Without a revision, use the cached "main" if recorded, else the only cached snapshot; files
    downloaded at a pinned commit leave no "main" entry, and with several snapshots the choice
    would be a guess.
    """
    from huggingface_hub import constants, try_to_load_from_cache
    p = try_to_load_from_cache(repo, "tokenizer.json", revision=rev)
    if isinstance(p, str):
        return p, rev or Path(p).parent.name
    if rev is None:
        snaps = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots"
        found = sorted(snaps.glob("*/tokenizer.json")) if snaps.is_dir() else []
        if len(found) == 1:
            return str(found[0]), found[0].parent.name
    return None, None


def _from_file(name: str, spec: str, path: str, source: dict) -> Tokenizer:
    from tokenizers import Tokenizer as HFTokenizer
    with open(path, "rb") as f:
        raw = f.read()
    try:
        tj = json.loads(raw.decode("utf-8"))
        hf = HFTokenizer.from_str(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{path}: not a valid tokenizer.json ({type(exc).__name__}: {str(exc)[:120]})") from None
    source["tokenizer_json_sha256"] = hashlib.sha256(raw).hexdigest()
    return Tokenizer(name, spec, "hf", source, _hf=hf, _json=tj)
