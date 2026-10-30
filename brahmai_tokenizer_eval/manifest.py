# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run manifests: a TOML file that pins everything a published result depends on.

    [run]
    name = "my-report"
    suite = "full"                    # and/or: corpora = ["in22/hin_Deva", ...]
    reference = "My-Tokenizer"
    probes = ["My-Tokenizer", "Llama-3.2"]   # tokenizers to probe; "all" for every one

    [[tokenizer]]
    name = "My-Tokenizer"
    hub = "org/my-tokenizer"
    revision = "0123456789abcdef0123456789abcdef01234567"   # full 40-character commit
    sha256 = "726e7636..."                                   # of the tokenizer.json that is loaded

    [[tokenizer]]
    name = "BPE-baseline"
    path = "tokenizers/bpe/tokenizer.json"   # relative to the manifest
    sha256 = "c8be77bd..."

    [[tokenizer]]
    name = "o200k"
    tiktoken = "o200k_base"
    fingerprint = "..."               # vocabulary fingerprint (Tokenizer.fingerprint())

    [groups]                          # optional named pools of corpora for the group table
    "Indic-12" = ["in22/ben_Beng", ...]

Every tokenizer must be pinned: a Hub tokenizer by its full commit and the
SHA-256 of its tokenizer file, a local file by its SHA-256, a tiktoken encoding
by its vocabulary fingerprint. `verify()` refuses a tokenizer whose loaded file
or vocabulary differs from the manifest, so a manifest either reproduces the
published counts or fails before counting anything.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .provenance import sha256_file
from .registry import CORPORA, SUITES

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    pass


@dataclass
class TokenizerPin:
    name: str
    spec: str
    sha256: str | None = None
    fingerprint: str | None = None


@dataclass
class Manifest:
    path: Path
    sha256: str
    name: str
    corpora: list[str]
    reference: str
    probes: list[str]
    tokenizers: list[TokenizerPin]
    groups: dict[str, list[str]] = field(default_factory=dict)

    def specs(self) -> list[str]:
        return [f"{t.name}={t.spec}" for t in self.tokenizers]


def _pin(d: dict, base: Path) -> TokenizerPin:
    name = d.get("name")
    if not name or "=" in name or "/" in name:
        raise ManifestError(f"tokenizer needs a plain `name` without '=' or '/': {d}")
    kinds = [k for k in ("hub", "path", "tiktoken") if k in d]
    if len(kinds) != 1:
        raise ManifestError(f"{name}: give exactly one of hub, path or tiktoken")
    sha, fp = d.get("sha256"), d.get("fingerprint")
    if sha is not None and not _SHA256.match(sha):
        raise ManifestError(f"{name}: sha256 must be 64 lowercase hex characters")
    if fp is not None and not _SHA256.match(fp):
        raise ManifestError(f"{name}: fingerprint must be 64 lowercase hex characters")
    if kinds[0] == "hub":
        rev = d.get("revision", "")
        if not _COMMIT.match(rev):
            raise ManifestError(f"{name}: hub tokenizers need `revision`, a full 40-character commit")
        if sha is None and fp is None:
            raise ManifestError(f"{name}: hub tokenizers need `sha256` (or `fingerprint` if no tokenizer.json)")
        return TokenizerPin(name, f"hf:{d['hub']}@{rev}", sha, fp)
    if kinds[0] == "path":
        p = (base / d["path"]).resolve()
        if sha is None:
            raise ManifestError(f"{name}: local tokenizers need `sha256`")
        return TokenizerPin(name, str(p), sha, fp)
    if fp is None:
        raise ManifestError(f"{name}: tiktoken encodings need `fingerprint`")
    return TokenizerPin(name, f"tiktoken:{d['tiktoken']}", None, fp)


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path).resolve()
    with open(path, "rb") as f:
        d = tomllib.load(f)
    run = d.get("run", {})
    toks = [_pin(t, path.parent) for t in d.get("tokenizer", [])]
    if not toks:
        raise ManifestError("manifest lists no [[tokenizer]]")
    names = [t.name for t in toks]
    if len(set(names)) != len(names):
        raise ManifestError(f"duplicate tokenizer names: {sorted({n for n in names if names.count(n) > 1})}")
    ids = list(run.get("corpora", []))
    if run.get("suite"):
        if run["suite"] not in SUITES:
            raise ManifestError(f"unknown suite {run['suite']!r}")
        ids += SUITES[run["suite"]]
    ids = list(dict.fromkeys(ids))
    groups = {g: list(c) for g, c in d.get("groups", {}).items()}
    unknown = [c for c in ids + [c for v in groups.values() for c in v] if c not in CORPORA]
    if unknown:
        raise ManifestError(f"unknown corpus ids: {unknown[:5]}")
    missing = [c for v in groups.values() for c in v if c not in ids]
    if missing:
        raise ManifestError(f"groups use corpora that are not run: {missing[:5]}")
    if not ids:
        raise ManifestError("[run] needs `suite` and/or `corpora`")
    ref = run.get("reference", names[0])
    if ref not in names:
        raise ManifestError(f"reference {ref!r} is not a listed tokenizer")
    probes = run.get("probes", [])
    probes = names if probes == "all" else list(probes)
    if [p for p in probes if p not in names]:
        raise ManifestError(f"probes name unknown tokenizers: {[p for p in probes if p not in names]}")
    return Manifest(path, sha256_file(path), run.get("name", path.stem), ids, ref, probes, toks, groups)


def verify(tok, pin: TokenizerPin) -> None:
    """Raise ManifestError if a loaded tokenizer differs from its pin."""
    got = tok.source.get("tokenizer_json_sha256")
    if pin.sha256 is not None and got != pin.sha256:
        raise ManifestError(f"{pin.name}: tokenizer file SHA-256 is {got}, manifest pins {pin.sha256}")
    if pin.fingerprint is not None and tok.fingerprint() != pin.fingerprint:
        raise ManifestError(f"{pin.name}: vocabulary fingerprint is {tok.fingerprint()}, "
                            f"manifest pins {pin.fingerprint}")
