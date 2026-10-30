# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""What produced a result: suite commit, package versions, platform and command line.

Token counts depend on the tokenizer file and on the `tokenizers` / `tiktoken`
library that runs it, so both are recorded with every run. A run made from a
working tree with uncommitted changes is marked `dirty`; such results should
not be published.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import __version__

PKG_ROOT = Path(__file__).resolve().parent.parent

# Packages whose version can change a token count or a corpus.
PACKAGES = ["tokenizers", "tiktoken", "transformers", "sentencepiece", "huggingface_hub",
            "numpy", "pandas", "pyarrow", "regex"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(PKG_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def suite_commit() -> dict:
    """Commit of the suite's source tree, and whether tracked files differ from it."""
    top = _git("rev-parse", "--show-toplevel")
    # an installed copy can sit inside some other repository (e.g. a project's .venv): not ours
    if top is None or Path(top).resolve() != PKG_ROOT:
        return {"commit": None, "dirty": None}
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no", "--", "brahmai_tokenizer_eval")
    return {"commit": commit, "dirty": bool(status)}


def package_versions() -> dict[str, str | None]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(argv: list[str] | None = None) -> dict:
    return {"suite_version": __version__, **{f"suite_{k}": v for k, v in suite_commit().items()},
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": package_versions(), "argv": list(sys.argv if argv is None else argv)}
