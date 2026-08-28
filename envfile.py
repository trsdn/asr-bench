"""
Loads the repo's `.env` before anything imports torch / NeMo / HF.

Model caches have to be redirected *before* those libraries are
imported, because they read the environment at import time and cache the
result. Every entry point therefore calls `load_env()` at the top,
ahead of its heavy imports.

`.env` is per-machine and gitignored; `.env.example` is the committed
template and is used as a fallback so a fresh clone still runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent


def load_env(verbose: bool = True) -> Path | None:
    """Populate os.environ from `.env` (or `.env.example`).

    Existing environment variables win — `setdefault`, not overwrite —
    so a caller can pin a cache location for a single run without
    editing the file. Returns the file that was read, or None."""
    env_file = REPO_DIR / ".env"
    if not env_file.exists():
        example = REPO_DIR / ".env.example"
        if not example.exists():
            return None
        if verbose:
            print(
                f"[asr-bench] no .env found — falling back to {example.name}. "
                f"Copy it to .env and edit paths for your machine.",
                file=sys.stderr,
            )
        env_file = example

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
    return env_file
