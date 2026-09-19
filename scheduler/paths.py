"""Filesystem locations, resolved rather than hardcoded.

Experiments run both on a development machine and on rented GPU instances
where the checkout lands in different places, so no absolute path is assumed.
Set ALPAMAYO_HOME to point at the upstream oom-free-alpamayo clone if it is
not a sibling of this repository.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_upstream() -> Path:
    """Locate the oom-free-alpamayo clone.

    Looked for in order: ALPAMAYO_HOME, then under WORKDIR if the setup script
    placed it there, then beside this repository, then the home directory. The
    default in the last resort is the sibling, which is where setup puts it.
    """
    env = os.environ.get("ALPAMAYO_HOME")
    if env:
        return Path(env).expanduser().resolve()

    sibling = REPO_ROOT.parent / "oom-free-alpamayo"
    candidates = [sibling, Path.home() / "oom-free-alpamayo"]
    workdir = os.environ.get("WORKDIR")
    if workdir:
        candidates.insert(0, Path(workdir).expanduser() / "oom-free-alpamayo")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return sibling


UPSTREAM = _find_upstream()
R1_CONFIG = UPSTREAM / "r1_config.json"

# Where experiment runs write their JSON. Override with VLA_RESULTS to keep a
# run's output outside the repository.
RESULTS = Path(os.environ.get("VLA_RESULTS") or (REPO_ROOT / "data"))


def bootstrap() -> None:
    """Put the repository root and the upstream clone on the import path."""
    for p in (str(UPSTREAM), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
