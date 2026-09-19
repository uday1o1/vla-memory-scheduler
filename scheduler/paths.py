"""Filesystem locations, resolved rather than hardcoded.

Experiments run both on a development machine and on rented GPU instances
where the checkout lands in different places, so nothing here assumes /root.
Set ALPAMAYO_HOME to point at the upstream oom-free-alpamayo clone if it is
not a sibling of this repository.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_upstream() -> Path:
    env = os.environ.get("ALPAMAYO_HOME")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (REPO_ROOT.parent / "oom-free-alpamayo", Path("/root/oom-free-alpamayo")):
        if candidate.is_dir():
            return candidate
    return REPO_ROOT.parent / "oom-free-alpamayo"


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
