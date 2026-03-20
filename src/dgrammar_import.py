"""
Ensure the dgrammar repo root is on sys.path so `import dgrammar` works.

SDSD diffusion uses ``dgrammar.checker.TokenChecker`` and helpers from
``dgrammar.generate``. That package is not on PyPI; install it by:

- Clone into ``anlp_final/vendor/dgrammar``, or
- ``pip install -e /path/to/dgrammar`` (editable), or
- Set ``DGRAMMAR_PATH`` to the repo root (directory that contains ``dgrammar/``).

Also requires ``pip install 'llguidance>=1.6'``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_roots() -> list[Path]:
    """This file lives in anlp_final/src/ — project root is parent.parent."""
    here = Path(__file__).resolve()
    anlp_root = here.parent.parent
    return [anlp_root, here.parent]


def ensure_dgrammar_path() -> bool:
    """
    Insert dgrammar repo root on sys.path if ``import dgrammar`` fails.

    Returns True if ``import dgrammar`` succeeds after setup (or already worked).
    """
    try:
        import dgrammar  # noqa: F401

        return True
    except ImportError:
        pass

    env = os.environ.get("DGRAMMAR_PATH", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser().resolve())

    for root in _repo_roots():
        candidates.append(root / "vendor" / "dgrammar")
        candidates.append(root.parent / "dgrammar")

    seen: set[str] = set()
    for p in candidates:
        p = p.resolve()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        # Layout: <repo>/dgrammar/__init__.py (package) or dgrammar/checker.py
        pkg = p / "dgrammar"
        if p.is_dir() and pkg.is_dir():
            if key not in sys.path:
                sys.path.insert(0, key)
            try:
                import dgrammar  # noqa: F401

                return True
            except ImportError:
                continue

    return False


def dgrammar_available() -> bool:
    """True if TokenChecker can be imported (dgrammar + llguidance)."""
    if not ensure_dgrammar_path():
        return False
    try:
        from dgrammar.checker import TokenChecker  # noqa: F401

        return True
    except ImportError:
        return False
