"""rusted — the compiled accelerator core for seared.

Implements the backend protocol ``seared._core.accel`` looks for:
:data:`SPEC_ABI` and :func:`compile_spec`. seared imports this module lazily
and only ever treats it as optional — nothing here is required for seared to
work, and a class this build can't take quietly keeps the Python path.

Install it and ``@s.seared`` classes get compiled ``load`` / ``dump`` with no
code change; uninstall it and they don't. That is the entire user-facing API.

Check what happened with ``seared.accel_status()`` (global) or a class's
``__seared_accel__`` (per class, with the reason it declined).
"""

from __future__ import annotations

from ._rusted import BUILD_PROFILE, SPEC_ABI, compile_spec

#: Diagnostic only. seared gates on :data:`SPEC_ABI` and never parses this —
#: it is zero-dependency and has no PEP 440 specifier parser to do it with.
#:
#: The floor is the seared release that introduced the seam and the spec keys
#: this build reads; the ceiling is a guess about where the spec might next
#: change, not a tested boundary. Because nothing enforces it, it is the one
#: thing here that can silently go stale — check it whenever seared's minor
#: version moves.
SUPPORTS_SEARED = '>=0.2.8,<0.4'

__version__ = '0.1.2'

__all__ = [
    'BUILD_PROFILE',
    'SPEC_ABI',
    'SUPPORTS_SEARED',
    '__version__',
    'compile_spec',
]
