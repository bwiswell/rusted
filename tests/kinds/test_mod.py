"""Tests for ``src/kinds/mod.rs`` — the container orchestration around a kind.

``apply`` is the transcription of seared's ``_core.decorator._apply``:
``keyed`` → dict-of-V, ``many`` → list-of-V, scalar otherwise, with the
strict-mode shape guards up front and lax mode left to whatever Python
raises. Per-kind coercion is the other three files' job; this one is about
the shapes wrapped around it.
"""

from __future__ import annotations

import pytest
from conftest import MODES, TIER1, load_outcome, one_field, same


@pytest.mark.parametrize('kind', TIER1)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestContainerParity:
    def test_many(self, kind, validate, mode):
        fast = one_field(kind, validate=validate, accel=True, many=True)
        slow = one_field(kind, validate=validate, accel=False, many=True)
        for payload in ([], [1], ['x'], [None], (1, 2), 'not-a-list', 7, None, {'a': 1}):
            assert same(load_outcome(fast, {'v': payload}), load_outcome(slow, {'v': payload}))

    def test_keyed(self, kind, validate, mode):
        fast = one_field(kind, validate=validate, accel=True, keyed=True)
        slow = one_field(kind, validate=validate, accel=False, keyed=True)
        for payload in ({}, {'a': 1}, {'a': 'x'}, {'a': None}, [1], 'nope', 7, None):
            assert same(load_outcome(fast, {'v': payload}), load_outcome(slow, {'v': payload}))
