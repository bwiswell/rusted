"""Tests for ``src/kinds/mod.rs`` — the container orchestration around a kind.

``apply`` is the transcription of seared's ``_core.decorator._apply``:
``keyed`` → dict-of-V, ``many`` → list-of-V, scalar otherwise, with the
strict-mode shape guards up front and lax mode left to whatever Python
raises. Per-kind coercion is the other three files' job; this one is about
the shapes wrapped around it.
"""

from __future__ import annotations

import pytest
from conftest import TIER1_MODES, load_outcome, one_field, outcome, same


@pytest.mark.parametrize(('kind', 'validate'), TIER1_MODES)
class TestContainerParity:
    def test_many(self, kind, validate):
        fast = one_field(kind, validate=validate, accel=True, many=True)
        slow = one_field(kind, validate=validate, accel=False, many=True)
        for payload in ([], [1], ['x'], [None], (1, 2), 'not-a-list', 7, None, {'a': 1}):
            assert same(load_outcome(fast, {'v': payload}), load_outcome(slow, {'v': payload}))

    def test_keyed(self, kind, validate):
        fast = one_field(kind, validate=validate, accel=True, keyed=True)
        slow = one_field(kind, validate=validate, accel=False, keyed=True)
        for payload in ({}, {'a': 1}, {'a': 'x'}, {'a': None}, [1], 'nope', 7, None):
            assert same(load_outcome(fast, {'v': payload}), load_outcome(slow, {'v': payload}))


class TestErrorPaths:
    """Pin ``apply``'s part of the error contract: a failing item adds its index or key."""

    @pytest.mark.parametrize(
        ('shape', 'payload', 'path'),
        [
            ('many', [1, 2, 'x'], ('v', 2)),
            ('keyed', {'a': 1, 'b': 'x'}, ('v', 'b')),
            ('keyed', {1: 'x'}, ('v', 1)),
            ('keyed', {(1, 2): 'x'}, ('v', (1, 2))),
        ],
    )
    def test_the_failing_item_is_named(self, shape, payload, path):
        fast = one_field('int', accel=True, **{shape: True})
        slow = one_field('int', accel=False, **{shape: True})
        got, want = outcome(fast.load, {'v': payload}), outcome(slow.load, {'v': payload})
        assert got == want
        assert got[4] == path

    def test_a_container_error_adds_no_index(self):
        fast = one_field('int', accel=True, many=True)
        got = outcome(fast.load, {'v': 'nope'})
        assert got[3:] == ('expected list for many field, got str', ('v',), 'C')

    def test_any_other_exception_passes_through_untouched(self):
        # Lax mode runs ``.items()`` on a non-dict, as seared's comprehension does.
        fast = one_field('int', validate=False, accel=True, keyed=True)
        slow = one_field('int', validate=False, accel=False, keyed=True)
        got, want = outcome(fast.load, {'v': [1]}), outcome(slow.load, {'v': [1]})
        assert got == want
        assert got[1] == 'AttributeError'
