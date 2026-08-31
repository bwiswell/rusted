"""Tests for ``src/kinds.rs`` — per-kind coercion parity.

A cross-product sweep: every scalar kind against every interesting value, in
both strict and lax mode, load and dump. Any divergence from seared's own
field methods — value, type, exception class, or message text — fails here.
"""

from __future__ import annotations

import pytest
from conftest import VALUE_IDS, VALUES, load_outcome, one_field, outcome, raw

KINDS = ['int', 'float', 'str', 'bool']
MODES = [(True, 'strict'), (False, 'lax')]


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
@pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
class TestScalarParity:
    def test_load(self, kind, validate, mode, value):
        fast = one_field(kind, validate=validate, accel=True)
        slow = one_field(kind, validate=validate, accel=False)
        assert load_outcome(fast, {'v': value}) == load_outcome(slow, {'v': value})

    def test_dump(self, kind, validate, mode, value):
        fast = one_field(kind, validate=validate, accel=True)
        slow = one_field(kind, validate=validate, accel=False)
        assert outcome(fast.dump, raw(fast, v=value)) == outcome(slow.dump, raw(slow, v=value))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestContainerParity:
    def test_many(self, kind, validate, mode):
        fast = one_field(kind, validate=validate, accel=True, many=True)
        slow = one_field(kind, validate=validate, accel=False, many=True)
        for payload in ([], [1], ['x'], [None], (1, 2), 'not-a-list', 7, None, {'a': 1}):
            assert load_outcome(fast, {'v': payload}) == load_outcome(slow, {'v': payload})

    def test_keyed(self, kind, validate, mode):
        fast = one_field(kind, validate=validate, accel=True, keyed=True)
        slow = one_field(kind, validate=validate, accel=False, keyed=True)
        for payload in ({}, {'a': 1}, {'a': 'x'}, {'a': None}, [1], 'nope', 7, None):
            assert load_outcome(fast, {'v': payload}) == load_outcome(slow, {'v': payload})


class TestBoolStringSpellings:
    # seared strips and lowercases with Python's own str methods; rusted calls
    # the same methods rather than Rust's trim/to_lowercase, whose Unicode
    # definitions differ at the edges.
    @pytest.mark.parametrize(
        'value',
        [
            'true',
            'TRUE',
            ' True ',
            '1',
            'yes',
            'ON',
            'false',
            'FALSE',
            ' off ',
            '0',
            'no',
            ' true ',
            'maybe',
            '2',
            # Deliberate: U+00A0 and U+3000 are whitespace to Python's str.strip()
            # and to Rust's char::is_whitespace, but the two definitions do not
            # agree everywhere. This is the case that would catch a core that
            # reached for Rust's trim() instead of calling Python's strip().
            '\u00a0true\u00a0',
            '\u3000yes\u3000',
            '\x1ctrue\x1c',
        ],
    )
    def test_spellings_match(self, value):
        fast = one_field('bool', accel=True)
        slow = one_field('bool', accel=False)
        assert load_outcome(fast, {'v': value}) == load_outcome(slow, {'v': value})
