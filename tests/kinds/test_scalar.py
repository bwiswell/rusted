"""Tests for ``src/kinds/scalar.rs`` — Tier 1 ``int`` / ``float`` / ``str`` / ``bool``.

A cross-product sweep: every scalar kind against every interesting value, in
both strict and lax mode, load and dump. Any divergence from seared's own
field methods — value, type, exception class, or message text — fails here.
Lax ``bool`` sits out until ``SPEC_ABI`` 2 (see ``conftest.LAX_TIER1``).
"""

from __future__ import annotations

import pytest
import rusted
import seared as s
from conftest import TIER1_MODES, VALUE_IDS, VALUES, load_outcome, one_field, outcome, raw, same


@pytest.mark.parametrize(('kind', 'validate'), TIER1_MODES)
@pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
class TestScalarParity:
    def test_load(self, kind, validate, value):
        fast = one_field(kind, validate=validate, accel=True)
        slow = one_field(kind, validate=validate, accel=False)
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))

    def test_dump(self, kind, validate, value):
        fast = one_field(kind, validate=validate, accel=True)
        slow = one_field(kind, validate=validate, accel=False)
        assert same(outcome(fast.dump, raw(fast, v=value)), outcome(slow.dump, raw(slow, v=value)))


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
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))


def _lax_bool_spec():
    """A hand-built lax ``bool`` spec: the only way past seared 0.3.3's decline."""

    class Plain:
        __slots__ = ('v',)

    return {
        'abi': rusted.SPEC_ABI,
        'cls': Plain,
        'name': 'Plain',
        'validate': False,
        'error': s.ValidationError,
        'fields': [
            {
                'attr': 'v',
                'wire': 'v',
                'kind': 'bool',
                'required': True,
                'many': False,
                'keyed': False,
                'dump': True,
                'default': None,
                'default_factory': None,
            },
        ],
    }


class TestLaxBoolIsTheAbi1Answer:
    """Lax ``bool`` here still gives the ABI 1 answer, bare truthiness, on purpose.

    seared 0.3.3 reads the spelling table first in lax mode too, and declines
    lax classes with a ``Bool`` because this core doesn't. Changing ``de_bool``
    under ABI 1 would make the two paths disagree silently under seared 0.3.1
    and 0.3.2, which still give this answer. It changes with ``SPEC_ABI`` 2,
    and so does this test (``test_differential.py::TestLaxBoolAwaitsAbi2``).
    """

    @pytest.mark.parametrize('value', ['false', '0', 'no', 'off'])
    def test_false_spellings_are_truthy(self, value):
        load, _dump = rusted.compile_spec(_lax_bool_spec())
        assert load({'v': value}).v is True
        assert s.Bool().deserialize(value, validate=False) is False
