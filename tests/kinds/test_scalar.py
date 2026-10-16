"""Tests for ``src/kinds/scalar.rs`` — Tier 1 ``int`` / ``float`` / ``str`` / ``bool``.

A cross-product sweep: every scalar kind against every interesting value, in
both strict and lax mode, load and dump. Any divergence from seared's own
field methods — value, type, exception class, or message text — fails here.
"""

from __future__ import annotations

import math

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
                'nullable': False,
                'many': False,
                'keyed': False,
                'dump': True,
                'default': None,
                'default_factory': None,
            },
        ],
    }


class TestLaxBoolReadsTheSpellingTable:
    """Lax ``bool`` reads the spelling table before truthiness, as seared's does.

    Under ``SPEC_ABI`` 1 this core went straight to truthiness, so ``'false'``
    loaded as ``True``. Checked through a hand-built spec, which no seared
    decline can route around.
    """

    @pytest.mark.parametrize(
        ('value', 'want'),
        [
            ('false', False),
            ('0', False),
            ('no', False),
            ('off', False),
            (' OFF ', False),
            ('true', True),
            ('maybe', True),
            ('', False),
            (2, True),
        ],
    )
    def test_matches_seared(self, value, want):
        load, _dump = rusted.compile_spec(_lax_bool_spec())
        assert load({'v': value}).v is want
        assert s.Bool().deserialize(value, validate=False) is want


@pytest.mark.parametrize('validate', [True, False], ids=['strict', 'lax'])
class TestNumericCoercion:
    """Pin ``SPEC_ABI`` 2's coercion: strict never loses data, and ``inf`` is never an ``OverflowError``."""

    @pytest.mark.parametrize('value', [3.0, 3.7, -2.0, math.inf, -math.inf, math.nan, '42', ' 7 ', '3.0', True, 10**20])
    def test_int(self, validate, value):
        fast, slow = one_field('int', validate=validate, accel=True), one_field('int', validate=validate, accel=False)
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))

    @pytest.mark.parametrize('value', [0, 1, 2, -1, 1.0, True, '1', '2'])
    def test_bool(self, validate, value):
        fast, slow = one_field('bool', validate=validate, accel=True), one_field('bool', validate=validate, accel=False)
        assert load_outcome(fast, {'v': value}) == load_outcome(slow, {'v': value})

    def test_strict_int_refuses_a_fraction_and_lax_truncates(self, validate):
        fast = one_field('int', validate=validate, accel=True)
        got = load_outcome(fast, {'v': 3.7})
        assert got == (
            (
                'raised',
                'ValidationError',
                'C.v: cannot deserialize 3.7 as int',
                'cannot deserialize 3.7 as int',
                ('v',),
                'C',
            )
            if validate
            else ('ok', 3, 'int')
        )

    def test_inf_is_a_validation_error_in_both_modes(self, validate):
        fast = one_field('int', validate=validate, accel=True)
        with pytest.raises(s.ValidationError, match=r'cannot deserialize inf as int$'):
            fast.load({'v': math.inf})

    def test_a_float_subclass_is_asked_whether_it_is_integral(self, validate):
        class Whole(float):
            def is_integer(self):
                return True

        fast, slow = one_field('int', validate=validate, accel=True), one_field('int', validate=validate, accel=False)
        assert load_outcome(fast, {'v': Whole(3.7)}) == load_outcome(slow, {'v': Whole(3.7)}) == ('ok', 3, 'int')
