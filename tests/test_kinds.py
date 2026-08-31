"""Tests for ``src/kinds.rs`` — per-kind coercion parity.

A cross-product sweep: every scalar kind against every interesting value, in
both strict and lax mode, load and dump. Any divergence from seared's own
field methods — value, type, exception class, or message text — fails here.
"""

from __future__ import annotations

import pathlib

import pytest
from conftest import (
    KIND_VALUES,
    TIER2,
    VALUE_IDS,
    VALUES,
    Color,
    Status,
    both,
    load_outcome,
    one_field,
    outcome,
    raw,
    same,
)

KINDS = ['int', 'float', 'str', 'bool']
MODES = [(True, 'strict'), (False, 'lax')]


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
@pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
class TestScalarParity:
    def test_load(self, kind, validate, mode, value):
        fast = one_field(kind, validate=validate, accel=True)
        slow = one_field(kind, validate=validate, accel=False)
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))

    def test_dump(self, kind, validate, mode, value):
        fast = one_field(kind, validate=validate, accel=True)
        slow = one_field(kind, validate=validate, accel=False)
        assert same(outcome(fast.dump, raw(fast, v=value)), outcome(slow.dump, raw(slow, v=value)))


@pytest.mark.parametrize('kind', KINDS)
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


# ---------------------------------------------------------------------------
# Tier 2 — parse-and-construct kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', TIER2)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestTier2GenericSweep:
    """Every Tier 2 kind against the same wrong-type values Tier 1 sees.

    This is where the type *guards* get exercised: which branches seared
    gates on `validate` and which raise regardless is not uniform across
    these fields, and each one has to be matched individually.
    """

    @pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
    def test_load(self, kind, validate, mode, value):
        fast, slow = both(kind, validate=validate)
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))

    @pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
    def test_dump(self, kind, validate, mode, value):
        fast, slow = both(kind, validate=validate)
        assert same(outcome(fast.dump, raw(fast, v=value)), outcome(slow.dump, raw(slow, v=value)))


@pytest.mark.parametrize('kind', TIER2)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestTier2NativeValues:
    """Each kind against the wire forms it is actually meant to accept."""

    def test_load(self, kind, validate, mode):
        fast, slow = both(kind, validate=validate)
        for value in KIND_VALUES[kind]:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value

    def test_round_trip(self, kind, validate, mode):
        fast, slow = both(kind, validate=validate)
        for value in KIND_VALUES[kind]:
            a, b = outcome(fast.load, {'v': value}), outcome(slow.load, {'v': value})
            if a[0] == 'raised':
                assert a == b, value
                continue
            assert same(outcome(fast.dump, a[1]), outcome(slow.dump, b[1])), value

    def test_many(self, kind, validate, mode):
        fast, slow = both(kind, validate=validate, many=True)
        for value in ([], list(KIND_VALUES[kind]), 'nope', None, 7):
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value


class TestKindOptions:
    @pytest.mark.parametrize('encoding', ['base64', 'hex'])
    def test_bytes_encoding(self, encoding):
        fast, slow = both('bytes', encoding=encoding)
        for value in ['ZGVhZGJlZWY=', 'deadbeef', 'zzz', '']:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value
        assert same(outcome(fast.dump, raw(fast, v=b'\xde\xad')), outcome(slow.dump, raw(slow, v=b'\xde\xad')))

    def test_bytes_native_under_msgpack(self):
        # The one kind that reads the carrier hint: raw bytes on msgpack,
        # base64 on JSON.
        fast, slow = both('bytes')
        obj_f, obj_s = raw(fast, v=b'\xde\xad'), raw(slow, v=b'\xde\xad')
        assert fast.dump(obj_f, 'msgpack') == slow.dump(obj_s, 'msgpack') == {'v': b'\xde\xad'}
        assert fast.dump(obj_f) == slow.dump(obj_s) == {'v': '3q0='}

    @pytest.mark.parametrize(
        ('kind', 'fmt', 'values'),
        [
            ('date', '%d/%m/%Y', ['31/08/2026', '2026-08-31', 'nope']),
            ('datetime', '%Y/%m/%d %H:%M', ['2026/08/31 12:30', 'nope']),
            ('time', '%H:%M', ['12:30', '12:30:45', 'nope']),
        ],
    )
    def test_strftime_format(self, kind, fmt, values):
        fast, slow = both(kind, format=fmt)
        for value in values:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value
        a, b = fast.load({'v': values[0]}), slow.load({'v': values[0]})
        assert fast.dump(a) == slow.dump(b)

    @pytest.mark.parametrize('as_number', [True, False])
    def test_decimal_as_number(self, as_number):
        fast, slow = both('decimal', as_number=as_number)
        for value in ['3.14159', '1e400', 0.5, 'nan']:
            a, b = outcome(fast.load, {'v': value}), outcome(slow.load, {'v': value})
            assert a[0] == b[0]
            if a[0] == 'ok':
                assert same(outcome(fast.dump, a[1]), outcome(slow.dump, b[1])), value

    @pytest.mark.parametrize('enum_cls', [Color, Status])
    def test_enum_int_and_str_valued(self, enum_cls):
        # seared picks `enum(int(value))` vs `enum(value)` off the first
        # member's type; rusted decides it once at compile time instead.
        fast, slow = both('enum', enum=enum_cls)
        for value in [0, 1, '0', 'active', 'nope', None, 2.0]:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value

    def test_path_concrete_class(self):
        fast, slow = both('path', concrete=pathlib.PurePosixPath)
        loaded = fast.load({'v': 'a/b'})
        assert isinstance(loaded.v, pathlib.PurePosixPath)
        assert same(load_outcome(fast, {'v': 'a/b'}), load_outcome(slow, {'v': 'a/b'}))

    def test_path_always_dumps_posix(self):
        fast, slow = both('path')
        obj_f = raw(fast, v=pathlib.PureWindowsPath('C:\\foo\\bar'))
        obj_s = raw(slow, v=pathlib.PureWindowsPath('C:\\foo\\bar'))
        assert fast.dump(obj_f) == slow.dump(obj_s) == {'v': 'C:/foo/bar'}
