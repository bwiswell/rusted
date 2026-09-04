"""Tests for ``src/kinds/value.rs`` — ``uuid`` / ``decimal`` / ``bytes`` / ``enum`` / ``path`` / ``dict``.

The generic sweep runs the same wrong-type values Tier 1 sees: which
branches seared gates on ``validate`` and which raise regardless is not
uniform across these fields, and each has to be matched individually. The
native sweep runs the wire forms each kind is actually meant to accept, and
the option tests cover the per-field configuration the spec carries.
"""

from __future__ import annotations

import pathlib

import pytest
from conftest import (
    KIND_VALUES,
    MODES,
    VALUE,
    VALUE_IDS,
    VALUES,
    Color,
    Status,
    both,
    load_outcome,
    outcome,
    raw,
    same,
)


@pytest.mark.parametrize('kind', VALUE)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestGenericSweep:
    @pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
    def test_load(self, kind, validate, mode, value):
        fast, slow = both(kind, validate=validate)
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))

    @pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
    def test_dump(self, kind, validate, mode, value):
        fast, slow = both(kind, validate=validate)
        assert same(outcome(fast.dump, raw(fast, v=value)), outcome(slow.dump, raw(slow, v=value)))


@pytest.mark.parametrize('kind', VALUE)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestNativeValues:
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


class TestBytes:
    @pytest.mark.parametrize('encoding', ['base64', 'hex'])
    def test_encoding(self, encoding):
        fast, slow = both('bytes', encoding=encoding)
        for value in ['ZGVhZGJlZWY=', 'deadbeef', 'zzz', '']:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value
        assert same(outcome(fast.dump, raw(fast, v=b'\xde\xad')), outcome(slow.dump, raw(slow, v=b'\xde\xad')))

    def test_native_under_msgpack(self):
        # The one kind that reads the carrier hint: raw bytes on msgpack,
        # base64 on JSON.
        fast, slow = both('bytes')
        obj_f, obj_s = raw(fast, v=b'\xde\xad'), raw(slow, v=b'\xde\xad')
        assert fast.dump(obj_f, 'msgpack') == slow.dump(obj_s, 'msgpack') == {'v': b'\xde\xad'}
        assert fast.dump(obj_f) == slow.dump(obj_s) == {'v': '3q0='}


class TestDecimal:
    @pytest.mark.parametrize('as_number', [True, False])
    def test_as_number(self, as_number):
        fast, slow = both('decimal', as_number=as_number)
        for value in ['3.14159', '1e400', 0.5, 'nan']:
            a, b = outcome(fast.load, {'v': value}), outcome(slow.load, {'v': value})
            assert a[0] == b[0]
            if a[0] == 'ok':
                assert same(outcome(fast.dump, a[1]), outcome(slow.dump, b[1])), value


class TestEnum:
    @pytest.mark.parametrize('enum_cls', [Color, Status])
    def test_int_and_str_valued(self, enum_cls):
        # seared picks `enum(int(value))` vs `enum(value)` off the first
        # member's type; rusted decides it once at compile time instead.
        fast, slow = both('enum', enum=enum_cls)
        for value in [0, 1, '0', 'active', 'nope', None, 2.0]:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value


class TestPath:
    def test_concrete_class(self):
        fast, slow = both('path', concrete=pathlib.PurePosixPath)
        loaded = fast.load({'v': 'a/b'})
        assert isinstance(loaded.v, pathlib.PurePosixPath)
        assert same(load_outcome(fast, {'v': 'a/b'}), load_outcome(slow, {'v': 'a/b'}))

    def test_always_dumps_posix(self):
        fast, slow = both('path')
        obj_f = raw(fast, v=pathlib.PureWindowsPath('C:\\foo\\bar'))
        obj_s = raw(slow, v=pathlib.PureWindowsPath('C:\\foo\\bar'))
        assert fast.dump(obj_f) == slow.dump(obj_s) == {'v': 'C:/foo/bar'}
