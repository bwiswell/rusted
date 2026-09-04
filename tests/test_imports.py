"""Tests for ``src/imports.rs`` — the cached stdlib handles.

The Tier 2 kinds build ``uuid.UUID``, ``datetime.*``, ``Decimal`` and path
objects through classes resolved once per interpreter. What is observable
from Python is that the objects that come out are instances of the *real*
stdlib classes — the same ones a caller's ``isinstance`` checks name — and
that the cache stays stable across calls.
"""

from __future__ import annotations

import datetime
import decimal
import pathlib
import uuid

import pytest
from conftest import KIND_VALUES, one_field

EXPECTED = {
    'uuid': uuid.UUID,
    'date': datetime.date,
    'datetime': datetime.datetime,
    'time': datetime.time,
    'timedelta': datetime.timedelta,
    'decimal': decimal.Decimal,
    # `Path(...)` constructs the host's concrete flavour (PosixPath here),
    # which is what `concrete=` defaults to on the seared side.
    'path': type(pathlib.Path()),
}


@pytest.mark.parametrize('kind', list(EXPECTED))
class TestStdlibIdentity:
    def test_builds_the_real_stdlib_class(self, kind):
        cls = one_field(kind, accel=True)
        assert cls.__seared_accel__.accelerated is True
        value = cls.load({'v': KIND_VALUES[kind][0]}).v
        assert type(value) is EXPECTED[kind], type(value)

    def test_cache_is_stable_across_calls(self, kind):
        cls = one_field(kind, accel=True)
        first = cls.load({'v': KIND_VALUES[kind][0]}).v
        second = cls.load({'v': KIND_VALUES[kind][0]}).v
        assert type(first) is type(second)


class TestBase64Module:
    def test_encodes_through_stdlib_base64(self):
        # `ser_bytes` goes through `base64.b64encode`; the wire form must be
        # byte-for-byte what the stdlib produces, ascii-decoded.
        import base64

        cls = one_field('bytes', accel=True)
        obj = cls.load({'v': b'\xde\xad\xbe\xef'})
        assert cls.dump(obj) == {'v': base64.b64encode(b'\xde\xad\xbe\xef').decode('ascii')}
