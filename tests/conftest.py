"""Shared helpers.

Every test compares the compiled core against seared's own Python path, so
the pattern throughout is: build the same schema twice — once normally
(rusted takes it) and once with ``accel=False`` (it can't) — and assert the
two are indistinguishable, values *and* exceptions.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import pathlib
import uuid

import pytest
import seared as s

_status = s.accel_status()
if not _status['available'] or _status['backend'] != 'rusted':
    msg = (
        f'rusted is not the active seared backend ({_status}). These tests '
        f'compare compiled behaviour against the Python path; running them '
        f'unaccelerated would silently compare Python to itself.'
    )
    raise RuntimeError(msg)


class Color(enum.Enum):
    """Int-valued: seared looks members up via ``enum(int(value))``."""

    RED = 0
    BLUE = 1


class Status(enum.StrEnum):
    """Str-valued: looked up via ``enum(value)``."""

    ACTIVE = 'active'
    IDLE = 'idle'


KIND_FIELDS = {
    # Tier 1
    'int': s.Int,
    'float': s.Float,
    'str': s.Str,
    'bool': s.Bool,
    # Tier 2
    'uuid': s.UUID,
    'date': s.Date,
    'datetime': s.DateTime,
    'time': s.Time,
    'timedelta': s.TimeDelta,
    'decimal': s.Decimal,
    'bytes': s.Bytes,
    'enum': s.Enum,
    'path': s.Path,
    'dict': s.Dict,
}

TIER1 = ['int', 'float', 'str', 'bool']
TIER2 = ['uuid', 'date', 'datetime', 'time', 'timedelta', 'decimal', 'bytes', 'enum', 'path', 'dict']

#: Config a field type cannot be constructed without.
KIND_KWARGS = {'enum': {'enum': Color}}

#: Well-formed values per kind, on top of the generic sweep — the wire forms
#: each kind is actually supposed to accept, plus the near-misses.
KIND_VALUES = {
    'uuid': [
        '12345678-1234-5678-1234-567812345678',
        '12345678123456781234567812345678',
        'not-a-uuid',
        uuid.UUID('12345678-1234-5678-1234-567812345678'),
    ],
    'date': [
        '2026-08-31',
        '2026-02-30',
        '31/08/2026',
        datetime.date(2026, 8, 31),
        datetime.datetime(2026, 8, 31, 12, 0),
    ],
    'datetime': [
        '2026-08-31T12:30:00',
        '2026-08-31 12:30:00+00:00',
        '2026-08-31',
        'nope',
        datetime.datetime(2026, 8, 31, 12, 0),
    ],
    'time': ['12:30', '12:30:45', '25:00', datetime.time(12, 30)],
    'timedelta': [0, 90.5, '90.5', -1, datetime.timedelta(seconds=90.5)],
    'decimal': ['3.14159', '1e400', 'nan', '0', 3.5, 1, decimal.Decimal('3.14')],
    'bytes': ['ZGVhZGJlZWY=', 'deadbeef', 'not!base64!', b'raw', bytearray(b'raw')],
    'enum': [0, 1, 2, '0', 'RED', Color.RED],
    'path': ['a/b/c.txt', '/abs/path', '', pathlib.Path('a/b')],
    'dict': [{}, {'k': 'v'}, {'n': 1}],
}

#: A stable instance, so its ``repr`` (address included) is identical across
#: both implementations' error messages.
SENTINEL = object()

#: Values swept across every scalar kind, in both strict and lax mode.
VALUES = [
    7,
    -3,
    0,
    7.9,
    -0.5,
    True,
    False,
    '7',
    '7.5',
    'nope',
    '',
    'true',
    'YES',
    ' on ',
    'off',
    '0',
    None,
    [],
    {},
    [1, 2],
    SENTINEL,
]
VALUE_IDS = [
    'int7',
    'int-3',
    'int0',
    'float7.9',
    'float-0.5',
    'True',
    'False',
    'str7',
    'str7.5',
    'strnope',
    'empty',
    'strtrue',
    'strYES',
    'str_on_',
    'stroff',
    'str0',
    'None',
    'emptylist',
    'emptydict',
    'list12',
    'object',
]


def one_field(kind: str, *, validate: bool = True, accel: bool = True, **field_kwargs):
    """A one-field class of the given kind."""
    field = KIND_FIELDS[kind]
    kwargs = {**KIND_KWARGS.get(kind, {}), **field_kwargs}

    @s.seared(accel=accel, validate=validate)
    class C(s.Seared):
        v: object = field(**kwargs)

    return C


def both(kind: str, **kwargs):
    """``(accelerated, pure)`` one-field classes, asserted to differ only in that."""
    fast = one_field(kind, accel=True, **kwargs)
    slow = one_field(kind, accel=False, **kwargs)
    assert fast.__seared_accel__.accelerated is True, f'{kind} is not accelerated: {fast.__seared_accel__.reason}'
    assert slow.__seared_accel__.accelerated is False
    return fast, slow


def bench_schema(*, validate: bool = True, accel: bool = True):
    """seared's bench schema — the canonical Tier 1 shape."""

    @s.seared(accel=accel, validate=validate)
    class Inner(s.Seared):
        x: int = s.Int(required=True)
        y: float = s.Float(required=True)
        label: str | None = s.Str(default=None)

    @s.seared(accel=accel, validate=validate)
    class Outer(s.Seared):
        name: str = s.Str(required=True)
        flag: bool = s.Bool(default=False)
        items: list[Inner] = s.T(Inner, many=True, required=True)
        tags: list[str] = s.Str(many=True, default_factory=list)

    return Outer


def raw(cls, **attrs):
    """An instance with attributes set directly, skipping ``__init__``.

    Lets a dump test feed values that ``__init__`` would never produce.
    """
    obj = cls.__new__(cls)
    for k, v in attrs.items():
        object.__setattr__(obj, k, v)
    return obj


def same(a, b):
    """Equality, except that NaN counts as equal to NaN.

    ``Decimal('NaN') != Decimal('NaN')`` by IEEE rule, so plain ``==`` reports
    a divergence where the two implementations in fact produced identical
    objects. Types are compared too, so int-vs-bool or float-vs-Decimal drift
    still fails.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is not type(b):
        return False
    if a != a and b != b:  # both NaN: identical by construction, unequal by rule
        return repr(a) == repr(b)
    return a == b


def outcome(fn, *args):
    """``('ok', value)`` or ``('raised', ExcType, message)`` — comparable."""
    try:
        return ('ok', fn(*args))
    except Exception as exc:  # noqa: BLE001 — comparing failures is the point
        return ('raised', type(exc).__name__, str(exc))


def load_outcome(cls, payload, attr='v'):
    """Load outcome, carrying the value's *type* so int/bool/float divergence shows."""
    try:
        obj = cls.load(payload)
    except Exception as exc:  # noqa: BLE001 — comparing failures is the point
        return ('raised', type(exc).__name__, str(exc))
    value = getattr(obj, attr)
    return ('ok', value, type(value).__name__)


@pytest.fixture
def pair():
    """``(accelerated, pure)`` builds of the bench schema."""
    return bench_schema(accel=True), bench_schema(accel=False)
