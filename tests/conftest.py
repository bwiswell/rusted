"""Shared helpers.

Every test compares the compiled core against seared's own Python path, so
the pattern throughout is: build the same schema twice — once normally
(rusted takes it) and once with ``accel=False`` (it can't) — and assert the
two are indistinguishable, values *and* exceptions.
"""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import math
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
#: Tier 2, split the way ``src/kinds/`` is: the date-likes in ``temporal.rs``,
#: the rest in ``value.rs``.
TEMPORAL = ['date', 'datetime', 'time', 'timedelta']
VALUE = ['uuid', 'decimal', 'bytes', 'enum', 'path', 'dict']
TIER2 = TEMPORAL + VALUE
MODES = [(True, 'strict'), (False, 'lax')]

#: ``(kind, validate)`` for each Tier 1 kind, in both modes. Lax ``bool``
#: joined with ``SPEC_ABI`` 2, which carries seared 0.3.3's lax-``Bool`` fix.
TIER1_MODES = [pytest.param(kind, validate, id=f'{kind}-{mode}') for validate, mode in MODES for kind in TIER1]

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
        dt.date(2026, 8, 31),
        dt.datetime(2026, 8, 31, 12, 0),
    ],
    'datetime': [
        '2026-08-31T12:30:00',
        '2026-08-31 12:30:00+00:00',
        '2026-08-31',
        'nope',
        dt.datetime(2026, 8, 31, 12, 0),
    ],
    'time': ['12:30', '12:30:45', '25:00', dt.time(12, 30)],
    'timedelta': [0, 90.5, '90.5', -1, dt.timedelta(seconds=90.5)],
    'decimal': ['3.14159', '1e400', 'nan', '0', 3.5, 1, decimal.Decimal('3.14')],
    'bytes': ['ZGVhZGJlZWY=', 'deadbeef', 'not!base64!', b'raw', bytearray(b'raw')],
    # Int-valued: seared's `Int` rules pick the key, so a fractional float or a
    # bool is refused in strict mode and truncated or read as 1 in lax mode;
    # `inf` and a `Decimal` are refused in both.
    'enum': [0, 1, 2, '0', 'RED', Color.RED, 1.0, 1.9, True, math.inf, decimal.Decimal(1)],
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
    1,
    7.9,
    -0.5,
    3.0,
    math.inf,
    math.nan,
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
    'int1',
    'float7.9',
    'float-0.5',
    'float3.0',
    'inf',
    'nan',
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


def checked(cls, *, accel: bool):
    """``cls``, asserted to be on the path that was asked for.

    Every builder below goes through this. A fast side that seared declined
    would turn a parity test into Python compared with Python, and it would
    pass: seared 0.3.3's lax ``Bool`` decline did exactly that to 102 cases
    here before the builders checked.
    """
    info = cls.__seared_accel__
    if accel:
        assert info.accelerated is True, f'{cls.__qualname__} is not accelerated: {info.reason}'
    else:
        assert info.accelerated is False, f'{cls.__qualname__} is accelerated despite accel=False'
    return cls


def one_field(kind: str, *, validate: bool = True, accel: bool = True, **field_kwargs):
    """A one-field class of the given kind."""
    field = KIND_FIELDS[kind]
    kwargs = {**KIND_KWARGS.get(kind, {}), **field_kwargs}

    @s.seared(accel=accel, validate=validate)
    class C(s.Seared):
        v: object = field(**kwargs)

    return checked(C, accel=accel)


def both(kind: str, **kwargs):
    """``(accelerated, pure)`` one-field classes, differing only in that."""
    return one_field(kind, accel=True, **kwargs), one_field(kind, accel=False, **kwargs)


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

    return checked(Outer, accel=accel)


def nested_bytes_schema(*, validate: bool = True, accel: bool = True):
    """A `Bytes` behind a `T` — the one shape where `format=` is observable.

    The bench schema is Tier 1 and ignores the carrier hint, so a core that
    threaded `format=` through nested classes while seared dropped it (as it
    did before 0.3.1) passed every other test here. The inner class is
    exposed as ``Outer.__inner__`` so tests can build instances of it.
    """

    @s.seared(accel=accel, validate=validate)
    class Inner(s.Seared):
        blob: bytes = s.Bytes(required=True)

    @s.seared(accel=accel, validate=validate)
    class Outer(s.Seared):
        one: Inner = s.T(Inner, required=True)
        many: list[Inner] = s.T(Inner, many=True, default_factory=list)

    Outer.__inner__ = Inner
    return checked(Outer, accel=accel)


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


def failure(exc):
    """``('raised', ExcType, message, …parts)`` — a failure, comparable.

    A ``ValidationError`` also carries its parts (``SPEC_ABI`` 2). Comparing
    the message alone would miss a path that renders the same but holds
    different types: an index ``1`` against a keyed-dict key ``'1'``.
    """
    if isinstance(exc, s.ValidationError):
        return ('raised', type(exc).__name__, str(exc), exc.reason, exc.path, exc.cls_name)
    return ('raised', type(exc).__name__, str(exc))


def outcome(fn, *args):
    """``('ok', value)``, or :func:`failure`'s tuple — comparable."""
    try:
        return ('ok', fn(*args))
    except Exception as exc:  # noqa: BLE001 — comparing failures is the point
        return failure(exc)


def load_outcome(cls, payload, attr='v'):
    """Load outcome, carrying the value's *type* so int/bool/float divergence shows."""
    try:
        obj = cls.load(payload)
    except Exception as exc:  # noqa: BLE001 — comparing failures is the point
        return failure(exc)
    value = getattr(obj, attr)
    return ('ok', value, type(value).__name__)


def null_schema(*, validate: bool = True, accel: bool = True):
    """One field per column of seared's null table, plus nullable containers.

    ``req`` / ``dflt`` / ``bare`` aren't nullable, so a ``null`` reads as an
    absent key; ``n_dflt`` / ``n_req`` are, so it's ``None``, even over a
    default. The ``n_*`` containers and ``n_leaf`` are nullable as a whole.
    """

    @s.seared(accel=accel, validate=validate)
    class Leaf(s.Seared):
        n: int = s.Int(required=True)

    @s.seared(accel=accel, validate=validate)
    class Row(s.Seared):
        req: int = s.Int(required=True)
        dflt: int = s.Int(default=5)
        bare: int | None = s.Int()
        n_dflt: int | None = s.Int(nullable=True, default=5)
        n_req: int | None = s.Int(nullable=True, required=True)
        factory: list[int] = s.Int(many=True, default_factory=list)
        n_many: list[int] | None = s.Int(many=True, nullable=True, default_factory=list)
        n_keyed: dict[str, int] | None = s.Int(keyed=True, nullable=True)
        leaf: Leaf | None = s.T(Leaf)
        n_leaf: Leaf | None = s.T(Leaf, nullable=True)

    return checked(Row, accel=accel)


#: ``null_schema``'s columns, each with a value it loads.
NULL_COLUMNS = {
    'req': 7,
    'dflt': 7,
    'bare': 7,
    'n_dflt': 7,
    'n_req': 7,
    'factory': [1],
    'n_many': [1],
    'n_keyed': {'a': 1},
    'leaf': {'n': 1},
    'n_leaf': {'n': 1},
}

#: The payload every other column is filled from, so only one column varies.
NULL_BASE = {'req': 1, 'n_req': 2}


def path_schema(*, accel: bool = True):
    """Nested ``T``, ``many``, ``keyed`` and ``data_key``, for error paths.

    Every class is built on the asked-for path, so an error raised deep inside
    passes back up through compiled levels only (or Python ones only).
    """

    @s.seared(accel=accel)
    class Leaf(s.Seared):
        n: int = s.Int(required=True)
        tags: list[str] = s.Str(many=True, default_factory=list)

    @s.seared(accel=accel)
    class Port(s.Seared):
        port: int = s.Int(required=True, data_key='p')

    @s.seared(accel=accel)
    class Mid(s.Seared):
        leaves: list[Leaf] = s.T(Leaf, many=True, required=True)

    @s.seared(accel=accel)
    class Root(s.Seared):
        items: list[Leaf] = s.T(Leaf, many=True, required=True)
        readers: dict[str, Port] = s.T(Port, keyed=True, default_factory=dict)
        counts: dict[str, int] = s.Int(keyed=True, default_factory=dict)
        mid: Mid | None = s.T(Mid, default=None)

    return checked(Root, accel=accel)


@pytest.fixture
def pair():
    """``(accelerated, pure)`` builds of the bench schema."""
    return bench_schema(accel=True), bench_schema(accel=False)
