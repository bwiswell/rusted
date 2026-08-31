"""Shared helpers.

Every test compares the compiled core against seared's own Python path, so
the pattern throughout is: build the same schema twice — once normally
(rusted takes it) and once with ``accel=False`` (it can't) — and assert the
two are indistinguishable, values *and* exceptions.
"""

from __future__ import annotations

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


KIND_FIELDS = {'int': s.Int, 'float': s.Float, 'str': s.Str, 'bool': s.Bool}

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

    @s.seared(accel=accel, validate=validate)
    class C(s.Seared):
        v: object = field(**field_kwargs)

    return C


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
