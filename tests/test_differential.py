"""The differential suite — the contract, not any one source file.

Everything else here tests a module; this asserts the property the whole
project rests on: with rusted installed, a seared class behaves *identically*
to one without it. Same values, same types, same exception classes, same
message text.

This is the seed of the generated matrix described in
``project-plans/02-rusted-outline.md`` §8. It sweeps the payload shapes that
have historically hidden divergence; broadening it to every kind and flag
combination is the next step, not a finished job.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import seared as s
from conftest import Color, bench_schema, outcome, same

if TYPE_CHECKING:
    # Annotation-only: seared reads the Field defaults, not the annotations.
    import datetime
    import decimal
    import pathlib
    import uuid

PAYLOAD = {
    'name': 'demo',
    'flag': True,
    'items': [{'x': i, 'y': i * 1.5, 'label': f'i{i}'} for i in range(20)],
    'tags': ['alpha', 'beta', 'gamma'],
}

CASES = {
    'valid': PAYLOAD,
    'missing-required': {'items': []},
    'missing-nested-required': {**PAYLOAD, 'items': [{'y': 1.0}]},
    'int-from-str': {**PAYLOAD, 'items': [{'x': '7', 'y': 1.0}]},
    'int-from-float': {**PAYLOAD, 'items': [{'x': 7.9, 'y': 1.0}]},
    'int-from-bool': {**PAYLOAD, 'items': [{'x': True, 'y': 1.0}]},
    'int-garbage': {**PAYLOAD, 'items': [{'x': 'nope', 'y': 1.0}]},
    'int-overflow-str': {**PAYLOAD, 'items': [{'x': '9' * 400, 'y': 1.0}]},
    'float-from-str': {**PAYLOAD, 'items': [{'x': 1, 'y': '2.5'}]},
    'float-from-int': {**PAYLOAD, 'items': [{'x': 1, 'y': 3}]},
    'float-garbage': {**PAYLOAD, 'items': [{'x': 1, 'y': 'nope'}]},
    'float-inf': {**PAYLOAD, 'items': [{'x': 1, 'y': 'inf'}]},
    'str-wrong-type': {**PAYLOAD, 'name': 42},
    'str-none': {**PAYLOAD, 'name': None},
    'bool-from-str': {**PAYLOAD, 'flag': 'yes'},
    'bool-from-int': {**PAYLOAD, 'flag': 1},
    'bool-garbage': {**PAYLOAD, 'flag': []},
    'many-not-a-list': {**PAYLOAD, 'tags': 'alpha'},
    'many-tuple': {**PAYLOAD, 'tags': ('alpha',)},
    'many-with-none': {**PAYLOAD, 'tags': [None]},
    'nested-not-a-dict': {**PAYLOAD, 'items': [7]},
    'nested-not-a-list': {**PAYLOAD, 'items': {'x': 1}},
    'unknown-keys': {**PAYLOAD, 'surprise': 1},
    'defaults-omitted': {'name': 'd', 'items': []},
    'empty': {},
    'top-level-list': ['not', 'a', 'dict'],
    'top-level-none': None,
}
IDS = list(CASES)


@pytest.fixture(params=[True, False], ids=['strict', 'lax'])
def schemas(request):
    validate = request.param
    return bench_schema(validate=validate, accel=True), bench_schema(validate=validate, accel=False)


@pytest.mark.parametrize('case', IDS)
class TestParity:
    def test_load(self, schemas, case):
        fast, slow = schemas
        payload = CASES[case]
        got, want = outcome(fast.load, payload), outcome(slow.load, payload)
        if want[0] == 'raised':
            assert got == want
        else:
            # Instances are of two distinct classes — compare through dump.
            assert got[0] == 'ok'
            assert fast.dump(got[1]) == slow.dump(want[1])

    def test_dump(self, schemas, case):
        fast, slow = schemas
        payload = CASES[case]
        try:
            fast_obj, slow_obj = fast.load(payload), slow.load(payload)
        except Exception:  # noqa: BLE001 — load parity is the other test's job
            pytest.skip('payload does not load')
        assert outcome(fast.dump, fast_obj) == outcome(slow.dump, slow_obj)


class TestCodecsRideAlong:
    """Every seared codec funnels through load/dump, so all of them inherit this."""

    def test_loads_dumps(self, pair):
        fast, slow = pair
        text = json.dumps(PAYLOAD)
        assert json.loads(fast.dumps(fast.loads(text))) == json.loads(slow.dumps(slow.loads(text)))

    def test_to_from_json(self, pair):
        fast, slow = pair
        assert fast.to_json(fast.load(PAYLOAD)) == slow.to_json(slow.load(PAYLOAD))
        assert fast.dump(fast.from_json(json.dumps(PAYLOAD))) == PAYLOAD

    def test_format_hint_threads_through(self, pair):
        # Tier 1 kinds ignore `format=`, but it must be accepted and passed
        # down to nested schemas identically.
        fast, slow = pair
        assert fast.dump(fast.load(PAYLOAD, 'msgpack'), 'msgpack') == slow.dump(
            slow.load(PAYLOAD, 'msgpack'),
            'msgpack',
        )


# ---------------------------------------------------------------------------
# A realistic mixed class — Tier 1 and Tier 2 together
# ---------------------------------------------------------------------------


def mixed_schema(*, validate: bool = True, accel: bool = True):
    """The shape a real message class actually has.

    Before Tier 2 this class was disqualified in its entirety by any one of
    `when`, `blob`, `state` or `where` — all-or-nothing cuts both ways, which
    is the whole reason Tier 2 exists.
    """

    @s.seared(accel=accel, validate=validate)
    class Reading(s.Seared):
        ident: uuid.UUID = s.UUID(required=True)
        when: datetime.datetime = s.DateTime(required=True)
        day: datetime.date = s.Date()
        elapsed: datetime.timedelta = s.TimeDelta()
        amount: decimal.Decimal = s.Decimal()
        blob: bytes = s.Bytes()
        state: Color = s.Enum(enum=Color)
        where: pathlib.Path = s.Path()
        extra: dict = s.Dict()
        label: str = s.Str(required=True)
        count: int = s.Int(default=0)
        ratios: list[float] = s.Float(many=True, default_factory=list)

    return Reading


MIXED = {
    'ident': '12345678-1234-5678-1234-567812345678',
    'when': '2026-08-31T12:30:00',
    'day': '2026-08-31',
    'elapsed': 90.5,
    'amount': '3.14159',
    'blob': 'ZGVhZGJlZWY=',
    'state': 1,
    'where': 'a/b/c.txt',
    'extra': {'k': 'v'},
    'label': 'demo',
    'count': 7,
    'ratios': [1.0, 2.5],
}

MIXED_CASES = {
    'valid': MIXED,
    'minimal': {'ident': MIXED['ident'], 'when': MIXED['when'], 'label': 'x'},
    'bad-uuid': {**MIXED, 'ident': 'not-a-uuid'},
    'bad-datetime': {**MIXED, 'when': 'nope'},
    'bad-date': {**MIXED, 'day': 99},
    'bad-decimal': {**MIXED, 'amount': 'nope'},
    'bad-bytes': {**MIXED, 'blob': 'not!base64!'},
    'bad-enum': {**MIXED, 'state': 99},
    'bad-path': {**MIXED, 'where': 7},
    'bad-dict': {**MIXED, 'extra': [1]},
    'bad-timedelta': {**MIXED, 'elapsed': 'nope'},
    'nulls': dict.fromkeys(MIXED),
    'missing-required': {'label': 'x'},
}


class TestMixedClassParity:
    @pytest.fixture(params=[True, False], ids=['strict', 'lax'])
    def mixed(self, request):
        validate = request.param
        return (
            mixed_schema(validate=validate, accel=True),
            mixed_schema(validate=validate, accel=False),
        )

    def test_the_whole_class_is_accelerated(self):
        fast = mixed_schema()
        assert fast.__seared_accel__.accelerated is True, fast.__seared_accel__.reason

    @pytest.mark.parametrize('case', list(MIXED_CASES))
    def test_load(self, mixed, case):
        fast, slow = mixed
        payload = MIXED_CASES[case]
        got, want = outcome(fast.load, payload), outcome(slow.load, payload)
        if want[0] == 'raised':
            assert same(got, want)
        else:
            assert got[0] == 'ok'
            assert same(fast.dump(got[1]), slow.dump(want[1]))

    @pytest.mark.parametrize('case', list(MIXED_CASES))
    def test_round_trip(self, mixed, case):
        fast, slow = mixed
        payload = MIXED_CASES[case]
        try:
            fast_obj, slow_obj = fast.load(payload), slow.load(payload)
        except Exception:  # noqa: BLE001 — load parity is the other test's job
            pytest.skip('payload does not load')
        assert same(outcome(fast.dump, fast_obj), outcome(slow.dump, slow_obj))

    def test_valid_payload_round_trips_exactly(self, mixed):
        fast, _slow = mixed
        assert fast.dump(fast.load(MIXED)) == MIXED

    def test_msgpack_carrier_switches_bytes(self, mixed):
        fast, slow = mixed
        a, b = fast.load(MIXED, 'msgpack'), slow.load(MIXED, 'msgpack')
        assert same(fast.dump(a, 'msgpack'), slow.dump(b, 'msgpack'))
        assert fast.dump(a, 'msgpack')['blob'] == b'deadbeef'
