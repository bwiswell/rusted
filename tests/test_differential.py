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

import pytest
from conftest import bench_schema, outcome

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
