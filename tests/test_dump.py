"""Tests for ``src/dump.rs`` — instance → wire dict."""

from __future__ import annotations

import pytest
import seared as s
from conftest import bench_schema, nested_bytes_schema, one_field, outcome, raw

PAYLOAD = {
    'name': 'demo',
    'flag': True,
    'items': [{'x': 1, 'y': 1.5, 'label': 'a'}],
    'tags': ['alpha'],
}


class TestOmissions:
    def test_round_trip_is_exact(self, pair):
        fast, slow = pair
        assert fast.dump(fast.load(PAYLOAD)) == slow.dump(slow.load(PAYLOAD)) == PAYLOAD

    def test_none_values_are_skipped(self, pair):
        fast, slow = pair
        minimal = {'name': 'd', 'items': []}
        # `flag` defaults to False (kept); `label` inside items would be None.
        assert fast.dump(fast.load(minimal)) == slow.dump(slow.load(minimal))

    def test_dump_false_field_is_omitted(self):
        @s.seared
        class Hidden(s.Seared):
            shown: str | None = s.Str(default=None)
            secret: str | None = s.Str(dump=False)

        assert Hidden.__seared_accel__.accelerated is True
        assert Hidden.dump(Hidden.load({'shown': 'a', 'secret': 'b'})) == {'shown': 'a'}

    def test_unset_slot_dumps_as_absent(self):
        # seared reads with `getattr(obj, attr, None)`; an instance built via
        # __new__ with no assignment must not raise.
        fast = one_field('int', accel=True)
        slow = one_field('int', accel=False)
        assert outcome(fast.dump, fast.__new__(fast)) == outcome(slow.dump, slow.__new__(slow))


class TestNested:
    def test_type_guard_uses_the_parents_flag(self):
        fast = bench_schema(accel=True)
        slow = bench_schema(accel=False)
        bad = raw(fast, name='n', flag=False, items=['not an Inner'], tags=[])
        bad_slow = raw(slow, name='n', flag=False, items=['not an Inner'], tags=[])
        assert outcome(fast.dump, bad) == outcome(slow.dump, bad_slow)

    def test_nested_dump_matches(self, pair):
        fast, slow = pair
        assert fast.dump(fast.load(PAYLOAD))['items'] == slow.dump(slow.load(PAYLOAD))['items']

    @pytest.mark.parametrize('fmt', ['json', 'msgpack'])
    def test_format_hint_crosses_the_nesting_boundary(self, fmt):
        # The one place the carrier hint is observable is a `Bytes` *inside*
        # a nested class. Before seared 0.3.1 `T` dropped the hint, so pure
        # seared emitted base64 under msgpack where this core went native —
        # the differential suite's blind spot, since the bench schema is
        # Tier 1 only.
        fast, slow = nested_bytes_schema(accel=True), nested_bytes_schema(accel=False)
        assert fast.__seared_accel__.accelerated is True, fast.__seared_accel__.reason
        a = raw(fast, one=raw(fast.__inner__, blob=b'\x01'), many=[raw(fast.__inner__, blob=b'\x02')])
        b = raw(slow, one=raw(slow.__inner__, blob=b'\x01'), many=[raw(slow.__inner__, blob=b'\x02')])
        got, want = fast.dump(a, fmt), slow.dump(b, fmt)
        assert got == want
        expected_blob = b'\x01' if fmt == 'msgpack' else 'AQ=='
        assert got['one']['blob'] == expected_blob


class TestContainers:
    def test_keyed_round_trip(self):
        @s.seared
        class Counts(s.Seared):
            counts: dict[str, int] = s.Int(keyed=True, required=True)

        assert Counts.__seared_accel__.accelerated is True
        assert Counts.dump(Counts.load({'counts': {'a': 1, 'b': 2}})) == {'counts': {'a': 1, 'b': 2}}

    @pytest.mark.parametrize('value', [[1, 2], (1, 2), [], 'nope', 7, {'a': 1}])
    def test_many_shapes_match(self, value):
        fast = one_field('int', accel=True, many=True)
        slow = one_field('int', accel=False, many=True)
        assert outcome(fast.dump, raw(fast, v=value)) == outcome(slow.dump, raw(slow, v=value))

    @pytest.mark.parametrize('value', [{'a': 1}, {}, [1], 'nope', 7])
    def test_keyed_shapes_match(self, value):
        fast = one_field('int', accel=True, keyed=True)
        slow = one_field('int', accel=False, keyed=True)
        assert outcome(fast.dump, raw(fast, v=value)) == outcome(slow.dump, raw(slow, v=value))
