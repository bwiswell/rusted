"""Tests for ``src/dump.rs`` — instance → wire dict."""

from __future__ import annotations

import pytest
import seared as s
from conftest import NULL_BASE, bench_schema, nested_bytes_schema, null_schema, one_field, outcome, path_schema, raw

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


def _nested(root):
    """``root``'s nested classes by name, read off its fields."""
    classes = {}
    for spec in root.__seared_fields__:
        schema = getattr(spec.field, 'schema', None)
        if schema is not None:
            classes[schema.__name__] = schema
            classes.update(_nested(schema))
    return classes


def _bad_objects(root):
    """(label, instance) — each fails to dump somewhere specific."""
    c = _nested(root)
    leaf, port, mid = c['Leaf'], c['Port'], c['Mid']
    return [
        ('leaf-in-a-list', raw(root, items=[leaf(n=1), raw(leaf, n='x', tags=[])])),
        ('keyed-key', raw(root, items=[], readers={'r-1': raw(port, port='bad')})),
        ('not-a-leaf', raw(root, items=[7])),
        ('many-not-a-list', raw(root, items=7)),
        ('three-deep', raw(root, items=[], mid=raw(mid, leaves=[leaf(n=1), raw(leaf, n=2, tags=[None])]))),
    ]


class TestErrorPaths:
    """Pin the ``SPEC_ABI`` 2 error contract on ``dump``: attribute names, not wire keys."""

    @pytest.mark.parametrize('label', ['leaf-in-a-list', 'keyed-key', 'not-a-leaf', 'many-not-a-list', 'three-deep'])
    def test_parts_match(self, label):
        fast, slow = path_schema(accel=True), path_schema(accel=False)
        got = outcome(fast.dump, dict(_bad_objects(fast))[label])
        want = outcome(slow.dump, dict(_bad_objects(slow))[label])
        assert want[:2] == ('raised', 'ValidationError')
        assert got == want

    def test_a_dump_path_names_attributes(self):
        fast = path_schema(accel=True)
        with pytest.raises(s.ValidationError) as info:
            fast.dump(dict(_bad_objects(fast))['keyed-key'])
        # ``port`` is the attribute; ``p`` is its wire key, which a load path would name.
        assert (info.value.path, info.value.cls_name) == (('readers', 'r-1', 'port'), 'Root')


class TestNullRule:
    """Pin: ``dump`` writes ``None`` only on a nullable field, and leaves it out everywhere else."""

    @pytest.fixture(scope='class')
    def rows(self):
        return null_schema(accel=True), null_schema(accel=False)

    def test_only_a_nullable_none_is_written(self, rows):
        fast, slow = rows
        nones = {'n_dflt': None, 'n_req': None, 'n_many': None, 'n_keyed': None, 'n_leaf': None, 'bare': None}
        got = fast.dump(fast(req=1, **nones))
        assert got == slow.dump(slow(req=1, **nones))
        assert got == {
            'req': 1,
            'dflt': 5,
            'n_dflt': None,
            'n_req': None,
            'factory': [],
            'n_many': None,
            'n_keyed': None,
            'n_leaf': None,
        }

    def test_an_unset_slot_counts_as_none(self, rows):
        # seared reads with ``getattr(obj, attr, None)``, so on a nullable field
        # an unset slot is written as ``None``.
        fast, slow = rows
        got = outcome(fast.dump, raw(fast, req=1))
        assert got == outcome(slow.dump, raw(slow, req=1))
        assert got[1] == {'req': 1, 'n_dflt': None, 'n_req': None, 'n_many': None, 'n_keyed': None, 'n_leaf': None}

    def test_a_nullable_none_survives_a_round_trip_over_its_default(self, rows):
        fast, _ = rows
        row = fast(**NULL_BASE, n_dflt=None)
        assert fast.load(fast.dump(row)).n_dflt is None

    def test_dump_false_wins_over_nullable(self):
        fast = one_field('str', accel=True, nullable=True, dump=False)
        slow = one_field('str', accel=False, nullable=True, dump=False)
        assert fast.dump(raw(fast, v=None)) == slow.dump(raw(slow, v=None)) == {}
