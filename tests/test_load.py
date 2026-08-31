"""Tests for ``src/load.rs`` — wire dict → instance."""

from __future__ import annotations

import pytest
import seared as s
from conftest import bench_schema, load_outcome, one_field, outcome

PAYLOAD = {
    'name': 'demo',
    'flag': True,
    'items': [{'x': i, 'y': i * 1.5, 'label': f'i{i}'} for i in range(3)],
    'tags': ['alpha', 'beta'],
}


class TestKeyResolution:
    def test_required_missing(self, pair):
        fast, slow = pair
        assert outcome(fast.load, {'items': []}) == outcome(slow.load, {'items': []})

    def test_defaults_when_absent(self, pair):
        fast, slow = pair
        minimal = {'name': 'd', 'items': []}
        a, b = fast.load(minimal), slow.load(minimal)
        assert (a.flag, a.tags) == (b.flag, b.tags) == (False, [])

    def test_default_factory_is_per_instance(self, pair):
        fast, _slow = pair
        first, second = fast.load({'name': 'a', 'items': []}), fast.load({'name': 'b', 'items': []})
        first.tags.append('x')
        assert second.tags == []

    def test_static_default_is_shared_exactly_as_seared_shares_it(self):
        # seared's load path assigns the same `missing` object to every
        # instance — deep-copying is the constructor wrapper's job, not
        # load's. Matching that matters more than "fixing" it here.
        template = [1]
        fast = one_field('int', accel=True, many=True, default=template)
        slow = one_field('int', accel=False, many=True, default=template)
        assert fast.load({}).v is slow.load({}).v is template

    def test_unknown_keys_ignored(self, pair):
        fast, slow = pair
        payload = {**PAYLOAD, 'surprise': 1}
        assert fast.dump(fast.load(payload)) == slow.dump(slow.load(payload))

    def test_explicit_null_beats_default(self, pair):
        fast, slow = pair
        payload = {**PAYLOAD, 'flag': None}
        assert fast.load(payload).flag is slow.load(payload).flag is None


class TestShape:
    @pytest.mark.parametrize('data', [None, [], 'nope', 7, (), 3.5])
    def test_non_dict_input(self, pair, data):
        fast, slow = pair
        assert outcome(fast.load, data) == outcome(slow.load, data)

    def test_error_is_seareds_own_class(self, pair):
        fast, _slow = pair
        # A caller's `except s.ValidationError` must keep catching ours.
        with pytest.raises(s.ValidationError):
            fast.load([])


class TestNested:
    def test_builds_real_nested_instances(self, pair):
        fast, _slow = pair
        obj = fast.load(PAYLOAD)
        assert [i.x for i in obj.items] == [0, 1, 2]
        assert type(obj.items[0]).__name__ == 'Inner'

    def test_existing_instance_passes_through(self):
        # T.deserialize returns an already-built instance untouched.
        outer = bench_schema(accel=True)
        inner_cls = type(outer.load(PAYLOAD).items[0])
        instance = inner_cls.load({'x': 9, 'y': 1.0})
        obj = outer.load({'name': 'n', 'items': [instance]})
        assert obj.items[0] is instance

    def test_nested_runs_under_its_own_validate_flag(self):
        # A lax Inner inside a strict Outer stays lax, as `schema.load` does.
        @s.seared(validate=False)
        class Inner(s.Seared):
            x: int = s.Int(required=True)

        @s.seared
        class Outer(s.Seared):
            inner: Inner = s.T(Inner, required=True)

        assert Outer.__seared_accel__.accelerated is True
        assert Outer.load({'inner': {'x': True}}).inner.x == 1

    def test_malformed_nested(self, pair):
        fast, slow = pair
        for items in ([7], [None], ['x'], [{}], {}, 7):
            payload = {**PAYLOAD, 'items': items}
            assert outcome(fast.load, payload) == outcome(slow.load, payload)


class TestConstruction:
    def test_produces_a_genuine_instance(self, pair):
        fast, _slow = pair
        obj = fast.load(PAYLOAD)
        assert isinstance(obj, fast)
        assert obj == fast.load(PAYLOAD)  # dataclass __eq__ still works

    def test_repr_matches(self, pair):
        fast, slow = pair
        assert repr(fast.load(PAYLOAD)) == repr(slow.load(PAYLOAD))

    def test_attribute_model_matches(self, pair):
        # Not asserting that extra attributes are refused — `s.Seared` has no
        # `__slots__`, so every subclass carries a `__dict__` and accepts them.
        # What matters here is that both paths agree.
        fast, slow = pair
        a, b = fast.load(PAYLOAD), slow.load(PAYLOAD)
        a.extra = 1
        b.extra = 1
        assert (a.extra, hasattr(a, '__dict__')) == (b.extra, hasattr(b, '__dict__'))

    def test_slots_false_classes_load(self):
        fast = bench_schema(accel=True)
        assert fast.load(PAYLOAD).name == 'demo'

        @s.seared(slots=False)
        class Loose(s.Seared):
            x: int = s.Int(required=True)

        assert Loose.__seared_accel__.accelerated is True
        assert Loose.load({'x': 1}).x == 1


class TestLaxMode:
    @pytest.mark.parametrize(
        'payload',
        [
            {'name': 42, 'items': []},
            {'name': 'n', 'flag': 'nonsense', 'items': []},
            {'name': 'n', 'items': [{'x': True, 'y': 1}]},
            {'name': 'n', 'items': [], 'tags': 'notalist'},
        ],
    )
    def test_matches(self, payload):
        fast = bench_schema(validate=False, accel=True)
        slow = bench_schema(validate=False, accel=False)
        assert load_outcome(fast, payload, attr='name') == load_outcome(slow, payload, attr='name')
        assert outcome(fast.dump, fast.load(payload)) == outcome(slow.dump, slow.load(payload))
