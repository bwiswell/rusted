"""Tests for ``src/load.rs`` — wire dict → instance."""

from __future__ import annotations

import functools
import operator

import pytest
import seared as s
from conftest import (
    NULL_BASE,
    NULL_COLUMNS,
    bench_schema,
    load_outcome,
    nested_bytes_schema,
    null_schema,
    one_field,
    outcome,
    path_schema,
)

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

    def test_a_null_on_a_defaulted_field_reads_as_absent(self, pair):
        # ``SPEC_ABI`` 2: the default, ``False``, where ABI 1 loaded ``None``.
        fast, slow = pair
        payload = {**PAYLOAD, 'flag': None}
        assert fast.load(payload).flag is slow.load(payload).flag is False


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

    @pytest.mark.parametrize('fmt', ['json', 'msgpack'])
    def test_format_hint_crosses_the_nesting_boundary(self, fmt):
        # Load-side twin of the dump test: `Bytes` accepts raw bytes off a
        # binary carrier and base64 text off JSON, and both must reach a
        # nested class identically on either path.
        fast, slow = nested_bytes_schema(accel=True), nested_bytes_schema(accel=False)
        blob = b'\x01' if fmt == 'msgpack' else 'AQ=='
        payload = {'one': {'blob': blob}, 'many': [{'blob': blob}]}
        a, b = fast.load(payload, fmt), slow.load(payload, fmt)
        assert (a.one.blob, a.many[0].blob) == (b.one.blob, b.many[0].blob) == (b'\x01', b'\x01')

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

    def test_plain_dataclass_field_declines(self):
        # `b` is a dataclass field but not a seared Field: it never reaches
        # the spec, and only `__init__` would set it. Pure seared runs
        # `__init__` from `load`; this core builds through `__new__` and
        # would leave the slot unset, so seared (0.3.1+) must decline the
        # class rather than hand it over.
        @s.seared
        class Mixed(s.Seared):
            a: int = s.Int(required=True)
            b: int = 5

        info = Mixed.__seared_accel__
        assert info.accelerated is False
        assert 'Mixed.b is a plain dataclass field' in info.reason
        obj = Mixed.load({'a': 1})
        assert (obj.a, obj.b) == (1, 5)
        assert repr(obj) == repr(Mixed(a=1))
        assert obj == Mixed(a=1)

    def test_slots_false_classes_load(self):
        fast = bench_schema(accel=True)
        assert fast.load(PAYLOAD).name == 'demo'

        @s.seared(slots=False)
        class Loose(s.Seared):
            x: int = s.Int(required=True)

        assert Loose.__seared_accel__.accelerated is True
        assert Loose.load({'x': 1}).x == 1


class TestLaxMode:
    # The whole bench schema, `Bool` included: lax `bool` reads the spelling
    # table before truthiness, as seared's does (SPEC_ABI 2).
    @pytest.mark.parametrize(
        'payload',
        [
            {'name': 42, 'items': []},
            {'name': 'n', 'items': [{'x': True, 'y': 1}]},
            {'name': 'n', 'items': [], 'tags': 'notalist'},
            {'name': 'n', 'items': [{'x': 3.7, 'y': 1}]},
            {'name': 'n', 'items': [], 'flag': 'false'},
            {'name': 'n', 'items': [], 'flag': ' OFF '},
            {'name': 'n', 'items': [], 'flag': 'maybe'},
            {'name': 'n', 'items': [], 'flag': 2},
        ],
    )
    def test_matches(self, payload):
        fast = bench_schema(validate=False, accel=True)
        slow = bench_schema(validate=False, accel=False)
        assert load_outcome(fast, payload, attr='flag') == load_outcome(slow, payload, attr='flag')
        assert outcome(fast.dump, fast.load(payload)) == outcome(slow.dump, slow.load(payload))


#: (label, payload) — each fails somewhere specific, for :class:`TestErrorPaths`.
PATH_CASES = [
    ('leaf-in-a-list', {'items': [{'n': 1}, {'n': 'x'}]}),
    ('list-in-a-leaf', {'items': [{'n': 1, 'tags': ['a', 7]}]}),
    ('keyed-str-key', {'items': [], 'readers': {'r-1': {'p': 'bad'}}}),
    ('keyed-int-key', {'items': [], 'counts': {1: 'x'}}),
    ('keyed-digit-str-key', {'items': [], 'counts': {'1': 'x'}}),
    ('three-deep', {'items': [], 'mid': {'leaves': [{'n': 1}, {'n': 2, 'tags': [None]}]}}),
    ('missing-at-the-top', {}),
    ('missing-three-deep', {'items': [], 'mid': {'leaves': [{}]}}),
    ('missing-by-its-data-key', {'items': [], 'readers': {'a': {'port': 1}}}),
    ('non-dict-at-the-top', ['x']),
    ('non-dict-nested', {'items': [7]}),
    ('many-not-a-list', {'items': 'nope'}),
    ('keyed-not-a-dict', {'items': [], 'readers': [1]}),
]

#: The cases whose path ends at a value that is there to retrieve.
RETRIEVABLE = [
    'leaf-in-a-list',
    'list-in-a-leaf',
    'keyed-str-key',
    'keyed-int-key',
    'keyed-digit-str-key',
    'three-deep',
]


class TestErrorPaths:
    """Pin the ``SPEC_ABI`` 2 error contract on ``load``: the same reason, path and class as seared.

    The path holds wire keys, list indices and keyed-dict keys (the key
    object itself), and ``cls_name`` is the outermost class.
    """

    @pytest.fixture(scope='class')
    def roots(self):
        return path_schema(accel=True), path_schema(accel=False)

    @pytest.mark.parametrize(('label', 'payload'), PATH_CASES, ids=[c[0] for c in PATH_CASES])
    def test_parts_match(self, roots, label, payload):
        fast, slow = roots
        got, want = outcome(fast.load, payload), outcome(slow.load, payload)
        assert want[:2] == ('raised', 'ValidationError'), label
        assert got == want

    @pytest.mark.parametrize(('label', 'payload'), [c for c in PATH_CASES if c[0] in RETRIEVABLE], ids=RETRIEVABLE)
    def test_the_path_retrieves_the_offending_value(self, roots, label, payload):
        fast, _ = roots
        with pytest.raises(s.ValidationError) as info:
            fast.load(payload)
        assert info.value.cls_name == 'Root'
        offending = functools.reduce(operator.getitem, info.value.path, payload)
        assert offending in ('x', 7, 'bad', None)

    def test_an_int_key_and_a_str_key_are_different_paths(self, roots):
        fast, _ = roots
        paths = []
        for key in (1, '1'):
            with pytest.raises(s.ValidationError) as info:
                fast.load({'items': [], 'counts': {key: 'x'}})
            paths.append(info.value.path)
        assert paths == [('counts', 1), ('counts', '1')]

    def test_a_missing_required_key_ends_the_path_at_its_wire_key(self, roots):
        fast, _ = roots
        with pytest.raises(s.ValidationError) as info:
            fast.load({'items': [], 'readers': {'a': {'port': 1}}})
        assert str(info.value) == 'Root.readers.a.p: required key missing'

    def test_a_non_dict_names_the_class_alone(self, roots):
        fast, _ = roots
        with pytest.raises(s.ValidationError) as info:
            fast.load(['x'])
        assert (str(info.value), info.value.path) == ('Root: expected dict, got list', ())


#: What ``load`` gives for each wire state: seared's null table (``SPEC_ABI`` 2).
NULL_TABLE = {
    #          absent                   null                     value
    'req': ('required key missing', 'required key missing', 7),
    'dflt': (5, 5, 7),
    'bare': (None, None, 7),
    'n_dflt': (5, None, 7),
    'n_req': ('required key missing', None, 7),
}

_ABSENT = object()


def _loaded(cls, column, wire):
    """``column``'s load outcome with ``wire`` as its value (or key absent).

    A nested instance is compared through its dump: the two paths' ``Leaf``
    classes are distinct, so their instances are never equal.
    """
    payload = {k: v for k, v in NULL_BASE.items() if k != column}
    if wire is not _ABSENT:
        payload[column] = wire
    got = load_outcome(cls, payload, attr=column)
    if got[0] == 'ok' and isinstance(got[1], s.Seared):
        return ('ok', type(got[1]).dump(got[1]), got[2])
    return got


@pytest.mark.parametrize('validate', [True, False], ids=['strict', 'lax'])
class TestNullRule:
    """Pin: a ``null`` is a value only on a nullable field; anywhere else it reads as an absent key."""

    @pytest.mark.parametrize('column', list(NULL_COLUMNS))
    def test_every_wire_state_matches(self, validate, column):
        fast, slow = null_schema(validate=validate, accel=True), null_schema(validate=validate, accel=False)
        for wire in (_ABSENT, None, NULL_COLUMNS[column]):
            assert _loaded(fast, column, wire) == _loaded(slow, column, wire), (column, wire)

    @pytest.mark.parametrize('column', list(NULL_TABLE))
    def test_the_null_table(self, validate, column):
        fast = null_schema(validate=validate, accel=True)
        for wire, want in zip((_ABSENT, None, 7), NULL_TABLE[column], strict=True):
            got = _loaded(fast, column, wire)
            assert (got[3] if got[0] == 'raised' else got[1]) == want, (column, wire)

    def test_a_nullable_container_or_class_can_be_null(self, validate):
        fast = null_schema(validate=validate, accel=True)
        got = fast.load({**NULL_BASE, 'n_many': None, 'n_keyed': None, 'n_leaf': None})
        assert (got.n_many, got.n_keyed, got.n_leaf) == (None, None, None)

    def test_a_null_takes_the_default_factory(self, validate):
        fast = null_schema(validate=validate, accel=True)
        assert fast.load({**NULL_BASE, 'factory': None}).factory == []

    def test_a_null_on_a_required_field_names_its_wire_key(self, validate):
        fast = null_schema(validate=validate, accel=True)
        with pytest.raises(s.ValidationError) as info:
            fast.load({**NULL_BASE, 'req': None})
        assert (str(info.value), info.value.path) == ('Row.req: required key missing', ('req',))

    def test_a_null_element_still_goes_through_the_field(self, validate):
        # ``nullable`` is about the field's own value, not its elements.
        fast, slow = null_schema(validate=validate, accel=True), null_schema(validate=validate, accel=False)
        payload = {**NULL_BASE, 'n_many': [1, None]}
        got, want = outcome(fast.load, payload), outcome(slow.load, payload)
        assert got == want
        assert got[4] == ('n_many', 1)
