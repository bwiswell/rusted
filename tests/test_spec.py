"""Tests for ``src/spec.rs`` — spec ingestion.

Two failure modes must stay distinct: *declining* a kind this build doesn't
implement (return ``None``, seared keeps the Python path) versus *erroring*
on a malformed spec (a bug, which seared surfaces in the decline reason).
"""

from __future__ import annotations

import enum
import pathlib

import pytest
import rusted
import seared as s

FIELD = {
    'attr': 'a',
    'wire': 'a',
    'kind': 'int',
    'required': False,
    'many': False,
    'keyed': False,
    'dump': True,
    'default': None,
    'default_factory': None,
}


class Plain:
    __slots__ = ('a', 'b')


def spec(**overrides):
    base = {
        'abi': rusted.SPEC_ABI,
        'cls': Plain,
        'name': 'Plain',
        'validate': True,
        'error': s.ValidationError,
        'fields': [dict(FIELD)],
    }
    base.update(overrides)
    return base


#: A kind name that will never be implemented. Deliberately not a real-but-
#: unsupported field type: `decimal` used to serve here, and these tests broke
#: the day Tier 2 landed. The stand-in for "unknown" has to stay unknown.
UNKNOWN_KIND = 'never-a-real-kind'


class TestDeclines:
    def test_unknown_kind_declines(self):
        # Not an error: a newer seared may emit kinds this build predates.
        assert rusted.compile_spec(spec(fields=[{**FIELD, 'kind': UNKNOWN_KIND}])) is None

    def test_unknown_nested_kind_declines_the_parent(self):
        # Acceleration is per-class all-or-nothing, recursively.
        nested = {
            **FIELD,
            'attr': 'b',
            'wire': 'b',
            'kind': 'nested',
            'schema': spec(fields=[{**FIELD, 'kind': UNKNOWN_KIND}]),
        }
        assert rusted.compile_spec(spec(fields=[nested])) is None

    def test_empty_enum_declines(self):
        # seared reads `next(iter(enum)).value` inside deserialize, so an empty
        # enum raises StopIteration there. Declining keeps that happening at
        # exactly the moment it always did, rather than at decoration time.
        empty = enum.Enum('Empty', {})
        field = {**FIELD, 'kind': 'enum', 'enum': empty}
        assert rusted.compile_spec(spec(fields=[field])) is None


class TestKindConfig:
    """Per-kind configuration the spec must carry, and what happens without it."""

    @pytest.mark.parametrize(
        ('kind', 'config'),
        [
            ('bytes', {'encoding': 'hex'}),
            ('date', {'format': None}),
            ('datetime', {'format': '%Y'}),
            ('time', {'format': None}),
            ('decimal', {'as_number': False}),
            ('path', {'concrete': pathlib.PurePosixPath}),
        ],
    )
    def test_config_is_required(self, kind, config):
        key = next(iter(config))
        assert rusted.compile_spec(spec(fields=[{**FIELD, 'kind': kind, **config}])) is not None
        with pytest.raises(KeyError, match=key):
            rusted.compile_spec(spec(fields=[{**FIELD, 'kind': kind}]))

    def test_enum_class_is_required(self):
        with pytest.raises(KeyError, match='enum'):
            rusted.compile_spec(spec(fields=[{**FIELD, 'kind': 'enum'}]))

    def test_enum_must_be_a_class(self):
        with pytest.raises(TypeError, match="needs an 'enum' class"):
            rusted.compile_spec(spec(fields=[{**FIELD, 'kind': 'enum', 'enum': 'nope'}]))

    def test_path_concrete_must_be_a_class(self):
        with pytest.raises(TypeError, match="needs a 'concrete' class"):
            rusted.compile_spec(spec(fields=[{**FIELD, 'kind': 'path', 'concrete': 'nope'}]))

    @pytest.mark.parametrize('kind', ['uuid', 'timedelta', 'dict'])
    def test_config_free_kinds(self, kind):
        assert rusted.compile_spec(spec(fields=[{**FIELD, 'kind': kind}])) is not None


class TestMalformed:
    def test_abi_mismatch_raises(self):
        with pytest.raises(ValueError, match='SPEC_ABI'):
            rusted.compile_spec(spec(abi=999))

    @pytest.mark.parametrize('key', ['abi', 'cls', 'name', 'validate', 'error', 'fields'])
    def test_missing_top_level_key_raises(self, key):
        bad = spec()
        del bad[key]
        with pytest.raises(KeyError, match=key):
            rusted.compile_spec(bad)

    @pytest.mark.parametrize('key', ['attr', 'wire', 'kind', 'required', 'many', 'keyed', 'dump'])
    def test_missing_field_key_raises(self, key):
        bad_field = dict(FIELD)
        del bad_field[key]
        with pytest.raises(KeyError, match=key):
            rusted.compile_spec(spec(fields=[bad_field]))

    def test_non_class_cls_raises(self):
        with pytest.raises(TypeError, match="'cls' must be a class"):
            rusted.compile_spec(spec(cls='not a class'))

    def test_non_class_error_raises(self):
        with pytest.raises(TypeError, match="'error' must be an exception class"):
            rusted.compile_spec(spec(error='nope'))

    def test_non_list_fields_raises(self):
        with pytest.raises(TypeError, match="'fields' must be a list"):
            rusted.compile_spec(spec(fields={'a': 1}))

    def test_non_dict_field_raises(self):
        with pytest.raises(TypeError, match='field spec must be a dict'):
            rusted.compile_spec(spec(fields=['nope']))

    def test_nested_without_schema_raises(self):
        with pytest.raises(KeyError, match='schema'):
            rusted.compile_spec(spec(fields=[{**FIELD, 'kind': 'nested'}]))


class TestFlagsAreHonoured:
    def test_data_key_maps_attr_to_wire(self):
        load, dump = rusted.compile_spec(spec(fields=[{**FIELD, 'wire': 'propertyA'}]))
        obj = load({'propertyA': 3})
        assert obj.a == 3
        assert dump(obj) == {'propertyA': 3}

    def test_dump_false_suppresses_output(self):
        load, dump = rusted.compile_spec(spec(fields=[{**FIELD, 'dump': False}]))
        assert dump(load({'a': 3})) == {}

    def test_default_is_used_when_absent(self):
        load, _ = rusted.compile_spec(spec(fields=[{**FIELD, 'default': 42}]))
        assert load({}).a == 42

    def test_default_factory_runs_per_load(self):
        field = {**FIELD, 'many': True, 'default_factory': list}
        load, _ = rusted.compile_spec(spec(fields=[field]))
        first, second = load({}), load({})
        first.a.append(1)
        assert second.a == []

    def test_required_missing_raises_searreds_error(self):
        load, _ = rusted.compile_spec(spec(fields=[{**FIELD, 'required': True}]))
        with pytest.raises(s.ValidationError, match=r'Plain\.a is required'):
            load({})
