"""Tests for ``src/lib.rs`` — the module surface and ``compile_spec`` entry."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import rusted
import seared as s
from seared._core import accel

_ROOT = Path(__file__).parent.parent


class TestModuleSurface:
    def test_abi_matches_seareds(self):
        # The whole compatibility gate. If these drift, seared declines the
        # backend outright rather than running a mismatched spec.
        assert rusted.SPEC_ABI == accel.SPEC_ABI

    def test_build_profile_is_release(self):
        # Debug-build numbers are noise; the bench refuses them, and a wheel
        # is always release. Catches an accidental `maturin develop` install.
        assert rusted.BUILD_PROFILE == 'release'

    def test_declares_a_diagnostic_seared_range(self):
        assert rusted.SUPPORTS_SEARED.startswith('>=')

    def test_exports(self):
        assert set(rusted.__all__) == {
            'BUILD_PROFILE',
            'SPEC_ABI',
            'SUPPORTS_SEARED',
            '__version__',
            'compile_spec',
        }

    def test_the_version_is_stamped_the_same_in_all_three_places(self):
        # The release workflow tags from these and refuses to run if they
        # differ; failing here catches it before the merge. Read from the
        # tree, not the installed module, which can be an older build.
        project = tomllib.loads((_ROOT / 'pyproject.toml').read_text())['project']['version']
        crate = tomllib.loads((_ROOT / 'Cargo.toml').read_text())['package']['version']
        init = (_ROOT / 'python' / 'rusted' / '__init__.py').read_text()
        stamped = re.search(r"^__version__ = '(.*)'$", init, re.MULTILINE)
        assert stamped is not None
        assert project == crate == stamped[1]


class TestCompileSpec:
    def test_returns_a_callable_pair(self, spec_of):
        compiled = rusted.compile_spec(spec_of)
        assert compiled is not None
        load, dump = compiled
        assert callable(load)
        assert callable(dump)

    def test_format_is_positional(self, spec_of):
        # The decorator calls `load_fn(data, format)` positionally — the
        # compiled pair must accept that shape exactly.
        load, dump = rusted.compile_spec(spec_of)
        obj = load({'a': 1}, 'msgpack')
        assert dump(obj, 'msgpack') == {'a': 1}

    def test_format_defaults_to_json(self, spec_of):
        load, _dump = rusted.compile_spec(spec_of)
        assert load({'a': 1}).a == 1

    def test_works_on_a_plain_slots_class(self, spec_of):
        # rusted never imports seared and doesn't require a @seared class —
        # it interprets a spec and constructs whatever `cls` it was handed.
        load, dump = rusted.compile_spec(spec_of)
        assert dump(load({'a': 5})) == {'a': 5}


@pytest.fixture
def plain_cls():
    class Plain:
        __slots__ = ('a',)

    return Plain


@pytest.fixture
def spec_of(plain_cls):
    return {
        'abi': rusted.SPEC_ABI,
        'cls': plain_cls,
        'name': 'Plain',
        'validate': True,
        'error': s.ValidationError,
        'fields': [
            {
                'attr': 'a',
                'wire': 'a',
                'kind': 'int',
                'required': True,
                'many': False,
                'keyed': False,
                'dump': True,
                'default': None,
                'default_factory': None,
            },
        ],
    }
