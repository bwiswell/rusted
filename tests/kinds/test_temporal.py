"""Tests for ``src/kinds/temporal.rs`` — ``date`` / ``datetime`` / ``time`` / ``timedelta``.

The generic sweep runs the same wrong-type values Tier 1 sees: which
branches seared gates on ``validate`` and which raise regardless is not
uniform across these fields, and each has to be matched individually. The
native sweep runs the wire forms each kind is actually meant to accept.
"""

from __future__ import annotations

import pytest
from conftest import KIND_VALUES, MODES, TEMPORAL, VALUE_IDS, VALUES, both, load_outcome, outcome, raw, same


@pytest.mark.parametrize('kind', TEMPORAL)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestGenericSweep:
    @pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
    def test_load(self, kind, validate, mode, value):
        fast, slow = both(kind, validate=validate)
        assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value}))

    @pytest.mark.parametrize('value', VALUES, ids=VALUE_IDS)
    def test_dump(self, kind, validate, mode, value):
        fast, slow = both(kind, validate=validate)
        assert same(outcome(fast.dump, raw(fast, v=value)), outcome(slow.dump, raw(slow, v=value)))


@pytest.mark.parametrize('kind', TEMPORAL)
@pytest.mark.parametrize(('validate', 'mode'), MODES, ids=[m[1] for m in MODES])
class TestNativeValues:
    def test_load(self, kind, validate, mode):
        fast, slow = both(kind, validate=validate)
        for value in KIND_VALUES[kind]:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value

    def test_round_trip(self, kind, validate, mode):
        fast, slow = both(kind, validate=validate)
        for value in KIND_VALUES[kind]:
            a, b = outcome(fast.load, {'v': value}), outcome(slow.load, {'v': value})
            if a[0] == 'raised':
                assert a == b, value
                continue
            assert same(outcome(fast.dump, a[1]), outcome(slow.dump, b[1])), value

    def test_many(self, kind, validate, mode):
        fast, slow = both(kind, validate=validate, many=True)
        for value in ([], list(KIND_VALUES[kind]), 'nope', None, 7):
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value


class TestStrftimeFormat:
    # A `format=` switches the three date-likes from isoformat/fromisoformat
    # to strftime/strptime — always via `datetime.strptime`, then narrowed.
    @pytest.mark.parametrize(
        ('kind', 'fmt', 'values'),
        [
            ('date', '%d/%m/%Y', ['31/08/2026', '2026-08-31', 'nope']),
            ('datetime', '%Y/%m/%d %H:%M', ['2026/08/31 12:30', 'nope']),
            ('time', '%H:%M', ['12:30', '12:30:45', 'nope']),
        ],
    )
    def test_matches(self, kind, fmt, values):
        fast, slow = both(kind, format=fmt)
        for value in values:
            assert same(load_outcome(fast, {'v': value}), load_outcome(slow, {'v': value})), value
        a, b = fast.load({'v': values[0]}), slow.load({'v': values[0]})
        assert fast.dump(a) == slow.dump(b)
