# Changelog

Each version of rusted, newest first.

A **Supported seared** line means `SUPPORTS_SEARED`, the seared range a build declares, has moved. seared never
enforces that range. The one check it makes is `SPEC_ABI`, 1 through 0.3.2 and 2 from 0.4.0: seared uses a build
only when the two match, and within that, outside a build's declared range, the two paths may differ.

The top entry, while it reads *unreleased*, collects what has merged for the next version; each change adds its own
lines as it lands, and the entry is dated when that version is released.

## 0.4.0 — 2026-10-16

**`SPEC_ABI` 2, for seared 0.4.0: errors that say where they happened, `null` read as absent, and lossless strict
coercion.**

### Added

- **`CHANGELOG.md`**, this file: every release from 0.1.0 on, newest first. It is kept in the repository and grows
  with each change as it merges, where it had been reconstructed after each release.
- **Differential cases for where an error happened:** paths through `T`, `many` and `keyed`, nested three deep; an
  int key against a string one; a `data_key` named by its wire key on `load` and by its attribute on `dump`; and
  errors from a compiled variant crossing back through a `Union` parent, which always runs on the Python path, in
  both envelope shapes.
- **Differential cases for the null rule:** seared's whole null table, with `nullable` on and off, in both modes;
  nullable `many`, `keyed` and `T` fields; a `null` element; a `default_factory`; what `dump` writes, an unset slot
  included; and `load(dump(x))` over every arrangement of `None`s.
- **Differential cases for coercion:** `int` with integral, fractional and infinite floats, `nan` and numeric
  strings, a float subclass that overrides `is_integer`, `bool` with every small int, and an int-valued enum with
  floats, bools, `inf` and a `Decimal`, in both modes. Lax `bool` rejoins every sweep, and the lax differential
  cases run over the whole bench schema again, its `Bool` included.

### Changed

- **`SPEC_ABI` 2,** the spec seared 0.4.0 emits. seared 0.4.0 declines rusted 0.3.x and seared 0.3.x declines this
  build, so the two releases go together.
- **Errors say where they happened,** with seared 0.4.0's parts: `.reason`, `.path` and `.cls_name`, and so the
  same message, `Outer.items[1].n: cannot deserialize 'x' as int`. On `load` the path holds wire keys, list
  indices and keyed-dict keys (the key object itself); on `dump` it holds attribute names; `cls_name` is the
  outermost class. Each level rebuilds the error through its class's constructor as it passes back up, so a
  successful `load` or `dump` pays nothing for it. Two messages are reworded with seared's:
  - a required key reads `Outer.n: required key missing`, named by its wire key (was `Outer.n is required`);
  - a non-dict reads `Outer: expected dict, got list` (was `Outer.load expected dict, got list`).
- **A `null` reads as absent unless the field is nullable,** as seared 0.4.0 has it. A `null` on a required field
  raises `required key missing`, and on a defaulted one gives the default, where it used to load as `None` either
  way. A field the spec marks `nullable` keeps a `null` as `None`, even over a default, and `dump` writes its
  `None`; every other `None` is still left out. An unset slot on a nullable field dumps as `None` too, since seared
  reads it as one. A `null` element inside `many` or `keyed` still goes through the field.
- **Strict coercion no longer loses data,** as in seared 0.4.0. The messages are unchanged; which inputs are refused
  changes.
  - Strict `int` refuses a float with a fractional part (`3.7`), which it used to truncate. Integral floats and
    numeric strings still load, and lax mode still truncates. The float and string path is shared with
    int-valued enums, as seared shares it.
  - Strict `bool` takes only the ints `0` and `1`; `2` used to load as `True`.
  - A strict int-valued enum follows strict `int`: `True` and `2.9` are refused, where they found the members for
    `1` and `2`. Anything but a bool, an int, a float or a string is refused in both modes, so `Decimal('1')` no
    longer finds its member either.
- **Lax `bool` reads the spelling table before truthiness,** so `'false'`, `'0'`, `'no'` and `'off'` load as
  `False`, as seared has since 0.3.3. Lax classes with a `Bool`, which seared kept off this core until `SPEC_ABI` 2,
  accelerate again.
- **Supported seared:** `>=0.4,<0.5`, from `>=0.3.4,<0.4`. The floor is the first seared to emit `SPEC_ABI` 2.
- **Dependencies:** seared, the test-only dependency, follows seared's `version/0.4.0` branch (`>=0.4,<0.5`), with
  the lockfile pinning a commit, until seared 0.4.0 is tagged. The dev group's `pytest` range lifts to
  `>=8.0,<10.0`, and the refreshed lockfile resolves pytest 9.1.1, ruff 0.17.1 and ty 0.0.89.
- **Built with pyo3 0.30 and Rust 1.99.1,** from pyo3 0.26 and Rust 1.98.0. Speed is unchanged: timed against a
  0.26 build of the same code, every `load` and `dump` figure moved by less than the spread between runs. The wheel
  is still `abi3-py314`, building from source still needs Rust 1.85 or later, and pyo3 0.30 needs five fewer
  crates.
- The README links seared's generated benchmark tables for the numbers, where it carried a hand-copied table that
  had drifted, installs by the `v0.4.0` tag, and names two more reasons a class stays on the Python path: a plain
  dataclass attribute, and an enum with no members.

### Fixed

- **`inf` into an `int`, or an int-valued enum, raises `ValidationError`** in both modes. It let `int()`'s
  `OverflowError` through, which `except s.ValidationError` didn't catch.

## 0.3.2 — 2026-10-16

**Tagged releases, and seared pinned to one.** No change to the compiled core or to `SPEC_ABI`.

### Added

- **Releases are tagged.** A `Release Tag` workflow (`.github/workflows/release-tag.yml`) runs when a version merges
  to `master`. It checks that `pyproject.toml`, `Cargo.toml` and `python/rusted/__init__.py` stamp the same version,
  then tags the merge `v<version>` and publishes a GitHub release; a version already released is skipped. 0.3.2 is
  the first tagged release, and earlier ones stay untagged.
- **A test that the three version stamps agree**, read from the tree rather than the installed build, so a
  mismatched bump fails before it reaches `master`.
- **A `no-commit-to-branch` hook** refuses a commit on `master`, which now moves only by merge. It is prek's
  built-in copy, so it fetches nothing.

### Changed

- **The README installs a release by its tag**, `uv add git+https://github.com/bwiswell/rusted --tag v0.3.2`, and
  shows the `[tool.uv.sources]` line that records. It says why: uv doesn't check a version specifier against a git
  source, so a source without a tag follows the tip of `master`. A git install still builds from source and needs a
  Rust toolchain.
- **README Development describes the release process.** `feat/` and `fix/` branches merge into a `version/` branch,
  one `chore: version bump` commit there bumps all three stamps, and the version branch merges into `master`, where
  it is tagged.
- **Supported seared:** `>=0.3.4,<0.4`, from `>=0.3.3,<0.4`. The suite now runs against seared 0.3.4, pinned by its
  tag. 0.3.3 matches it for everything the spec carries, but it isn't what's tested.
- **Dependencies:** seared, the test-only dependency, is pinned to the `v0.3.4` tag (`>=0.3.4,<0.4`) and locked to
  it. Its source used to follow seared's `master`, so an upgrade lock could pull in any later seared, 0.4.0
  included. Against 0.3.4, the full suite passes, differential tests included, and so does seared's own suite with
  this core required.

## 0.3.1 — 2026-10-15

**Parity with seared 0.3.3.** seared 0.3.3 fixed lax `Bool` in Python and keeps lax classes with one off the
accelerator until `SPEC_ABI` 2; this release raises the supported floor to it and makes the suite honest about it.
The compiled core is unchanged.

### Added

- **The test builders check the fast side.** Every builder asserts that a class it was asked to accelerate really
  did, as the Tier 2 helper already did. Against seared 0.3.3, 102 parity cases had been comparing Python with
  Python and passing: lax classes with a `Bool`, which seared now declines. A decline seared adds now fails loudly
  instead.
- **Tripwires for the decline:**
  - a lax class with a `Bool` declines with seared's reason, and so does a strict class nesting one;
  - strict `Bool` and lax classes without one still accelerate;
  - lax `bool` here still gives the ABI 1 answer (`'false'` is `True`), checked through a hand-built spec. Changing
    it under ABI 1 would make the two paths disagree silently under seared 0.3.1 and 0.3.2.

  All of them fail when `SPEC_ABI` 2 lifts the decline, which is when the lax `Bool` fix lands here.

### Changed

- **Supported seared:** `>=0.3.3,<0.4`, from `>=0.3.1,<0.4`. The floor is now the seared this build is tested
  against: under 0.3.3 lax `bool` never reaches this core, so it goes untested, and older seareds are outside the
  tested range.
- **Lax coverage without `Bool`:** the lax half of the differential suite and `TestLaxMode` run over the bench
  schema minus its `Bool`, and the scalar and container sweeps take lax over `int`, `float` and `str`. `bool`
  rejoins them with `SPEC_ABI` 2.
- The README lists lax classes with a `Bool` among those that keep the Python path, until the next `SPEC_ABI`.
- **Dependencies:** seared, the test-only dependency, is pinned to `>=0.3.3,<0.4`, and the lockfile resolves 0.3.3.
  It is the suite's first run against seared 0.3.2 or 0.3.3, and nothing else diverged.

## 0.3.0 — 2026-09-04

**Tests laid out like the source**, and seared taken from GitHub. The compiled core is unchanged.

### Changed

- **The tests mirror the source:** a `test_*.py` per Rust source file (`tests/kinds/test_scalar.py` for
  `src/kinds/scalar.rs`, and so on), with `tests/test_differential.py` for the contract as a whole.
  `tests/test_kinds.py` is split into `tests/kinds/` (`test_mod`, `test_scalar`, `test_temporal`, `test_value`), and
  `src/imports.rs` gains `tests/test_imports.py`. The README says so.
- `ruff` leaves Markdown unformatted, as in seared and zeared, and ignores `FLY`.
- **Dependencies:** seared, a test-only dependency, comes from its GitHub remote rather than the sibling checkout,
  as zeared takes it, pinned to `>=0.3.1,<0.4`, the range `SUPPORTS_SEARED` declares. To test a seam change before
  pushing it, point `[tool.uv.sources]` back at `../seared` locally. The lockfile resolves seared 0.3.1.

## 0.2.1 — 2026-09-04

**Parity with seared 0.3.1.** seared 0.3.1 caught up with two things this core already did; this release raises the
supported floor to it and tests both. The compiled core is unchanged.

### Added

- **Tests for both:** a schema with a `Bytes` inside a nested class, dumped and loaded under `json` and `msgpack` on
  both paths, and a check that a class with a plain dataclass attribute is declined and still loads correctly on the
  Python path. The bench schema is Tier 1 and ignores the carrier hint, so the differential suite couldn't see the
  first.

### Changed

- **Supported seared:** `>=0.3.1,<0.4`, from `>=0.2.8,<0.4`. The floor is now the seared this build's parity holds
  against. Under an older seared, the two paths differ in two places:
  - `format=` through a nested `T`: this core passes the carrier hint into nested classes, so under
    `format='msgpack'` a nested `Bytes` goes as native bytes, where seared before 0.3.1 dropped the hint and sent
    base64;
  - a class with a plain dataclass attribute (`b: int = 5`): this core builds instances through `__new__`, which
    leaves that slot unset, so `repr` and `==` raise. seared 0.3.1 declines such classes.
- **Dependencies:** the lockfile resolves seared 0.3.1.

## 0.2.0 — 2026-08-31

**Supported seared 0.3.x**: the release seared 0.3.0 treats as shipped. The compiled core is unchanged.

### Changed

- **Supported seared:** `>=0.2.8,<0.4`, from `>=0.2.8,<0.3`. The floor is the seared that introduced the seam and
  the spec keys this build reads; the ceiling is a guess at where the spec might next change, not a tested boundary.
- **The README's numbers come from seared's `bench/results.json`**, the run seared's own docs quote, so the two
  repos can't drift apart: load is about 11.4× (34.6 µs to 3.0 µs) and dump about 9.1× (22.3 µs to 2.4 µs) faster
  than pure seared. One run is one sample. Repeated runs on an idle machine land within a few percent, but
  background load spreads them by up to ±20%, and load was seen anywhere from 10.0× to 11.8×; read them as about 11×
  and 9×.
- **Validation's cost, stated precisely:** on `load`, strict and lax land within noise of each other, since the
  guards become C-level type checks. On `dump`, strict still costs about 15% in both implementations, because there
  the guards are most of the work.
- **Installing from git builds from source**, and the README says so: it needs a Rust toolchain (`rustup`, 1.85 or
  later). Until rusted is published to an index, treat it as for developers only. A failed install is the one
  failure seared can't fall back around: it degrades around a missing wheel, not a broken one.
- **Dependencies:** the lockfile resolves seared 0.3.0.

## 0.1.2 — 2026-08-31

**A roadmap**, and docs that stand on their own.

### Added

- **The README's roadmap**, each item deferred for a reason:
  - `Union` and `Tuple`: `Union` consumes several of its parent's keys and merges its output back at the parent's
    level, which a single-pass interpreter isn't shaped for, and `Tuple`'s per-slot fields add a second dimension of
    nesting;
  - `NDArray`, `PandasFrame` and `PolarsFrame`: optional imports crossing the boundary, for work whose cost is the
    frame conversion either way;
  - a fused bytes path, parsing JSON or msgpack straight into instances. It was prototyped at about 7.5× end to end,
    but it can't be a silent swap: a streaming parser can't reproduce seared's shape-mismatch messages or its lax
    `str(list)` coercion, and serde rejects the `NaN` and `Infinity` that `json.dumps` writes. It would have to be
    an opt-in surface with its own name;
  - free-threaded (3.14t) wheels, a separate ABI from abi3. The pure-Python path already covers those interpreters.

  Not planned at all: per-field hybrid acceleration. A Rust loop calling back into Python for one exotic field would
  cost most of the gain on a mixed class, and double the semantics under test.

### Changed

- **No pointers to the design notes:** the README and `src/lib.rs` pointed at the workspace's plan doc, and now say
  what they meant. `src/lib.rs` describes the crate's layout: `spec` ingests the spec, `kinds` holds the coercions,
  `load` and `dump` walk a schema, and `imports` caches the stdlib types.
- `tests/test_differential.py` says what it covers (coercions, wrong types, container shape mismatches, nulls,
  absent keys) and where it is going: every supported kind, crossed with every flag, in strict and lax mode,
  generated rather than written by hand.
- **Dependencies:** the lockfile resolves seared 0.2.9.

## 0.1.1 — 2026-08-31

**Tier 2.** Ten more field types accelerate, so one `Bytes` field no longer keeps a whole class on the Python path.

### Added

- **The Tier 2 kinds:** `UUID`, `Date`, `DateTime`, `Time`, `TimeDelta`, `Decimal`, `Bytes`, `Enum`, `Path` and
  `Dict`, with `many`, `keyed`, `required`, `default`, `default_factory`, `data_key` and `dump=False` on any of
  them.
  - What the user chose for a field arrives in the spec: an enum class, a concrete path type, a strftime `format=`,
    `as_number=`, the bytes encoding. The stdlib types the kinds parse into are imported once and cached.
  - `Bytes` is the only kind that reads the carrier hint: native bytes under `format='msgpack'`, hex or base64 text
    otherwise.
  - An `Enum` with no members is declined, so seared's own path raises its `StopIteration` at the moment it always
    did.
- **A mixed-class differential suite:** one class across Tier 1 and Tier 2, which must accelerate whole, load and
  round-trip as the Python path does, and switch its `Bytes` under msgpack.
- **Speed:** Tier 2 doesn't reach Tier 1's 10×, since its kinds spend their time building Python objects
  (`uuid.UUID`, `datetime.strptime`, `Decimal`) that no compiled core can avoid. On a realistic mixed message class
  it is about 2.6× on load and 2.1× on dump.

### Changed

- `src/kinds.rs` becomes `src/kinds/`: `scalar.rs` for Tier 1, `temporal.rs` for the date-likes, `value.rs` for the
  rest.
- The README says why each remaining type stays on the Python path: `Union` consumes several of its parent's keys;
  `Tuple` has per-slot sub-fields; the frame types are optional imports, dominated by conversion anyway; a
  user-defined `Field` subclass can override `serialize` in Python, which is why exact type identity gates
  acceleration; and `__new__` would skip a class's own `__init__` or `__post_init__`.

## 0.1.0 — 2026-08-31

**The compiled core.** Install it and `@s.seared` classes get compiled `load` and `dump`; don't, and seared behaves
as it always has. There is nothing to import and nothing to call: seared finds it.

### Added

- **The backend protocol** seared 0.2.8 looks for: `SPEC_ABI` (1) and `compile_spec(spec)`, which returns a
  `(load, dump)` pair, or `None` to decline a kind this build doesn't implement. A malformed spec, or one on another
  ABI, raises, and seared gives the error as the class's reason.
- **Tier 1:** `Int`, `Float`, `Str`, `Bool` and nested `T`, with `many`, `keyed`, `required`, `default`,
  `default_factory`, `data_key` and `dump=False`, in strict and lax mode. Each coercion is a transcription of
  seared's field method, message text included. Any other field, a user-defined `Field` subclass, or a class with
  its own `__init__` or `__post_init__` keeps the Python path. Acceleration is all or nothing per class, recursively
  through `T`.
- **A leaf that knows nothing of seared:** it imports nothing from it, not even `ValidationError`, which arrives in
  the spec so that `except s.ValidationError` still catches errors raised from Rust. Instances are built with
  `__new__` and slot assignment, as pydantic-core builds them.
- **`SUPPORTS_SEARED`** (`>=0.2.8,<0.3`), for diagnosis only: seared has no PEP 440 parser and gates on `SPEC_ABI`
  alone. **`BUILD_PROFILE`**, which the tests assert is `release`, since a debug build's numbers are noise.
- **The differential suite:** each test builds a schema twice, accelerated and with `accel=False`, and asserts the
  same values, the same exception classes and the same message text. seared's own suite is the second gate, run with
  `SEARED_ACCEL=require SEARED_ACCEL_BACKEND=rusted`.
- **Speed**, on seared's bench schema: load about 10.7× (32.4 µs to 3.0 µs) and dump about 8.7× (21.2 µs to 2.4 µs)
  faster than pure seared. What remains is mostly building Python objects; a pure-Rust round trip of the same
  payload takes about 2.0 µs, so no second order of magnitude waits behind this one.
- **The build:**
  - PyO3 0.26 with `abi3-py314`, so one wheel per platform serves Python 3.14 and later;
  - `maturin` as the build backend, with the extension at `rusted._rusted`, a `.pyi` stub and a `py.typed` marker;
  - LTO and one codegen unit in release, and no `panic = "abort"`, so a Rust panic becomes a Python exception
    instead of taking down the interpreter;
  - the toolchain pinned at Rust 1.98.0, while building from source needs 1.85 or later.
- **Tooling:** `prek` hooks for `ruff`, `ty`, `deptry`, `cargo fmt` and `cargo clippy`, with Clippy's warnings as
  errors. seared is a dev dependency only, from the sibling checkout.
