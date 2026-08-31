# rusted

`rusted` is an **optional compiled accelerator core for
[`seared`](https://www.github.com/bwiswell/seared)**. Install it and
`@s.seared` classes get compiled `load` / `dump`; don't, and seared behaves
exactly as it always has. No code changes either way.

```sh
uv add git+https://www.github.com/bwiswell/rusted.git
```

That is the entire user-facing API. There is nothing to import and nothing to
call — seared finds it.

**A git install builds from source and needs a Rust toolchain** (`rustup`,
1.85+). That is the current state, not the intent: the point of an accelerator
is that it arrives as a prebuilt wheel, and a git source can't serve one. Until
it is published to an index, treat this as developer-only — and note that a
failed *install* is the one failure mode this design can't turn into a graceful
fallback, since seared can only degrade around a wheel that is absent, not one
that is broken.

## What it buys

Measured by seared's own bench harness (20k iterations, one process, Python
3.14.3, Linux x86_64), from the run recorded in seared's
`bench/results.json` — the same artifact seared's own docs quote, so the two
repos cannot drift apart on this:

| op   | seared (pure Python) | with `rusted` | speedup |
|------|---------------------:|--------------:|--------:|
| load | 32.7 µs              | **2.9 µs**    | ~11.2×  |
| dump | 22.5 µs              | **2.4 µs**    | ~9.4×   |

Run-to-run spread on this hardware is roughly ±10%; the ratios are the
durable claim, the absolutes are one sample.

On `load`, validation becomes effectively free — strict and lax land within
noise of each other, because the guards turn into C-level type checks
instead of `isinstance` calls. On `dump` strict still costs ~15%, in *both*
implementations: there the guards are most of what the pass does, so making
them cheaper doesn't make them disappear.

The remaining cost is dominated by Python object construction, which no
compiled core can remove; a pure-Rust round trip of the same payload runs
~2.0 µs, so there is no second order of magnitude waiting behind this one.

## What gets accelerated

Per class, **all or nothing**. A class is accelerated only if every field —
recursively, through `T` — is one this build implements:

- **Tier 1** — `Int`, `Float`, `Str`, `Bool`, `T`
- **Tier 2** — `UUID`, `Date`, `DateTime`, `Time`, `TimeDelta`, `Decimal`,
  `Bytes`, `Enum`, `Path`, `Dict`

with `many`, `keyed`, `required`, `default`, `default_factory`, `data_key`
and `dump=False` on any of them.

Still on the Python path, each for a reason: `Union` (an UNWRAP field
consuming several keys from its *parent's* map), `Tuple` (per-slot
sub-fields), `NDArray` / `PandasFrame` / `PolarsFrame` (optional imports, and
dominated by frame conversion anyway), user-defined `Field` subclasses (they
can override `serialize` in Python, so exact type identity is what gates
acceleration), and any class with a hand-written `__init__` or a
`__post_init__` (the compiled core builds instances via `__new__`, which
would skip them).

Tier 2 does not run 10× — those kinds spend their time constructing Python
objects (`uuid.UUID`, `datetime.strptime`, `Decimal`), which is work no
compiled core removes. On a realistic mixed message class it is ~2.6× load /
~2.1× dump. The point of Tier 2 is that **all-or-nothing cuts both ways**: a
single `Bytes` field used to disqualify an entire class, Tier 1 fields
included. Now it doesn't.

Ask seared what happened:

```python
import seared as s

s.accel_status()  # backend loaded? which one? if not, why not
MyClass.__seared_accel__  # AccelInfo(accelerated=..., backend=..., reason=...)
```

`reason` names the field and type that blocked a class, so "why isn't mine
fast?" has an answer.

Two environment variables: `SEARED_ACCEL` (`auto` / `off` / `require`) and
`SEARED_ACCEL_BACKEND` (the module to import instead of `rusted`). `require`
raises if no backend loads — it exists so CI can assert the wheel is actually
engaged rather than silently falling back.

## How it fits

seared stays canonical, pure Python and zero-dependency. It owns all knowledge
of itself: its accelerator seam walks `__seared_fields__` and emits a
plain-data spec.

`rusted` owns the compiled interpreter and knows nothing about seared — it
imports nothing from it, not even the exception class, which rides in the
spec so that a caller's `except s.ValidationError` keeps catching errors
raised from Rust. That is what keeps the family's dependency arrow pointing
one way, and it means `rusted` can only ever be a no-op: a missing wheel, an
ABI mismatch, or a backend that raises all end in the Python path.

Compatibility is one integer. `rusted.SPEC_ABI` must equal seared's or seared
declines the backend outright; `SUPPORTS_SEARED` is diagnostic text, never
enforced (seared is zero-dependency and has no PEP 440 parser to enforce it
with).

## Development

```sh
uv sync                      # builds the extension into .venv
uv run pytest                # the suite, including the differential tests
uv run prek install          # ruff + ty + deptry + cargo fmt/clippy on commit
```

After editing Rust, rebuild with `uv sync --reinstall-package rusted` (or
`uv run maturin develop --release`). Release profile matters: `BUILD_PROFILE`
is asserted in the tests, because debug-build numbers are noise.

**The differential suite is the point of this repo.** Two implementations of
one semantics stay honest exactly as long as something mechanically proves
they agree, so every test builds the same schema twice — once accelerated,
once with `accel=False` — and asserts identical values, identical exception
classes and identical message text. seared's own suite is the second gate:

```sh
# from the seared checkout, with rusted installed
SEARED_ACCEL=require SEARED_ACCEL_BACKEND=rusted uv run pytest
```

If that diverges from a plain run, the accelerator is wrong, not seared.

## Roadmap

Deferred, each for a reason rather than for lack of time:

- **`Union` / `Tuple`.** `Union` is an UNWRAP field — it consumes several keys
  from its *parent's* map and merges its output back at the parent's level,
  which the single-pass interpreter here is not shaped for. `Tuple` carries
  per-slot sub-fields, a second dimension of nesting.
- **`NDArray` / `PandasFrame` / `PolarsFrame`.** Optional imports crossing the
  boundary, for workloads whose cost is the frame conversion either way.
- **A fused bytes path** (parse JSON or msgpack straight into instances,
  skipping the intermediate dict). Prototyped at ~7.5× end-to-end, but it
  cannot be a silent swap: a streaming parser cannot reproduce seared's
  shape-mismatch messages, nor its lax `str(list)` coercion, and serde rejects
  the `NaN` / `Infinity` that `json.dumps` writes. It would have to be an
  explicitly-named opt-in surface, not a transparent replacement.
- **Free-threaded (3.14t) wheels.** A separate ABI from abi3; the pure-Python
  fallback already covers those interpreters correctly.

Not deferred, and not planned: hybrid per-field acceleration. A Rust loop
calling back into Python for one exotic field would cost most of the win on a
mixed class and double the semantics under test.
