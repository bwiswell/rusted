//! rusted — the compiled accelerator core for `seared`.
//!
//! seared stays canonical and pure Python. When this wheel is installed, its
//! accelerator seam hands over a plain-data spec per class and swaps the
//! generated `load` / `dump` closures for the compiled pair built here.
//!
//! This crate knows nothing about seared: no imports, no `Field`
//! introspection, not even its exception class (that rides in the spec). It
//! interprets a spec tree and nothing else, which is what keeps the family's
//! dependency arrow pointing one way.
//!
//! Layout: `spec` ingests the plain-data tree into something the hot loop can
//! walk; `kinds` holds the per-field coercions, each a transcription of one
//! seared field method; `load` and `dump` walk a schema in either direction.
//! `imports` caches the stdlib types the parse-and-construct kinds build.

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod dump;
mod imports;
mod kinds;
mod load;
mod spec;

use spec::Schema;

/// Spec shape this build understands; must equal `seared._core.accel.SPEC_ABI`.
/// seared declines the backend outright on a mismatch — this integer is the
/// whole compatibility gate.
const SPEC_ABI: u32 = 1;

#[cfg(debug_assertions)]
const BUILD_PROFILE: &str = "debug";
#[cfg(not(debug_assertions))]
const BUILD_PROFILE: &str = "release";

/// One compiled class. Its bound `load` / `dump` methods are what the seam
/// installs, so they carry the decorator's signatures exactly.
#[pyclass(frozen, module = "rusted._rusted")]
pub(crate) struct Compiled {
    schema: Schema,
}

#[pymethods]
impl Compiled {
    #[pyo3(signature = (data, format = "json"))]
    fn load(&self, py: Python<'_>, data: &Bound<'_, PyAny>, format: &str) -> PyResult<Py<PyAny>> {
        load::load(&self.schema, py, data, format)
    }

    #[pyo3(signature = (obj, format = "json"))]
    fn dump(&self, py: Python<'_>, obj: &Bound<'_, PyAny>, format: &str) -> PyResult<Py<PyAny>> {
        dump::dump(&self.schema, py, obj, format)
    }
}

/// Compile one class spec into a `(load, dump)` pair.
///
/// Returns `None` to decline — a kind this build doesn't implement — which
/// the seam records as the class's reason for keeping the Python path. Raises
/// only on a malformed or ABI-mismatched spec.
#[pyfunction]
fn compile_spec(
    py: Python<'_>,
    spec: &Bound<'_, PyDict>,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let Some(schema) = spec::parse(py, spec)? else {
        return Ok(None);
    };
    let compiled = Bound::new(py, Compiled { schema })?;
    Ok(Some((
        compiled.getattr("load")?.unbind(),
        compiled.getattr("dump")?.unbind(),
    )))
}

#[pymodule]
fn _rusted(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SPEC_ABI", SPEC_ABI)?;
    m.add("BUILD_PROFILE", BUILD_PROFILE)?;
    m.add_class::<Compiled>()?;
    m.add_function(wrap_pyfunction!(compile_spec, m)?)?;
    Ok(())
}
