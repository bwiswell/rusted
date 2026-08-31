//! Cached handles to the stdlib types the Tier 2 kinds parse into.
//!
//! Importing `uuid` or `decimal` is not a dependency on seared: the leaf
//! property this crate maintains is about *seared*, not about Python's own
//! standard library. What must travel in the spec is anything the **user**
//! chose — an enum class, a concrete path type, a strftime format — because
//! that is knowledge seared has and this crate cannot infer.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

pub(crate) struct Stdlib {
    pub(crate) uuid: Py<PyType>,
    pub(crate) date: Py<PyType>,
    pub(crate) datetime: Py<PyType>,
    pub(crate) time: Py<PyType>,
    pub(crate) timedelta: Py<PyType>,
    pub(crate) decimal: Py<PyType>,
    pub(crate) invalid_operation: Py<PyType>,
    pub(crate) pure_path: Py<PyType>,
    pub(crate) base64: Py<PyAny>,
}

static STDLIB: PyOnceLock<Stdlib> = PyOnceLock::new();

fn cls(py: Python<'_>, module: &str, name: &str) -> PyResult<Py<PyType>> {
    Ok(py
        .import(module)?
        .getattr(name)?
        .downcast_into::<PyType>()?
        .unbind())
}

/// Resolved once per interpreter, on first use of a Tier 2 kind.
pub(crate) fn stdlib(py: Python<'_>) -> PyResult<&'static Stdlib> {
    STDLIB.get_or_try_init(py, || {
        Ok(Stdlib {
            uuid: cls(py, "uuid", "UUID")?,
            date: cls(py, "datetime", "date")?,
            datetime: cls(py, "datetime", "datetime")?,
            time: cls(py, "datetime", "time")?,
            timedelta: cls(py, "datetime", "timedelta")?,
            decimal: cls(py, "decimal", "Decimal")?,
            invalid_operation: cls(py, "decimal", "InvalidOperation")?,
            pure_path: cls(py, "pathlib", "PurePath")?,
            base64: py.import("base64")?.into_any().unbind(),
        })
    })
}
