//! Per-kind coercion, and the container orchestration around it.
//!
//! Every coercion here is a transcription of one seared field method — the
//! message text included, because the seam's promise is that swapping the
//! core changes nothing observable. The seared field modules are the source
//! of truth; the differential suite is what proves each transcription
//! stayed faithful.

pub(crate) mod scalar;
pub(crate) mod temporal;
pub(crate) mod value;

use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use crate::spec::FieldSpec;

/// Raise seared's `ValidationError` — the class carried in the spec, so a
/// caller's `except s.ValidationError` catches ours identically.
pub(crate) fn verr(error: &Bound<'_, PyType>, msg: String) -> PyErr {
    PyErr::from_type(error.clone(), msg)
}

pub(crate) fn type_name(v: &Bound<'_, PyAny>) -> String {
    v.get_type()
        .name()
        .map_or_else(|_| "?".to_string(), |n| n.to_string())
}

pub(crate) fn repr(v: &Bound<'_, PyAny>) -> String {
    v.repr().map_or_else(|_| "?".to_string(), |r| r.to_string())
}

pub(crate) fn ctor<T: PyTypeInfo>(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(T::type_object(py).call1((v,))?.unbind())
}

// ---------------------------------------------------------------------------
// Container orchestration — seared's `_core.decorator._apply`
// ---------------------------------------------------------------------------

/// Apply `one` across a field's container shape.
///
/// `keyed` → dict-of-V, `many` → list-of-V, otherwise the scalar itself. In
/// strict mode the container shape is guarded up front; in lax mode seared
/// just runs the comprehension, so a wrong shape surfaces as whatever Python
/// raises (`AttributeError` for a keyed non-dict, `TypeError` for a
/// non-iterable) — reproduced here rather than smoothed over.
pub(crate) fn apply<F>(
    py: Python<'_>,
    f: &FieldSpec,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    one: F,
) -> PyResult<Py<PyAny>>
where
    F: Fn(&Bound<'_, PyAny>) -> PyResult<Py<PyAny>>,
{
    if f.keyed {
        if validate && !v.is_instance_of::<PyDict>() {
            return Err(verr(
                err,
                format!("expected dict for keyed field, got {}", type_name(v)),
            ));
        }
        let out = PyDict::new(py);
        if let Ok(d) = v.downcast::<PyDict>() {
            for (k, item) in d.iter() {
                out.set_item(k, one(&item)?)?;
            }
        } else {
            for pair in v.call_method0("items")?.try_iter()? {
                let pair = pair?;
                let (k, item) = (pair.get_item(0)?, pair.get_item(1)?);
                out.set_item(k, one(&item)?)?;
            }
        }
        return Ok(out.into_any().unbind());
    }

    if f.many {
        if validate && !(v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>()) {
            return Err(verr(
                err,
                format!("expected list for many field, got {}", type_name(v)),
            ));
        }
        let mut out: Vec<Py<PyAny>> = Vec::new();
        if let Ok(list) = v.downcast::<PyList>() {
            out.reserve(list.len());
            for item in list.iter() {
                out.push(one(&item)?);
            }
        } else {
            for item in v.try_iter()? {
                out.push(one(&item?)?);
            }
        }
        return Ok(PyList::new(py, out)?.into_any().unbind());
    }

    one(v)
}
