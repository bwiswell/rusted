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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyDict, PyList, PyString, PyTuple, PyType};

use crate::spec::FieldSpec;

/// Raise seared's `ValidationError` — the class carried in the spec, so a
/// caller's `except s.ValidationError` catches ours identically.
///
/// A bare reason, with no path and no class, which is what every seared
/// field raises: the location is added as the error passes back up, by
/// [`relocate`].
pub(crate) fn verr(error: &Bound<'_, PyType>, msg: String) -> PyErr {
    PyErr::from_type(error.clone(), msg)
}

/// Raise the spec's error class with its location already set — for the
/// errors seared's decorator builds itself (a required key missing, a
/// non-dict to `load`), which know where they are when they're raised.
pub(crate) fn verr_at(
    py: Python<'_>,
    error: &Bound<'_, PyType>,
    reason: String,
    path: &Bound<'_, PyTuple>,
    cls_name: &str,
) -> PyErr {
    let built = (|| {
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "path"), path)?;
        kwargs.set_item(intern!(py, "cls_name"), cls_name)?;
        error.call((reason,), Some(&kwargs))
    })();
    match built {
        Ok(instance) => PyErr::from_value(instance),
        Err(e) => e,
    }
}

/// `e`, with `segment` prepended to its path and, when given, `cls_name` set.
///
/// The `SPEC_ABI` 2 error contract: a backend must end with the same
/// `reason`, `path` and `cls_name` as seared's Python path, which prepends in
/// place as an error unwinds (`add_location`). This rebuilds through the
/// class's public constructor instead, so the ABI carries no private method,
/// and costs nothing until something has failed. Anything that isn't an
/// instance of the spec's error class passes through untouched, as it does in
/// seared.
pub(crate) fn relocate(
    py: Python<'_>,
    e: PyErr,
    error: &Bound<'_, PyType>,
    segment: Option<&Bound<'_, PyAny>>,
    cls_name: Option<&str>,
) -> PyErr {
    if !e.is_instance(py, error.as_any()) {
        return e;
    }
    let rebuilt = (|| {
        let old = e.value(py);
        let mut path: Vec<Bound<'_, PyAny>> = segment.into_iter().cloned().collect();
        for part in old.getattr(intern!(py, "path"))?.try_iter()? {
            path.push(part?);
        }
        let cls_name = match cls_name {
            Some(name) => PyString::new(py, name).into_any(),
            None => old.getattr(intern!(py, "cls_name"))?,
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "path"), PyTuple::new(py, path)?)?;
        kwargs.set_item(intern!(py, "cls_name"), cls_name)?;
        error.call((old.getattr(intern!(py, "reason"))?,), Some(&kwargs))
    })();
    match rebuilt {
        Ok(instance) => PyErr::from_value(instance),
        Err(failed) => failed,
    }
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
        // A failing item's key joins the error's path, as seared's
        // `_first_failure` adds it: the key object itself, not its str.
        let out = PyDict::new(py);
        if let Ok(d) = v.downcast::<PyDict>() {
            for (k, item) in d.iter() {
                let x = one(&item).map_err(|e| relocate(py, e, err, Some(&k), None))?;
                out.set_item(k, x)?;
            }
        } else {
            for pair in v.call_method0("items")?.try_iter()? {
                let pair = pair?;
                let (k, item) = (pair.get_item(0)?, pair.get_item(1)?);
                let x = one(&item).map_err(|e| relocate(py, e, err, Some(&k), None))?;
                out.set_item(k, x)?;
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
        // A failing item's index joins the error's path.
        let at = |i: usize, e: PyErr| {
            let Ok(index) = i.into_pyobject(py);
            relocate(py, e, err, Some(index.as_any()), None)
        };
        let mut out: Vec<Py<PyAny>> = Vec::new();
        if let Ok(list) = v.downcast::<PyList>() {
            out.reserve(list.len());
            for (i, item) in list.iter().enumerate() {
                out.push(one(&item).map_err(|e| at(i, e))?);
            }
        } else {
            for (i, item) in v.try_iter()?.enumerate() {
                out.push(one(&item?).map_err(|e| at(i, e))?);
            }
        }
        return Ok(PyList::new(py, out)?.into_any().unbind());
    }

    one(v)
}
