//! Tier 1 scalars — `int`, `float`, `str`, `bool`.
//!
//! Transcribed from `seared/fields/{int_,float_,str_,bool_}.py`, message
//! text included. Note which branches are gated on `validate` and which are
//! not: seared's `deserialize` raises on genuinely un-coercible input in
//! *both* modes, and only the type *guards* are strict-only.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyType};

use super::{ctor, repr, type_name, verr};

// ---------------------------------------------------------------------------
// int
// ---------------------------------------------------------------------------

pub(crate) fn de_int(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance_of::<PyBool>() {
        if validate {
            return Err(verr(err, "expected int, got bool".to_string()));
        }
        return ctor::<PyInt>(py, v);
    }
    if v.is_instance_of::<PyInt>() {
        return Ok(v.clone().unbind());
    }
    if v.is_instance_of::<PyString>() || v.is_instance_of::<PyFloat>() {
        // seared: int(value), with TypeError/ValueError folded into a
        // ValidationError. Anything else (OverflowError, a user __int__)
        // propagates untouched.
        return match PyInt::type_object(py).call1((v,)) {
            Ok(x) => Ok(x.unbind()),
            Err(e)
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) =>
            {
                Err(verr(err, format!("cannot deserialize {} as int", repr(v))))
            }
            Err(e) => Err(e),
        };
    }
    Err(verr(err, format!("cannot deserialize {} as int", repr(v))))
}

pub(crate) fn ser_int(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if validate && (v.is_instance_of::<PyBool>() || !v.is_instance_of::<PyInt>()) {
        return Err(verr(err, format!("expected int, got {}", type_name(v))));
    }
    if v.is_exact_instance_of::<PyInt>() {
        return Ok(v.clone().unbind());
    }
    ctor::<PyInt>(py, v)
}

// ---------------------------------------------------------------------------
// float
// ---------------------------------------------------------------------------

pub(crate) fn de_float(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance_of::<PyBool>() {
        if validate {
            return Err(verr(err, "expected float, got bool".to_string()));
        }
        return ctor::<PyFloat>(py, v);
    }
    if v.is_exact_instance_of::<PyFloat>() {
        return Ok(v.clone().unbind());
    }
    if v.is_instance_of::<PyInt>() || v.is_instance_of::<PyFloat>() {
        return ctor::<PyFloat>(py, v);
    }
    if v.is_instance_of::<PyString>() {
        // seared catches only ValueError here, not TypeError.
        return match PyFloat::type_object(py).call1((v,)) {
            Ok(x) => Ok(x.unbind()),
            Err(e) if e.is_instance_of::<PyValueError>(py) => Err(verr(
                err,
                format!("cannot deserialize {} as float", repr(v)),
            )),
            Err(e) => Err(e),
        };
    }
    Err(verr(
        err,
        format!("cannot deserialize {} as float", repr(v)),
    ))
}

pub(crate) fn ser_float(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if validate
        && (v.is_instance_of::<PyBool>()
            || !(v.is_instance_of::<PyInt>() || v.is_instance_of::<PyFloat>()))
    {
        return Err(verr(err, format!("expected float, got {}", type_name(v))));
    }
    if v.is_exact_instance_of::<PyFloat>() {
        return Ok(v.clone().unbind());
    }
    ctor::<PyFloat>(py, v)
}

// ---------------------------------------------------------------------------
// str
// ---------------------------------------------------------------------------

pub(crate) fn de_str(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance_of::<PyString>() {
        return Ok(v.clone().unbind());
    }
    if validate {
        return Err(verr(err, format!("expected str, got {}", type_name(v))));
    }
    ctor::<PyString>(py, v)
}

pub(crate) fn ser_str(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if validate && !v.is_instance_of::<PyString>() {
        return Err(verr(err, format!("expected str, got {}", type_name(v))));
    }
    if v.is_exact_instance_of::<PyString>() {
        return Ok(v.clone().unbind());
    }
    ctor::<PyString>(py, v)
}

// ---------------------------------------------------------------------------
// bool
// ---------------------------------------------------------------------------

const TRUE_WORDS: [&str; 4] = ["true", "1", "yes", "on"];
const FALSE_WORDS: [&str; 4] = ["false", "0", "no", "off"];

pub(crate) fn de_bool(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance_of::<PyBool>() {
        return Ok(v.clone().unbind());
    }
    if !validate {
        return ctor::<PyBool>(py, v);
    }
    if v.is_instance_of::<PyString>() {
        // Python's own strip()/lower() rather than Rust's — the Unicode
        // definitions differ at the edges, and parity beats a few nanoseconds
        // on a path that is already the slow branch.
        let low: String = v.call_method0("strip")?.call_method0("lower")?.extract()?;
        if TRUE_WORDS.contains(&low.as_str()) {
            return Ok(PyBool::new(py, true).to_owned().into_any().unbind());
        }
        if FALSE_WORDS.contains(&low.as_str()) {
            return Ok(PyBool::new(py, false).to_owned().into_any().unbind());
        }
    }
    if v.is_instance_of::<PyInt>() {
        return ctor::<PyBool>(py, v);
    }
    Err(verr(err, format!("cannot deserialize {} as bool", repr(v))))
}

pub(crate) fn ser_bool(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if validate && !v.is_instance_of::<PyBool>() {
        return Err(verr(err, format!("expected bool, got {}", type_name(v))));
    }
    if v.is_instance_of::<PyBool>() {
        return Ok(v.clone().unbind());
    }
    ctor::<PyBool>(py, v)
}
