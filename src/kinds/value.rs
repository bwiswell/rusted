//! UUID, Decimal, Bytes, Enum, Path and Dict.
//!
//! Transcribed from `seared/fields/{uuid_,decimal_,bytes_,enum_,path,dict_}.py`.
//!
//! These are the "parse and construct" kinds: most of their cost is building a
//! Python object, which is work no compiled core can remove. Accelerating them
//! is worth it not for their own speed but because acceleration is per-class
//! all-or-nothing — one `Bytes` field used to disqualify an otherwise fast
//! class, and now doesn't.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyInt, PyString, PyType};

use super::{ctor, repr, type_name, verr};
use crate::imports::stdlib;

// ---------------------------------------------------------------------------
// uuid
// ---------------------------------------------------------------------------

pub(crate) fn ser_uuid(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance(stdlib(py)?.uuid.bind(py).as_any())? {
        return ctor::<PyString>(py, v);
    }
    if validate {
        return Err(verr(err, format!("expected UUID, got {}", type_name(v))));
    }
    ctor::<PyString>(py, v)
}

pub(crate) fn de_uuid(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    let uuid = stdlib(py)?.uuid.bind(py);
    if v.is_instance(uuid.as_any())? {
        return Ok(v.clone().unbind());
    }
    if v.is_instance_of::<PyString>() {
        return match uuid.call1((v,)) {
            Ok(x) => Ok(x.unbind()),
            Err(e) if e.is_instance_of::<PyValueError>(py) => {
                Err(verr(err, format!("invalid UUID: {}", repr(v))))
            }
            Err(e) => Err(e),
        };
    }
    Err(verr(err, format!("cannot deserialize {} as UUID", repr(v))))
}

// ---------------------------------------------------------------------------
// decimal
// ---------------------------------------------------------------------------

pub(crate) fn ser_decimal(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    as_number: bool,
) -> PyResult<Py<PyAny>> {
    if !v.is_instance(stdlib(py)?.decimal.bind(py).as_any())? {
        if validate {
            return Err(verr(err, format!("expected Decimal, got {}", type_name(v))));
        }
        return Ok(v.clone().unbind());
    }
    if as_number {
        return ctor::<pyo3::types::PyFloat>(py, v);
    }
    ctor::<PyString>(py, v)
}

pub(crate) fn de_decimal(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    let std = stdlib(py)?;
    let decimal = std.decimal.bind(py);
    if v.is_instance(decimal.as_any())? {
        return Ok(v.clone().unbind());
    }
    // seared: Decimal(str(value)).
    let built = (|| -> PyResult<Py<PyAny>> {
        let text = PyString::type_object(py).call1((v,))?;
        Ok(decimal.call1((text,))?.unbind())
    })();
    match built {
        Ok(x) => Ok(x),
        Err(e)
            if e.is_instance_of::<PyTypeError>(py)
                || e.is_instance_of::<PyValueError>(py)
                || e.is_instance(py, std.invalid_operation.bind(py).as_any()) =>
        {
            if validate {
                // The message embeds the underlying exception's own text.
                let cause = e.value(py).str()?.to_string();
                return Err(verr(
                    err,
                    format!("cannot parse {} as Decimal: {cause}", repr(v)),
                ));
            }
            Ok(v.clone().unbind())
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// bytes
// ---------------------------------------------------------------------------

pub(crate) fn ser_bytes(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    hex: bool,
    format: &str,
) -> PyResult<Py<PyAny>> {
    if validate
        && !(v.is_instance_of::<PyBytes>()
            || v.is_instance_of::<PyByteArray>()
            || v.is_instance_of::<pyo3::types::PyMemoryView>())
    {
        return Err(verr(err, format!("expected bytes, got {}", type_name(v))));
    }
    let raw = PyBytes::type_object(py).call1((v,))?;
    // The one kind that reads the carrier hint: msgpack takes native bytes,
    // skipping base64's ~33% overhead. JSON can't, so it doesn't.
    if format == "msgpack" {
        return Ok(raw.unbind());
    }
    if hex {
        return Ok(raw.call_method0("hex")?.unbind());
    }
    let encoded = stdlib(py)?
        .base64
        .bind(py)
        .call_method1("b64encode", (raw,))?;
    Ok(encoded.call_method1("decode", ("ascii",))?.unbind())
}

pub(crate) fn de_bytes(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    hex: bool,
) -> PyResult<Py<PyAny>> {
    // Native bytes off a binary carrier pass straight through — note
    // memoryview is accepted on the way out but not on the way in, as in seared.
    if v.is_instance_of::<PyBytes>() || v.is_instance_of::<PyByteArray>() {
        return ctor::<PyBytes>(py, v);
    }
    if validate && !v.is_instance_of::<PyString>() {
        return Err(verr(
            err,
            format!("expected str for Bytes, got {}", type_name(v)),
        ));
    }
    let decoded = if hex {
        PyBytes::type_object(py).call_method1("fromhex", (v,))
    } else {
        stdlib(py)?.base64.bind(py).call_method1("b64decode", (v,))
    };
    match decoded {
        Ok(x) => Ok(x.unbind()),
        Err(e) if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py) => {
            let encoding = if hex { "hex" } else { "base64" };
            Err(verr(err, format!("invalid {encoding} bytes: {}", repr(v))))
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// enum
// ---------------------------------------------------------------------------

pub(crate) fn ser_enum(
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    cls: &Bound<'_, PyType>,
    name: &str,
) -> PyResult<Py<PyAny>> {
    if v.is_instance(cls.as_any())? {
        return Ok(v.getattr("value")?.unbind());
    }
    if validate {
        return Err(verr(err, format!("expected {name}, got {}", type_name(v))));
    }
    Ok(v.clone().unbind())
}

pub(crate) fn de_enum(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    err: &Bound<'_, PyType>,
    cls: &Bound<'_, PyType>,
    name: &str,
    int_valued: bool,
) -> PyResult<Py<PyAny>> {
    if v.is_instance(cls.as_any())? {
        return Ok(v.clone().unbind());
    }
    // Whether members are int-valued is decided once, at compile time; seared
    // re-derives it from `next(iter(enum))` on every call.
    let looked_up = if int_valued {
        PyInt::type_object(py)
            .call1((v,))
            .and_then(|as_int| cls.call1((as_int,)))
    } else {
        cls.call1((v,))
    };
    match looked_up {
        Ok(x) => Ok(x.unbind()),
        Err(e) if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py) => {
            Err(verr(err, format!("{} is not a valid {name}", repr(v))))
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// path
// ---------------------------------------------------------------------------

pub(crate) fn ser_path(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if !v.is_instance(stdlib(py)?.pure_path.bind(py).as_any())? {
        if validate {
            return Err(verr(
                err,
                format!("expected pathlib.Path, got {}", type_name(v)),
            ));
        }
        return Ok(v.clone().unbind());
    }
    // Always POSIX on the wire, on every host.
    Ok(v.call_method0("as_posix")?.unbind())
}

pub(crate) fn de_path(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    concrete: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance(stdlib(py)?.pure_path.bind(py).as_any())? {
        return Ok(v.clone().unbind());
    }
    if !v.is_instance_of::<PyString>() {
        if validate {
            return Err(verr(
                err,
                format!("expected str path, got {}", type_name(v)),
            ));
        }
        return Ok(v.clone().unbind());
    }
    Ok(concrete.call1((v,))?.unbind())
}

// ---------------------------------------------------------------------------
// dict
// ---------------------------------------------------------------------------

/// Shallow copy in both directions; values pass through untouched.
pub(crate) fn copy_dict(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if validate && !v.is_instance_of::<PyDict>() {
        return Err(verr(err, format!("expected dict, got {}", type_name(v))));
    }
    ctor::<PyDict>(py, v)
}
