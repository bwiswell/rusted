//! Instance → wire dict.
//!
//! Mirrors `seared._core.decorator._make_dump`: skip `dump=False` fields,
//! skip `None` values, read attributes with a missing-slot tolerance, and
//! coerce through the same per-kind serialisers.

use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::kinds::temporal::{self, DateKind};
use crate::kinds::{apply, scalar, type_name, value, verr};
use crate::spec::{FieldSpec, Kind, Schema};

pub(crate) fn dump(
    schema: &Schema,
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    for f in &schema.fields {
        if !f.dump {
            continue;
        }
        let v = match obj.getattr(f.attr.bind(py)) {
            Ok(v) => v,
            // seared reads with `getattr(obj, attr, None)`: an unset slot
            // dumps as absent, same as None.
            Err(e) if e.is_instance_of::<PyAttributeError>(py) => continue,
            Err(e) => return Err(e),
        };
        if v.is_none() {
            continue;
        }
        out.set_item(f.wire.bind(py), ser_field(schema, py, f, &v, format)?)?;
    }
    Ok(out.into_any().unbind())
}

fn ser_field(
    schema: &Schema,
    py: Python<'_>,
    f: &FieldSpec,
    v: &Bound<'_, PyAny>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    let err = schema.error.bind(py);
    apply(py, f, v, schema.validate, err, |x| {
        ser_one(schema, py, f, x, format)
    })
}

fn ser_one(
    schema: &Schema,
    py: Python<'_>,
    f: &FieldSpec,
    v: &Bound<'_, PyAny>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    let err = schema.error.bind(py);
    let validate = schema.validate;
    match &f.kind {
        Kind::Int => scalar::ser_int(py, v, validate, err),
        Kind::Float => scalar::ser_float(py, v, validate, err),
        Kind::Str => scalar::ser_str(py, v, validate, err),
        Kind::Bool => scalar::ser_bool(py, v, validate, err),
        Kind::Uuid => value::ser_uuid(py, v, validate, err),
        Kind::Date(fmt) => {
            temporal::ser_dateish(py, v, validate, err, DateKind::Date, fmt.as_ref())
        }
        Kind::DateTime(fmt) => {
            temporal::ser_dateish(py, v, validate, err, DateKind::DateTime, fmt.as_ref())
        }
        Kind::Time(fmt) => {
            temporal::ser_dateish(py, v, validate, err, DateKind::Time, fmt.as_ref())
        }
        Kind::TimeDelta => temporal::ser_timedelta(py, v, validate, err),
        Kind::Decimal { as_number } => value::ser_decimal(py, v, validate, err, *as_number),
        // The only kind that reads the carrier hint.
        Kind::Bytes { hex } => value::ser_bytes(py, v, validate, err, *hex, format),
        Kind::Enum { cls, name, .. } => value::ser_enum(v, validate, err, cls.bind(py), name),
        Kind::Path { .. } => value::ser_path(py, v, validate, err),
        Kind::Dict => value::copy_dict(py, v, validate, err),
        Kind::Nested(sub) => {
            // T.serialize guards with the *parent's* validate flag, then dumps
            // through the nested class's own — two different flags, as in seared.
            if validate && !v.is_instance(sub.cls.bind(py))? {
                return Err(verr(
                    err,
                    format!("expected {}, got {}", sub.name, type_name(v)),
                ));
            }
            dump(sub, py, v, format)
        }
    }
}
