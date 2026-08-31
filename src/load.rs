//! Wire dict → instance.
//!
//! Mirrors the closure `seared._core.decorator._make_load` builds: the same
//! key-resolution order, the same required/default fallbacks, the same
//! per-field coercion. The one deliberate difference is construction —
//! `__new__` plus attribute assignment instead of `cls(**kwargs)`, which is
//! sound only because the seam declines any class where that is observable
//! (a hand-written `__init__`, or a `__post_init__`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::kinds::{self, apply, type_name, verr};
use crate::spec::{FieldSpec, Kind, Schema};

pub(crate) fn load(
    schema: &Schema,
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    let err = schema.error.bind(py);
    let Ok(dict) = data.downcast::<PyDict>() else {
        return Err(verr(
            err,
            format!(
                "{}.load expected dict, got {}",
                schema.name,
                type_name(data)
            ),
        ));
    };

    let inst = schema.new_fn.bind(py).call1((schema.cls.bind(py),))?;
    for f in &schema.fields {
        let value: Py<PyAny> = match dict.get_item(f.wire.bind(py))? {
            Some(v) => de_field(schema, py, f, &v, format)?,
            None => {
                if f.required {
                    return Err(verr(
                        err,
                        format!("{}.{} is required", schema.name, f.attr.bind(py)),
                    ));
                } else if let Some(factory) = &f.default_factory {
                    factory.bind(py).call0()?.unbind()
                } else if let Some(d) = &f.default {
                    // Shared, not copied — seared's load path assigns the same
                    // `missing` object too. Deep-copying is the *constructor*
                    // wrapper's job, not load's.
                    d.clone_ref(py)
                } else {
                    py.None()
                }
            }
        };
        inst.setattr(f.attr.bind(py), value)?;
    }
    Ok(inst.unbind())
}

fn de_field(
    schema: &Schema,
    py: Python<'_>,
    f: &FieldSpec,
    v: &Bound<'_, PyAny>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    if v.is_none() {
        return Ok(py.None());
    }
    let err = schema.error.bind(py);
    apply(py, f, v, schema.validate, err, |x| {
        de_one(schema, py, f, x, format)
    })
}

fn de_one(
    schema: &Schema,
    py: Python<'_>,
    f: &FieldSpec,
    v: &Bound<'_, PyAny>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    let err = schema.error.bind(py);
    let validate = schema.validate;
    match &f.kind {
        Kind::Int => kinds::de_int(py, v, validate, err),
        Kind::Float => kinds::de_float(py, v, validate, err),
        Kind::Str => kinds::de_str(py, v, validate, err),
        Kind::Bool => kinds::de_bool(py, v, validate, err),
        Kind::Nested(sub) => {
            // T.deserialize: an already-built instance passes straight through;
            // otherwise the nested class loads under *its own* validate flag,
            // exactly as `self.schema.load(value)` does.
            if v.is_instance(sub.cls.bind(py))? {
                Ok(v.clone().unbind())
            } else {
                load(sub, py, v, format)
            }
        }
    }
}
