//! Date, datetime, time and timedelta.
//!
//! Transcribed from `seared/fields/{date,datetime_,time_,timedelta}.py`.
//!
//! The three date-likes share a shape: an `isinstance` guard, then either
//! `isoformat()` / `fromisoformat()` or a `strftime` / `strptime` round trip
//! through the field's `format=`. They differ only in the class they check
//! and the three distinct labels seared puts in its messages.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyDict, PyFloat, PyString, PyType};

use super::{ctor, repr, type_name, verr};
use crate::imports::stdlib;

/// Which `datetime.*` class a date-like kind binds to, and the three labels
/// seared uses for it — they are not the same word in every message.
#[derive(Clone, Copy)]
pub(crate) enum DateKind {
    Date,
    DateTime,
    Time,
}

impl DateKind {
    /// `expected {}, got ...` on serialize.
    fn ser_label(self) -> &'static str {
        match self {
            Self::Date => "date",
            Self::DateTime => "datetime",
            Self::Time => "time",
        }
    }

    /// `expected str for {}, got ...` on deserialize.
    fn field_label(self) -> &'static str {
        match self {
            Self::Date => "Date",
            Self::DateTime => "DateTime",
            Self::Time => "Time",
        }
    }

    fn class<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyType>> {
        let std = stdlib(py)?;
        Ok(match self {
            Self::Date => std.date.bind(py).clone(),
            Self::DateTime => std.datetime.bind(py).clone(),
            Self::Time => std.time.bind(py).clone(),
        })
    }
}

pub(crate) fn ser_dateish(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
    kind: DateKind,
    format: Option<&Py<PyString>>,
) -> PyResult<Py<PyAny>> {
    if !v.is_instance(kind.class(py)?.as_any())? {
        if validate {
            return Err(verr(
                err,
                format!("expected {}, got {}", kind.ser_label(), type_name(v)),
            ));
        }
        return ctor::<PyString>(py, v);
    }
    match format {
        None => Ok(v.call_method0("isoformat")?.unbind()),
        Some(f) => Ok(v.call_method1("strftime", (f.bind(py),))?.unbind()),
    }
}

pub(crate) fn de_dateish(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    err: &Bound<'_, PyType>,
    kind: DateKind,
    format: Option<&Py<PyString>>,
) -> PyResult<Py<PyAny>> {
    let cls = kind.class(py)?;
    if v.is_instance(cls.as_any())? {
        return Ok(v.clone().unbind());
    }
    // Deliberately *not* gated on `validate` — seared raises here in both modes.
    if !v.is_instance_of::<PyString>() {
        return Err(verr(
            err,
            format!(
                "expected str for {}, got {}",
                kind.field_label(),
                type_name(v)
            ),
        ));
    }
    let parsed = match format {
        None => cls.call_method1("fromisoformat", (v,)),
        Some(f) => {
            // seared always parses through `datetime.strptime`, then narrows.
            let dt = stdlib(py)?
                .datetime
                .bind(py)
                .call_method1("strptime", (v, f.bind(py)));
            match dt {
                Err(e) => Err(e),
                Ok(dt) => match kind {
                    DateKind::Date => dt.call_method0("date"),
                    DateKind::Time => dt.call_method0("time"),
                    DateKind::DateTime => Ok(dt),
                },
            }
        }
    };
    match parsed {
        Ok(x) => Ok(x.unbind()),
        Err(e) if e.is_instance_of::<PyValueError>(py) => Err(verr(
            err,
            format!("invalid {} {}", kind.ser_label(), repr(v)),
        )),
        Err(e) => Err(e),
    }
}

pub(crate) fn ser_timedelta(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    validate: bool,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    if v.is_instance(stdlib(py)?.timedelta.bind(py).as_any())? {
        return Ok(v.call_method0("total_seconds")?.unbind());
    }
    if validate {
        return Err(verr(
            err,
            format!("expected timedelta, got {}", type_name(v)),
        ));
    }
    ctor::<PyFloat>(py, v)
}

pub(crate) fn de_timedelta(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    err: &Bound<'_, PyType>,
) -> PyResult<Py<PyAny>> {
    let td = stdlib(py)?.timedelta.bind(py);
    if v.is_instance(td.as_any())? {
        return Ok(v.clone().unbind());
    }
    // seared: timedelta(seconds=float(value)), with TypeError/ValueError from
    // either call folded into one message.
    let built = (|| -> PyResult<Py<PyAny>> {
        let seconds = PyFloat::type_object(py).call1((v,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seconds", seconds)?;
        Ok(td.call((), Some(&kwargs))?.unbind())
    })();
    match built {
        Ok(x) => Ok(x),
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            Err(verr(
                err,
                format!("cannot deserialize {} as timedelta", repr(v)),
            ))
        }
        Err(e) => Err(e),
    }
}
