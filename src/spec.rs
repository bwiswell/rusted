//! Spec ingestion.
//!
//! seared hands over a plain-data tree (see `seared._core.accel`); this turns
//! it into something the hot loop can walk without touching a Python dict
//! again — attr/wire strings interned once, `__new__` resolved once, kinds
//! reduced to an enum.
//!
//! Two failure modes, deliberately distinct:
//!
//! - **Decline** (`Ok(None)`) — a kind this build doesn't implement. Not an
//!   error: a newer seared may emit kinds an older `rusted` predates, and the
//!   seam's contract is that the class quietly keeps the Python path.
//! - **Error** (`Err`) — a malformed or mismatched spec. That's a bug worth
//!   surfacing, and the seam reports it in the class's decline reason.

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyType};

use crate::SPEC_ABI;

pub(crate) enum Kind {
    // Tier 1
    Int,
    Float,
    Str,
    Bool,
    Nested(Box<Schema>),
    // Tier 2 — parse-and-construct. Configuration that the *user* chose (an
    // enum class, a concrete path type, a strftime format) arrives in the
    // spec; how to use it is this crate's business.
    Uuid,
    Date(Option<Py<PyString>>),
    DateTime(Option<Py<PyString>>),
    Time(Option<Py<PyString>>),
    TimeDelta,
    Decimal {
        as_number: bool,
    },
    Bytes {
        hex: bool,
    },
    Enum {
        cls: Py<PyType>,
        name: String,
        /// Whether members carry int values, decided once here rather than
        /// re-derived from `next(iter(enum))` on every deserialize.
        int_valued: bool,
    },
    Path {
        concrete: Py<PyType>,
    },
    Dict,
}

pub(crate) struct FieldSpec {
    pub(crate) attr: Py<PyString>,
    pub(crate) wire: Py<PyString>,
    pub(crate) kind: Kind,
    pub(crate) required: bool,
    pub(crate) many: bool,
    pub(crate) keyed: bool,
    pub(crate) dump: bool,
    /// The resolved default — seared has already folded `default=` into
    /// `missing`. `None` means the default is Python `None`.
    pub(crate) default: Option<Py<PyAny>>,
    pub(crate) default_factory: Option<Py<PyAny>>,
}

pub(crate) struct Schema {
    pub(crate) cls: Py<PyType>,
    pub(crate) new_fn: Py<PyAny>,
    pub(crate) name: String,
    pub(crate) validate: bool,
    /// seared's `ValidationError`, carried *in the spec*. This crate never
    /// imports seared — the error class is data like everything else, which
    /// is what keeps `rusted` a leaf with no runtime dependency.
    pub(crate) error: Py<PyType>,
    pub(crate) fields: Vec<FieldSpec>,
}

fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("spec is missing '{key}'")))
}

fn flag(d: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    item(d, key)?.extract()
}

/// A date-like kind's optional `format=` (a strftime string), interned.
fn opt_format(py: Python<'_>, d: &Bound<'_, PyDict>) -> PyResult<Option<Py<PyString>>> {
    let v = item(d, "format")?;
    if v.is_none() {
        return Ok(None);
    }
    let s: String = v.extract()?;
    Ok(Some(PyString::intern(py, &s).unbind()))
}

fn opt(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Py<PyAny>>> {
    let v = item(d, key)?;
    Ok(if v.is_none() { None } else { Some(v.unbind()) })
}

pub(crate) fn parse(py: Python<'_>, spec: &Bound<'_, PyDict>) -> PyResult<Option<Schema>> {
    let abi: u32 = item(spec, "abi")?.extract()?;
    if abi != SPEC_ABI {
        return Err(PyValueError::new_err(format!(
            "rusted understands SPEC_ABI {SPEC_ABI}, got {abi}"
        )));
    }
    let cls = item(spec, "cls")?
        .downcast_into::<PyType>()
        .map_err(|_| PyTypeError::new_err("spec 'cls' must be a class"))?;
    let error = item(spec, "error")?
        .downcast_into::<PyType>()
        .map_err(|_| PyTypeError::new_err("spec 'error' must be an exception class"))?;
    let name: String = item(spec, "name")?.extract()?;
    let validate: bool = item(spec, "validate")?.extract()?;
    // Resolved once: `load` builds instances via `__new__` + slot assignment,
    // bypassing __init__ the way pydantic-core does. The seam declines any
    // class where that would be observable.
    let new_fn = cls.getattr("__new__")?.unbind();

    let raw = item(spec, "fields")?;
    let raw = raw
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("spec 'fields' must be a list"))?;
    let mut fields = Vec::with_capacity(raw.len());
    for f in raw.iter() {
        let f = f
            .downcast_into::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each field spec must be a dict"))?;
        let Some(field) = parse_field(py, &f)? else {
            return Ok(None);
        };
        fields.push(field);
    }

    Ok(Some(Schema {
        cls: cls.unbind(),
        new_fn,
        name,
        validate,
        error: error.unbind(),
        fields,
    }))
}

fn parse_field(py: Python<'_>, f: &Bound<'_, PyDict>) -> PyResult<Option<FieldSpec>> {
    let attr: String = item(f, "attr")?.extract()?;
    let wire: String = item(f, "wire")?.extract()?;
    let kind_name: String = item(f, "kind")?.extract()?;

    let kind = match kind_name.as_str() {
        "int" => Kind::Int,
        "float" => Kind::Float,
        "str" => Kind::Str,
        "bool" => Kind::Bool,
        "uuid" => Kind::Uuid,
        "date" => Kind::Date(opt_format(py, f)?),
        "datetime" => Kind::DateTime(opt_format(py, f)?),
        "time" => Kind::Time(opt_format(py, f)?),
        "timedelta" => Kind::TimeDelta,
        "dict" => Kind::Dict,
        "decimal" => Kind::Decimal {
            as_number: flag(f, "as_number")?,
        },
        "bytes" => {
            let encoding: String = item(f, "encoding")?.extract()?;
            // seared treats anything that isn't 'hex' as base64.
            Kind::Bytes {
                hex: encoding == "hex",
            }
        }
        "path" => Kind::Path {
            concrete: item(f, "concrete")?
                .downcast_into::<PyType>()
                .map_err(|_| PyTypeError::new_err("a path field spec needs a 'concrete' class"))?
                .unbind(),
        },
        "enum" => {
            let cls = item(f, "enum")?
                .downcast_into::<PyType>()
                .map_err(|_| PyTypeError::new_err("an enum field spec needs an 'enum' class"))?;
            // seared reads `next(iter(enum)).value` to choose int-vs-plain
            // lookup, which raises StopIteration on an empty enum. Rather than
            // reproduce that at *compile* time, decline and let the Python path
            // raise it at the same moment seared always did.
            let Some(first) = cls.as_any().try_iter()?.next() else {
                return Ok(None);
            };
            let sample = first?.getattr("value")?;
            let int_valued = sample.is_instance_of::<PyInt>() && !sample.is_instance_of::<PyBool>();
            Kind::Enum {
                name: cls.name()?.to_string(),
                cls: cls.unbind(),
                int_valued,
            }
        }
        "nested" => {
            let sub = item(f, "schema")?;
            let sub = sub
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("a nested field spec needs a 'schema' dict"))?;
            // A nested class this build can't take disqualifies the parent:
            // acceleration is per-class all-or-nothing, recursively.
            let Some(schema) = parse(py, sub)? else {
                return Ok(None);
            };
            Kind::Nested(Box::new(schema))
        }
        _ => return Ok(None),
    };

    Ok(Some(FieldSpec {
        attr: PyString::intern(py, &attr).unbind(),
        wire: PyString::intern(py, &wire).unbind(),
        kind,
        required: flag(f, "required")?,
        many: flag(f, "many")?,
        keyed: flag(f, "keyed")?,
        dump: flag(f, "dump")?,
        default: opt(f, "default")?,
        default_factory: opt(f, "default_factory")?,
    }))
}
