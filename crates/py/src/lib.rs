//! Python bindings for the Pauli propagation engine.

use pauliprop::{
    Coefficient, Complex64, CustomTruncation, Gate, PauliOp, PauliString, PauliSum as CoreSum,
    Picture, Truncation, WeightDamping,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyFloat, PyType};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

fn value_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// The rows of a square matrix one after the other.
fn flatten<T>(matrix: Vec<Vec<T>>) -> PyResult<Vec<T>> {
    let size = matrix.len();
    if matrix.iter().any(|row| row.len() != size) {
        return Err(value_error("the matrix is not square"));
    }
    Ok(matrix.into_iter().flatten().collect())
}

/// Pairs the characters of `paulis` with `qubits`.
fn parse_ops(paulis: &str, qubits: &[usize]) -> PyResult<Vec<(usize, PauliOp)>> {
    let ops: Vec<PauliOp> = paulis
        .chars()
        .map(|c| {
            PauliOp::from_char(c)
                .ok_or_else(|| value_error(format!("'{c}' is not one of I, X, Y, Z")))
        })
        .collect::<PyResult<_>>()?;
    if ops.len() != qubits.len() {
        return Err(value_error(format!(
            "{} Pauli operators were given for {} qubits",
            ops.len(),
            qubits.len()
        )));
    }
    Ok(qubits.iter().copied().zip(ops).collect())
}

/// A coefficient as Python sees it: a `float` or a `complex`.
#[derive(Clone, Copy)]
enum Number {
    Real(f64),
    Complex(Complex64),
}

impl Number {
    fn scale(self, factor: f64) -> Self {
        match self {
            Number::Real(value) => Number::Real(value * factor),
            Number::Complex(value) => Number::Complex(value * factor),
        }
    }
}

impl<'py> FromPyObject<'py> for Number {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<PyComplex>() {
            Ok(Number::Complex(ob.extract()?))
        } else {
            Ok(Number::Real(ob.extract()?))
        }
    }
}

impl<'py> IntoPyObject<'py> for Number {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(match self {
            Number::Real(value) => value.into_pyobject(py)?.into_any(),
            Number::Complex(value) => value.into_pyobject(py)?.into_any(),
        })
    }
}

/// A coefficient type of the engine, with its conversions to and from Python numbers.
trait Scalar: Coefficient {
    fn number(self) -> Number;
    fn from_number(number: Number) -> PyResult<Self>;
}

impl Scalar for f64 {
    fn number(self) -> Number {
        Number::Real(self)
    }

    fn from_number(number: Number) -> PyResult<Self> {
        match number {
            Number::Real(value) => Ok(value),
            Number::Complex(_) => Err(PyTypeError::new_err(
                "the sum has real coefficients; create it with dtype=complex to add a complex one",
            )),
        }
    }
}

impl Scalar for Complex64 {
    fn number(self) -> Number {
        Number::Complex(self)
    }

    fn from_number(number: Number) -> PyResult<Self> {
        Ok(match number {
            Number::Real(value) => Complex64::new(value, 0.0),
            Number::Complex(value) => value,
        })
    }
}

/// A Pauli sum with real or complex coefficients, whose strings are as wide as its number
/// of qubits needs.
#[derive(Clone)]
enum AnySum {
    R1(CoreSum<1>),
    R2(CoreSum<2>),
    R4(CoreSum<4>),
    R8(CoreSum<8>),
    C1(CoreSum<1, Complex64>),
    C2(CoreSum<2, Complex64>),
    C4(CoreSum<4, Complex64>),
    C8(CoreSum<8, Complex64>),
}

macro_rules! dispatch {
    ($sum:expr, $s:ident => $body:expr) => {
        match $sum {
            AnySum::R1($s) => $body,
            AnySum::R2($s) => $body,
            AnySum::R4($s) => $body,
            AnySum::R8($s) => $body,
            AnySum::C1($s) => $body,
            AnySum::C2($s) => $body,
            AnySum::C4($s) => $body,
            AnySum::C8($s) => $body,
        }
    };
}

/// Evaluates the body for two sums of the same kind, and gives `None` for two of different kinds.
macro_rules! pairwise {
    ($a:expr, $b:expr, $x:ident, $y:ident => $body:expr) => {
        match ($a, $b) {
            (AnySum::R1($x), AnySum::R1($y)) => Some($body),
            (AnySum::R2($x), AnySum::R2($y)) => Some($body),
            (AnySum::R4($x), AnySum::R4($y)) => Some($body),
            (AnySum::R8($x), AnySum::R8($y)) => Some($body),
            (AnySum::C1($x), AnySum::C1($y)) => Some($body),
            (AnySum::C2($x), AnySum::C2($y)) => Some($body),
            (AnySum::C4($x), AnySum::C4($y)) => Some($body),
            (AnySum::C8($x), AnySum::C8($y)) => Some($body),
            _ => None,
        }
    };
}

/// Evaluates the body for a sum with real coefficients, and returns an error for a complex one.
macro_rules! dispatch_real {
    ($sum:expr, $s:ident => $body:expr) => {
        match $sum {
            AnySum::R1($s) => $body,
            AnySum::R2($s) => $body,
            AnySum::R4($s) => $body,
            AnySum::R8($s) => $body,
            _ => return Err(value_error("this takes a sum with real coefficients")),
        }
    };
}

/// A sum of one width and kind of coefficient as the sum of any.
trait Wrap {
    fn wrap(self) -> AnySum;
}

/// The sum with real coefficients and strings of `W` words, if it is one.
trait AsReal<const W: usize> {
    fn as_real(&self) -> Option<&CoreSum<W>>;
}

macro_rules! kinds {
    ($(($real:ident, $complex:ident, $width:literal)),*) => {$(
        impl Wrap for CoreSum<$width> {
            fn wrap(self) -> AnySum {
                AnySum::$real(self)
            }
        }
        impl Wrap for CoreSum<$width, Complex64> {
            fn wrap(self) -> AnySum {
                AnySum::$complex(self)
            }
        }
        impl AsReal<$width> for AnySum {
            fn as_real(&self) -> Option<&CoreSum<$width>> {
                match self {
                    AnySum::$real(sum) => Some(sum),
                    _ => None,
                }
            }
        }
    )*};
}
kinds!((R1, C1, 1), (R2, C2, 2), (R4, C4, 4), (R8, C8, 8));

// After the macros, which it uses.
mod extras;

fn complex_copy<const W: usize>(sum: &CoreSum<W>) -> CoreSum<W, Complex64> {
    let mut copy = CoreSum::new(sum.num_qubits()).expect("the width fits the same qubits");
    for (string, coeff) in sum.iter() {
        copy.add_string(*string, Complex64::new(coeff, 0.0));
    }
    copy
}

/// The sum with real coefficients, or the first term whose coefficient is not real.
fn real_copy<const W: usize>(sum: &CoreSum<W, Complex64>) -> Result<CoreSum<W>, String> {
    let mut copy = CoreSum::new(sum.num_qubits()).expect("the width fits the same qubits");
    for (string, coeff) in sum.sorted_terms() {
        if coeff.im != 0.0 {
            return Err(string.label(sum.num_qubits()));
        }
        copy.add_string(string, coeff.re);
    }
    Ok(copy)
}

/// Whether `dtype` is `complex` rather than `float`.
fn is_complex_type(py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<bool> {
    if dtype.is(py.get_type::<PyFloat>()) {
        Ok(false)
    } else if dtype.is(py.get_type::<PyComplex>()) {
        Ok(true)
    } else {
        Err(value_error("dtype must be float or complex"))
    }
}

impl AnySum {
    fn new(num_qubits: usize, complex: bool) -> PyResult<Self> {
        macro_rules! make {
            ($variant:ident) => {
                AnySum::$variant(CoreSum::new(num_qubits).map_err(value_error)?)
            };
        }
        Ok(match (num_qubits, complex) {
            (0..=64, false) => make!(R1),
            (65..=128, false) => make!(R2),
            (129..=256, false) => make!(R4),
            (_, false) => make!(R8),
            (0..=64, true) => make!(C1),
            (65..=128, true) => make!(C2),
            (129..=256, true) => make!(C4),
            (_, true) => make!(C8),
        })
    }

    fn is_complex(&self) -> bool {
        matches!(
            self,
            AnySum::C1(_) | AnySum::C2(_) | AnySum::C4(_) | AnySum::C8(_)
        )
    }

    /// The sum with complex coefficients, as a copy if it has real ones.
    fn to_complex(&self) -> AnySum {
        match self {
            AnySum::R1(s) => AnySum::C1(complex_copy(s)),
            AnySum::R2(s) => AnySum::C2(complex_copy(s)),
            AnySum::R4(s) => AnySum::C4(complex_copy(s)),
            AnySum::R8(s) => AnySum::C8(complex_copy(s)),
            complex => complex.clone(),
        }
    }

    /// The sum with real coefficients, as a copy if it has complex ones that are all real.
    fn to_real(&self) -> PyResult<AnySum> {
        let not_real = |label: String| {
            value_error(format!("the coefficient of {label} has an imaginary part"))
        };
        Ok(match self {
            AnySum::C1(s) => AnySum::R1(real_copy(s).map_err(not_real)?),
            AnySum::C2(s) => AnySum::R2(real_copy(s).map_err(not_real)?),
            AnySum::C4(s) => AnySum::R4(real_copy(s).map_err(not_real)?),
            AnySum::C8(s) => AnySum::R8(real_copy(s).map_err(not_real)?),
            real => real.clone(),
        })
    }

    /// The scalar product of two sums with the same kind of coefficients.
    fn scalar_product(&self, other: &AnySum) -> Option<Number> {
        pairwise!(self, other, a, b => {
            if a.num_qubits() != b.num_qubits() {
                return None;
            }
            a.scalar_product(b).number()
        })
    }

    /// The two sums with the same kind of coefficients: complex if one of them is.
    fn alike(&self, other: &AnySum) -> (AnySum, AnySum) {
        if self.is_complex() || other.is_complex() {
            (self.to_complex(), other.to_complex())
        } else {
            (self.clone(), other.clone())
        }
    }
}

fn different_registers() -> PyErr {
    value_error("the two sums are on different numbers of qubits")
}

/// A sum of Pauli strings: an observable, or a state written in the Pauli basis.
///
/// The coefficients are real, or complex with ``dtype=complex``.
#[pyclass(module = "paulipropagation_qdk")]
#[derive(Clone)]
struct PauliSum {
    inner: AnySum,
}

#[pymethods]
impl PauliSum {
    #[new]
    #[pyo3(signature = (num_qubits, dtype=None))]
    fn new(py: Python<'_>, num_qubits: usize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let complex = match dtype {
            None => false,
            Some(dtype) => is_complex_type(py, dtype)?,
        };
        Ok(Self {
            inner: AnySum::new(num_qubits, complex)?,
        })
    }

    /// Adds `coeff` times the Pauli string with operator `paulis[i]` on qubit `qubits[i]`.
    #[pyo3(signature = (paulis, qubits, coeff=Number::Real(1.0)))]
    fn add(&mut self, paulis: &str, qubits: Vec<usize>, coeff: Number) -> PyResult<()> {
        let ops = parse_ops(paulis, &qubits)?;
        dispatch!(&mut self.inner, s => {
            s.add(&ops, Scalar::from_number(coeff)?).map_err(value_error)
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        dispatch!(&self.inner, s => s.num_qubits())
    }

    /// The type of the coefficients: `float` or `complex`.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyType> {
        if self.inner.is_complex() {
            py.get_type::<PyComplex>()
        } else {
            py.get_type::<PyFloat>()
        }
    }

    fn __len__(&self) -> usize {
        dispatch!(&self.inner, s => s.len())
    }

    /// The coefficient of the Pauli string with operator `paulis[i]` on qubit `qubits[i]`.
    fn coeff(&self, paulis: &str, qubits: Vec<usize>) -> PyResult<Number> {
        let ops = parse_ops(paulis, &qubits)?;
        let num_qubits = self.num_qubits();
        if let Some(&(q, _)) = ops.iter().find(|&&(q, _)| q >= num_qubits) {
            return Err(value_error(format!(
                "qubit {q} is out of range for {num_qubits} qubits"
            )));
        }
        Ok(dispatch!(&self.inner, s => s.coeff(&PauliString::from_ops(&ops)).number()))
    }

    /// The terms as `(label, coefficient)` pairs in a fixed order, where `label[q]` is the
    /// operator on qubit `q`.
    fn terms(&self) -> Vec<(String, Number)> {
        dispatch!(&self.inner, s => {
            let n = s.num_qubits();
            s.sorted_terms().into_iter().map(|(p, c)| (p.label(n), c.number())).collect()
        })
    }

    /// The expectation value in the state where every qubit is 0.
    fn overlap_with_zero(&self) -> Number {
        dispatch!(&self.inner, s => s.overlap_with_zero().number())
    }

    /// The expectation value in the state where every qubit is in the plus state.
    fn overlap_with_plus(&self) -> Number {
        dispatch!(&self.inner, s => s.overlap_with_plus().number())
    }

    /// The expectation value in the computational basis state whose qubits in `ones` are 1
    /// and whose other qubits are 0.
    fn overlap_with_computational(&self, ones: Vec<usize>) -> PyResult<Number> {
        dispatch!(&self.inner, s => {
            s.overlap_with_computational(&ones).map(Scalar::number).map_err(value_error)
        })
    }

    /// The expectation value in the maximally mixed state: the coefficient of the identity.
    fn overlap_with_max_mixed(&self) -> Number {
        dispatch!(&self.inner, s => s.overlap_with_max_mixed().number())
    }

    /// The sum over the Pauli strings of the products of their coefficients in the two sums,
    /// which is `Tr[A B] / 2^n`. Complex coefficients are not conjugated.
    fn scalar_product(&self, other: &PauliSum) -> PyResult<Number> {
        let product = if self.inner.is_complex() == other.inner.is_complex() {
            self.inner.scalar_product(&other.inner)
        } else {
            self.inner
                .to_complex()
                .scalar_product(&other.inner.to_complex())
        };
        product.ok_or_else(|| {
            value_error(format!(
                "the sums are on {} and {} qubits",
                self.num_qubits(),
                other.num_qubits()
            ))
        })
    }

    /// The expectation value `Tr[state O]` in a state given as a Pauli sum.
    fn overlap_with_pauli_sum(&self, state: &PauliSum) -> PyResult<Number> {
        Ok(self
            .scalar_product(state)?
            .scale(2f64.powi(self.num_qubits() as i32)))
    }

    /// The square root of the sum of the squared absolute values of the coefficients.
    fn norm(&self) -> f64 {
        dispatch!(&self.inner, s => s.norm_sqr()).sqrt()
    }

    /// The largest absolute value of a coefficient, zero for the empty sum.
    fn max_abs_coeff(&self) -> f64 {
        dispatch!(&self.inner, s => s.max_abs_coeff())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    /// A copy whose coefficients have the type `dtype`: `float` or `complex`. A coefficient
    /// with an imaginary part has no `float` value, which is an error.
    fn astype(&self, py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = if is_complex_type(py, dtype)? {
            self.inner.to_complex()
        } else {
            self.inner.to_real()?
        };
        Ok(Self { inner })
    }

    /// `self + factor * other`, complex if one of the two is.
    fn plus(&self, other: &PauliSum, factor: f64) -> PyResult<Self> {
        let (mut total, other) = self.inner.alike(&other.inner);
        let added = pairwise!(&mut total, &other, a, b => {
            a.add_scaled(b, Coefficient::from_real(factor))
        });
        added
            .ok_or_else(different_registers)?
            .map_err(value_error)?;
        Ok(Self { inner: total })
    }

    fn __add__(&self, other: &PauliSum) -> PyResult<Self> {
        self.plus(other, 1.0)
    }

    fn __sub__(&self, other: &PauliSum) -> PyResult<Self> {
        self.plus(other, -1.0)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.__mul__(Number::Real(-1.0))
    }

    /// The sum with every coefficient multiplied by a number.
    fn __mul__(&self, factor: Number) -> PyResult<Self> {
        let mut inner = match factor {
            Number::Complex(_) => self.inner.to_complex(),
            Number::Real(_) => self.inner.clone(),
        };
        dispatch!(&mut inner, s => s.scale(Scalar::from_number(factor)?));
        Ok(Self { inner })
    }

    fn __rmul__(&self, factor: Number) -> PyResult<Self> {
        self.__mul__(factor)
    }

    /// The operator product `self * other`, with complex coefficients.
    fn product(&self, other: &PauliSum) -> PyResult<Self> {
        let (a, b) = self.inner.alike(&other.inner);
        let product = pairwise!(&a, &b, a, b => a.product(b).map(Wrap::wrap));
        let inner = product
            .ok_or_else(different_registers)?
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    /// The commutator `self * other - other * self`, with complex coefficients.
    fn commutator(&self, other: &PauliSum) -> PyResult<Self> {
        let (a, b) = self.inner.alike(&other.inner);
        let commutator = pairwise!(&a, &b, a, b => a.commutator(b).map(Wrap::wrap));
        let inner = commutator
            .ok_or_else(different_registers)?
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    /// The sum with the Pauli strings that are translations of each other merged into one
    /// of them, the one with its operators on the lowest qubits.
    ///
    /// Without arguments the qubits are a ring. With `nx` and `ny` they are a torus of `nx`
    /// columns and `ny` rows, on which the qubit in column `x` of row `y` is `y * nx + x`.
    /// Where the circuit and the state are the same after a translation, the merged sum has
    /// the expectation value of the sum it was made from.
    #[pyo3(signature = (nx=None, ny=None))]
    fn translation_merge(&self, nx: Option<usize>, ny: Option<usize>) -> PyResult<Self> {
        let mut merged = self.clone();
        match (nx, ny) {
            (None, None) => dispatch!(&mut merged.inner, s => s.translation_merge()),
            (Some(nx), Some(ny)) => {
                dispatch!(&mut merged.inner, s => s.translation_merge_on_grid(nx, ny))
                    .map_err(value_error)?;
            }
            _ => return Err(value_error("a grid needs both nx and ny")),
        }
        Ok(merged)
    }

    /// The sum with every Pauli string replaced by `representative(label)`, and the terms
    /// that have become equal added up. `label[q]` is the operator on qubit `q`.
    fn symmetry_merge(&self, representative: &Bound<'_, PyAny>) -> PyResult<Self> {
        let num_qubits = self.num_qubits();
        let qubits: Vec<usize> = (0..num_qubits).collect();
        let failure = RefCell::new(None);
        let mut merged = self.clone();
        dispatch!(&mut merged.inner, s => s.merge_by(|string| {
            let image = representative
                .call1((string.label(num_qubits),))
                .and_then(|label| label.extract::<String>())
                .and_then(|label| parse_ops(&label, &qubits));
            match image {
                Ok(ops) => PauliString::from_ops(&ops),
                Err(err) => {
                    failure.borrow_mut().get_or_insert(err);
                    *string
                }
            }
        }));
        match failure.into_inner() {
            Some(err) => Err(err),
            None => Ok(merged),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliSum(num_qubits={}, terms={}, dtype={})",
            self.num_qubits(),
            self.__len__(),
            if self.inner.is_complex() {
                "complex"
            } else {
                "float"
            }
        )
    }
}

/// A circuit of gates and noise channels, in the order they act on a state.
#[pyclass(module = "paulipropagation_qdk")]
#[derive(Clone)]
struct Circuit {
    num_qubits: usize,
    gates: Vec<Gate>,
}

impl Circuit {
    fn push(&mut self, gate: Gate) -> PyResult<()> {
        gate.validate(self.num_qubits).map_err(value_error)?;
        self.gates.push(gate);
        Ok(())
    }
}

#[pymethods]
impl Circuit {
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn __len__(&self) -> usize {
        self.gates.len()
    }

    /// The number of rotations, which can split a Pauli string in two.
    #[getter]
    fn num_rotations(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::Rotation { .. }))
            .count()
    }

    /// The number of noise channels.
    #[getter]
    fn num_noise_channels(&self) -> usize {
        self.gates.iter().filter(|g| g.is_noise()).count()
    }

    fn h(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::H(q))
    }
    fn x(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::X(q))
    }
    fn y(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::Y(q))
    }
    fn z(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::Z(q))
    }
    fn s(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::S(q))
    }
    fn s_adj(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::Sdg(q))
    }
    fn sx(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::Sx(q))
    }
    fn sx_adj(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::Sxdg(q))
    }
    fn t(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::t(q))
    }
    fn t_adj(&mut self, q: usize) -> PyResult<()> {
        self.push(Gate::tdg(q))
    }
    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.push(Gate::Cx(control, target))
    }
    fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.push(Gate::Cy(control, target))
    }
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.push(Gate::Cz(control, target))
    }
    fn swap(&mut self, q0: usize, q1: usize) -> PyResult<()> {
        self.push(Gate::Swap(q0, q1))
    }
    /// The Toffoli gate, added as Clifford and T gates.
    fn ccx(&mut self, control0: usize, control1: usize, target: usize) -> PyResult<()> {
        if control0 == control1 || control0 == target || control1 == target {
            return Err(value_error("the qubits of a CCX gate must be distinct"));
        }
        for gate in Gate::ccx(control0, control1, target) {
            self.push(gate)?;
        }
        Ok(())
    }
    fn rx(&mut self, theta: f64, q: usize) -> PyResult<()> {
        self.push(Gate::rx(theta, q))
    }
    fn ry(&mut self, theta: f64, q: usize) -> PyResult<()> {
        self.push(Gate::ry(theta, q))
    }
    fn rz(&mut self, theta: f64, q: usize) -> PyResult<()> {
        self.push(Gate::rz(theta, q))
    }
    fn rxx(&mut self, theta: f64, q0: usize, q1: usize) -> PyResult<()> {
        self.push(Gate::rxx(theta, q0, q1))
    }
    fn ryy(&mut self, theta: f64, q0: usize, q1: usize) -> PyResult<()> {
        self.push(Gate::ryy(theta, q0, q1))
    }
    fn rzz(&mut self, theta: f64, q0: usize, q1: usize) -> PyResult<()> {
        self.push(Gate::rzz(theta, q0, q1))
    }
    /// The rotation `exp(-i * theta / 2 * P)` for the Pauli string `P` with operator
    /// `paulis[i]` on qubit `qubits[i]`.
    fn rotation(&mut self, paulis: &str, qubits: Vec<usize>, theta: f64) -> PyResult<()> {
        let ops = parse_ops(paulis, &qubits)?;
        // Validated before identities are dropped, so that every qubit given is checked.
        Gate::Rotation {
            generator: ops.clone(),
            theta,
        }
        .validate(self.num_qubits)
        .map_err(value_error)?;
        self.push(Gate::rotation(&ops, theta))
    }

    /// A random Pauli error: the string `errors[k][0]` with probability `errors[k][1]`, and
    /// nothing otherwise. Character `i` of an error string acts on `qubits[i]`.
    fn pauli_channel(&mut self, qubits: Vec<usize>, errors: Vec<(String, f64)>) -> PyResult<()> {
        let errors: Vec<(Vec<PauliOp>, f64)> = errors
            .iter()
            .map(|(paulis, probability)| {
                let ops = parse_ops(paulis, &qubits)?;
                Ok((ops.into_iter().map(|(_, op)| op).collect(), *probability))
            })
            .collect::<PyResult<_>>()?;
        self.push(Gate::pauli_channel(&qubits, &errors).map_err(value_error)?)
    }
    /// Depolarizing noise, which multiplies X, Y and Z on the qubit by `1 - strength`: a
    /// random X, Y or Z error with probability `strength / 4` each.
    fn depolarizing(&mut self, strength: f64, q: usize) -> PyResult<()> {
        self.push(Gate::depolarizing(strength, q).map_err(value_error)?)
    }
    /// Dephasing, which multiplies X and Y on the qubit by `1 - strength`: a random Z error
    /// with probability `strength / 2`.
    fn dephasing(&mut self, strength: f64, q: usize) -> PyResult<()> {
        self.push(Gate::pauli_z_noise(strength, q).map_err(value_error)?)
    }
    /// A random X error with probability `strength / 2`.
    fn pauli_x_noise(&mut self, strength: f64, q: usize) -> PyResult<()> {
        self.push(Gate::pauli_x_noise(strength, q).map_err(value_error)?)
    }
    /// A random Y error with probability `strength / 2`.
    fn pauli_y_noise(&mut self, strength: f64, q: usize) -> PyResult<()> {
        self.push(Gate::pauli_y_noise(strength, q).map_err(value_error)?)
    }
    /// A random Z error with probability `strength / 2`.
    fn pauli_z_noise(&mut self, strength: f64, q: usize) -> PyResult<()> {
        self.push(Gate::pauli_z_noise(strength, q).map_err(value_error)?)
    }
    /// Amplitude damping, which takes the state 1 to the state 0 with probability `gamma`.
    fn amplitude_damping(&mut self, gamma: f64, q: usize) -> PyResult<()> {
        self.push(Gate::amplitude_damping(gamma, q).map_err(value_error)?)
    }

    /// The gate with the unitary `matrix` on `qubits`, given as `2^n` rows of `2^n` numbers.
    /// `qubits[0]` is the most significant bit of a row or column index, so the matrix of
    /// independent gates on the qubits is their Kronecker product in the order of `qubits`.
    fn unitary(&mut self, qubits: Vec<usize>, matrix: Vec<Vec<Complex64>>) -> PyResult<()> {
        let flat = flatten(matrix)?;
        self.push(Gate::unitary(&qubits, &flat).map_err(value_error)?)
    }
    /// The gate or channel with the Pauli transfer `matrix` on `qubits`, as it acts on an
    /// observable, given as `4^n` rows of `4^n` numbers. The Pauli string with index `j` on
    /// the qubits becomes the sum over `i` of `matrix[i][j]` times the string with index `i`,
    /// where the `k`-th base-4 digit of an index is the operator on `qubits[k]`, with
    /// `I, X, Y, Z = 0, 1, 2, 3`.
    fn transfer_map(&mut self, qubits: Vec<usize>, matrix: Vec<Vec<f64>>) -> PyResult<()> {
        let flat = flatten(matrix)?;
        self.push(Gate::transfer_map(&qubits, &flat).map_err(value_error)?)
    }

    /// The angles of the rotations, in the order of the circuit. T gates are rotations.
    #[getter]
    fn angles(&self) -> Vec<f64> {
        let rotations = self.gates.iter().filter_map(|gate| match gate {
            Gate::Rotation { theta, .. } => Some(*theta),
            _ => None,
        });
        rotations.collect()
    }

    /// The circuit with the angles of its rotations replaced by `angles`, one for every
    /// rotation in the order of the circuit.
    fn with_angles(&self, angles: Vec<f64>) -> PyResult<Self> {
        if angles.len() != self.num_rotations() {
            return Err(value_error(format!(
                "{} angles were given for {} rotations",
                angles.len(),
                self.num_rotations()
            )));
        }
        let mut circuit = Self::new(self.num_qubits);
        let mut angles = angles.into_iter();
        for gate in &self.gates {
            circuit.push(match gate {
                Gate::Rotation { generator, .. } => Gate::Rotation {
                    generator: generator.clone(),
                    theta: angles.next().expect("one for every rotation"),
                },
                other => other.clone(),
            })?;
        }
        Ok(circuit)
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, gates={})",
            self.num_qubits,
            self.gates.len()
        )
    }
}

/// The pool of `threads` threads, which is made when it is first asked for and kept.
fn pool_of(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(Vec::new());
    if threads == 0 {
        return Err(value_error("threads must be at least 1"));
    }
    let mut pools = POOLS.lock().expect("the lock is not poisoned");
    if let Some((_, pool)) = pools.iter().find(|(size, _)| *size == threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let pool = Arc::new(pool);
    pools.push((threads, Arc::clone(&pool)));
    Ok(pool)
}

/// Runs `propagate` on `sum` with the threads of `pool`, handing it a rule that asks
/// `callback(label, coefficient)` about a term if there is a callback.
fn run<const W: usize, C: Scalar>(
    py: Python<'_>,
    sum: &mut CoreSum<W, C>,
    callback: Option<&Py<PyAny>>,
    pool: Option<&rayon::ThreadPool>,
    propagate: impl FnOnce(
            &mut CoreSum<W, C>,
            Option<&CustomTruncation<'_, W, C>>,
        ) -> Result<(), pauliprop::Error>
        + Send,
) -> PyResult<()> {
    let num_qubits = sum.num_qubits();
    let failure: Mutex<Option<PyErr>> = Mutex::new(None);
    let ask = |string: &PauliString<W>, coeff: C| -> bool {
        let Some(callback) = callback else {
            return false;
        };
        let mut failure = failure.lock().expect("the lock is not poisoned");
        // After a failure everything is dropped, so that the propagation ends quickly.
        if failure.is_some() {
            return true;
        }
        let answer = Python::with_gil(|py| {
            callback
                .call1(py, (string.label(num_qubits), coeff.number()))
                .and_then(|answer| answer.is_truthy(py))
        });
        answer.unwrap_or_else(|err| {
            *failure = Some(err);
            true
        })
    };
    let custom: Option<&CustomTruncation<'_, W, C>> = callback.is_some().then_some(&ask);

    py.allow_threads(|| match pool {
        Some(pool) => pool.install(|| propagate(sum, custom)),
        None => propagate(sum, custom),
    })
    .map_err(value_error)?;
    match failure.into_inner().expect("the lock is not poisoned") {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Returns `U† O U` for the unitary `U` of `circuit` and the observable `O`. If the circuit
/// has noise channels, the result gives the expectation values of the noisy circuit.
///
/// With `heisenberg=False` the sum is a state in the Pauli basis and the result is the state
/// after the circuit, `U rho U†`.
///
/// After every gate the terms that a truncation selects are dropped:
///
/// - `max_weight`: more non-identity operators than this.
/// - `min_abs_coeff`: a coefficient below this in absolute value.
/// - `min_rel_coeff`: a coefficient below this fraction of the largest in the sum.
/// - `max_freq`: more rotations that split the Pauli string on the way than this.
/// - `max_sins`: more sine branches of rotations on the way than this.
/// - `weight_damping=(gamma, threshold)`: `|coeff| * 10**(-gamma * weight) < threshold`.
/// - `custom_truncation`: a function of the label and the coefficient that returns true.
///
/// A Clifford gate changes no coefficient but its sign, so after one only `max_weight`,
/// `weight_damping` and `custom_truncation` can drop a term.
///
/// With `max_terms` the sum never has more terms than that after a gate: of a sum that the
/// other truncations leave larger, the `max_terms` terms with the largest coefficients in
/// absolute value are kept. This bounds the memory and the time of a propagation whatever
/// the circuit. It cannot be combined with `max_freq` or `max_sins`, and close to
/// `max_terms` the gates are applied on one thread.
///
/// `threads` is the number of threads to use, by default one per processor. The result does
/// not depend on it.
#[pyfunction]
#[pyo3(signature = (
    circuit,
    observable,
    *,
    max_weight=None,
    min_abs_coeff=1e-10,
    min_rel_coeff=None,
    max_freq=None,
    max_sins=None,
    weight_damping=None,
    custom_truncation=None,
    max_terms=None,
    heisenberg=true,
    threads=None,
))]
#[allow(clippy::too_many_arguments)]
fn propagate(
    py: Python<'_>,
    circuit: &Circuit,
    observable: &PauliSum,
    max_weight: Option<u32>,
    min_abs_coeff: f64,
    min_rel_coeff: Option<f64>,
    max_freq: Option<u32>,
    max_sins: Option<u32>,
    weight_damping: Option<(f64, f64)>,
    custom_truncation: Option<Py<PyAny>>,
    max_terms: Option<usize>,
    heisenberg: bool,
    threads: Option<usize>,
) -> PyResult<PauliSum> {
    check_register(circuit, observable)?;
    let truncation = truncation_of(
        max_weight,
        min_abs_coeff,
        min_rel_coeff,
        max_freq,
        max_sins,
        weight_damping,
    )?;
    check_callable(py, custom_truncation.as_ref())?;
    let picture = picture_of(heisenberg);

    let mut result = observable.clone();
    let gates = &circuit.gates;
    let callback = custom_truncation.as_ref();
    let pool = threads.map(pool_of).transpose()?;
    let pool = pool.as_deref();
    dispatch!(&mut result.inner, s => run(py, s, callback, pool, |sum, custom| {
        match max_terms {
            Some(max_terms) => {
                pauliprop::propagate_capped(gates, sum, &truncation, picture, custom, max_terms)
            }
            None => pauliprop::propagate_with(gates, sum, &truncation, picture, custom),
        }
    }))?;
    Ok(result)
}

fn check_register(circuit: &Circuit, sum: &PauliSum) -> PyResult<()> {
    if circuit.num_qubits > sum.num_qubits() {
        return Err(value_error(format!(
            "the circuit has {} qubits but the observable has {}",
            circuit.num_qubits,
            sum.num_qubits()
        )));
    }
    Ok(())
}

fn check_callable(py: Python<'_>, callback: Option<&Py<PyAny>>) -> PyResult<()> {
    match callback {
        Some(callback) if !callback.bind(py).is_callable() => {
            Err(PyTypeError::new_err("custom_truncation must be callable"))
        }
        _ => Ok(()),
    }
}

/// The truncation with the given options, after checking that they are thresholds.
fn truncation_of(
    max_weight: Option<u32>,
    min_abs_coeff: f64,
    min_rel_coeff: Option<f64>,
    max_freq: Option<u32>,
    max_sins: Option<u32>,
    weight_damping: Option<(f64, f64)>,
) -> PyResult<Truncation> {
    let not_a_threshold = |value: f64| value.is_nan() || value < 0.0;
    if not_a_threshold(min_abs_coeff) {
        return Err(value_error("min_abs_coeff must be zero or positive"));
    }
    if min_rel_coeff.is_some_and(not_a_threshold) {
        return Err(value_error("min_rel_coeff must be zero or positive"));
    }
    if weight_damping.is_some_and(|(gamma, threshold)| gamma.is_nan() || not_a_threshold(threshold))
    {
        return Err(value_error(
            "weight_damping takes a damping rate and a threshold that is zero or positive",
        ));
    }
    Ok(Truncation {
        max_weight,
        min_abs_coeff,
        min_rel_coeff,
        max_freq,
        max_sins,
        weight_damping: weight_damping.map(|(gamma, min_abs_coeff)| WeightDamping {
            gamma,
            min_abs_coeff,
        }),
    })
}

fn picture_of(heisenberg: bool) -> Picture {
    if heisenberg {
        Picture::Heisenberg
    } else {
        Picture::Schrodinger
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PauliSum>()?;
    m.add_class::<Circuit>()?;
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    extras::register(m)?;
    Ok(())
}
