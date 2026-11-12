//! Python bindings for the Pauli propagation engine.

use pauliprop::{
    Coefficient, Complex64, CustomTruncation, Gate, PauliOp, PauliString, PauliSum as CoreSum,
    Picture, Truncation, WeightDamping,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyFloat, PyType};
use std::sync::Mutex;

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
        macro_rules! same {
            ($($variant:ident),*) => {
                match (self, other) {
                    $((AnySum::$variant(a), AnySum::$variant(b)) => {
                        (a.num_qubits() == b.num_qubits()).then(|| a.scalar_product(b).number())
                    })*
                    _ => None,
                }
            };
        }
        same!(R1, R2, R4, R8, C1, C2, C4, C8)
    }
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

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, gates={})",
            self.num_qubits,
            self.gates.len()
        )
    }
}

/// Propagates `sum` through the gates, asking `callback(label, coefficient)` about every
/// term after every gate if there is one.
fn run<const W: usize, C: Scalar>(
    py: Python<'_>,
    gates: &[Gate],
    sum: &mut CoreSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    callback: Option<&Py<PyAny>>,
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

    py.allow_threads(|| pauliprop::propagate_with(gates, sum, truncation, picture, custom))
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
    heisenberg=true,
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
    heisenberg: bool,
) -> PyResult<PauliSum> {
    if circuit.num_qubits > observable.num_qubits() {
        return Err(value_error(format!(
            "the circuit has {} qubits but the observable has {}",
            circuit.num_qubits,
            observable.num_qubits()
        )));
    }
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
    if let Some(callback) = &custom_truncation {
        if !callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err("custom_truncation must be callable"));
        }
    }
    let truncation = Truncation {
        max_weight,
        min_abs_coeff,
        min_rel_coeff,
        max_freq,
        max_sins,
        weight_damping: weight_damping.map(|(gamma, min_abs_coeff)| WeightDamping {
            gamma,
            min_abs_coeff,
        }),
    };
    let picture = if heisenberg {
        Picture::Heisenberg
    } else {
        Picture::Schrodinger
    };

    let mut result = observable.clone();
    let gates = &circuit.gates;
    let callback = custom_truncation.as_ref();
    dispatch!(&mut result.inner, s => run(py, gates, s, &truncation, picture, callback))?;
    Ok(result)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PauliSum>()?;
    m.add_class::<Circuit>()?;
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    Ok(())
}
