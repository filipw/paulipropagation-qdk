//! Python bindings for the Pauli propagation engine.

use pauliprop::{Gate, PauliOp, PauliString, PauliSum as CoreSum, Truncation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
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

/// A Pauli sum whose strings are as wide as its number of qubits needs.
#[derive(Clone)]
enum AnySum {
    W1(CoreSum<1>),
    W2(CoreSum<2>),
    W4(CoreSum<4>),
    W8(CoreSum<8>),
}

macro_rules! dispatch {
    ($sum:expr, $s:ident => $body:expr) => {
        match $sum {
            AnySum::W1($s) => $body,
            AnySum::W2($s) => $body,
            AnySum::W4($s) => $body,
            AnySum::W8($s) => $body,
        }
    };
}

impl AnySum {
    fn new(num_qubits: usize) -> PyResult<Self> {
        Ok(match num_qubits {
            0..=64 => AnySum::W1(CoreSum::new(num_qubits).map_err(value_error)?),
            65..=128 => AnySum::W2(CoreSum::new(num_qubits).map_err(value_error)?),
            129..=256 => AnySum::W4(CoreSum::new(num_qubits).map_err(value_error)?),
            _ => AnySum::W8(CoreSum::new(num_qubits).map_err(value_error)?),
        })
    }
}

/// An observable as a sum of Pauli strings with real coefficients.
#[pyclass(module = "paulipropagation_qdk")]
#[derive(Clone)]
struct PauliSum {
    inner: AnySum,
}

#[pymethods]
impl PauliSum {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: AnySum::new(num_qubits)?,
        })
    }

    /// Adds `coeff` times the Pauli string with operator `paulis[i]` on qubit `qubits[i]`.
    #[pyo3(signature = (paulis, qubits, coeff=1.0))]
    fn add(&mut self, paulis: &str, qubits: Vec<usize>, coeff: f64) -> PyResult<()> {
        let ops = parse_ops(paulis, &qubits)?;
        dispatch!(&mut self.inner, s => s.add(&ops, coeff)).map_err(value_error)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        dispatch!(&self.inner, s => s.num_qubits())
    }

    fn __len__(&self) -> usize {
        dispatch!(&self.inner, s => s.len())
    }

    /// The coefficient of the Pauli string with operator `paulis[i]` on qubit `qubits[i]`.
    fn coeff(&self, paulis: &str, qubits: Vec<usize>) -> PyResult<f64> {
        let ops = parse_ops(paulis, &qubits)?;
        let num_qubits = self.num_qubits();
        if let Some(&(q, _)) = ops.iter().find(|&&(q, _)| q >= num_qubits) {
            return Err(value_error(format!(
                "qubit {q} is out of range for {num_qubits} qubits"
            )));
        }
        Ok(dispatch!(&self.inner, s => s.coeff(&PauliString::from_ops(&ops))))
    }

    /// The terms as `(label, coefficient)` pairs in a fixed order, where `label[q]` is the
    /// operator on qubit `q`.
    fn terms(&self) -> Vec<(String, f64)> {
        dispatch!(&self.inner, s => {
            let n = s.num_qubits();
            s.sorted_terms().into_iter().map(|(p, c)| (p.label(n), c)).collect()
        })
    }

    /// The expectation value in the state where every qubit is 0.
    fn overlap_with_zero(&self) -> f64 {
        dispatch!(&self.inner, s => s.overlap_with_zero())
    }

    /// The square root of the sum of the squared coefficients.
    fn norm(&self) -> f64 {
        dispatch!(&self.inner, s => s.norm_sqr()).sqrt()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliSum(num_qubits={}, terms={})",
            self.num_qubits(),
            self.__len__()
        )
    }
}

/// A circuit of Clifford gates and Pauli rotations, in the order the gates act on a state.
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

    /// The number of gates that can split a Pauli string in two.
    #[getter]
    fn num_rotations(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::Rotation { .. }))
            .count()
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

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, gates={})",
            self.num_qubits,
            self.gates.len()
        )
    }
}

/// Returns `U† O U` for the unitary `U` of `circuit` and the observable `O`.
///
/// After every gate, the terms with more than `max_weight` non-identity operators or with a
/// coefficient below `min_abs_coeff` in absolute value are dropped.
#[pyfunction]
#[pyo3(signature = (circuit, observable, *, max_weight=None, min_abs_coeff=1e-10))]
fn propagate(
    py: Python<'_>,
    circuit: &Circuit,
    observable: &PauliSum,
    max_weight: Option<u32>,
    min_abs_coeff: f64,
) -> PyResult<PauliSum> {
    if circuit.num_qubits > observable.num_qubits() {
        return Err(value_error(format!(
            "the circuit has {} qubits but the observable has {}",
            circuit.num_qubits,
            observable.num_qubits()
        )));
    }
    if min_abs_coeff.is_nan() || min_abs_coeff < 0.0 {
        return Err(value_error("min_abs_coeff must be zero or positive"));
    }
    let truncation = Truncation {
        max_weight,
        min_abs_coeff,
    };
    let mut result = observable.clone();
    let gates = &circuit.gates;
    py.allow_threads(
        || dispatch!(&mut result.inner, s => pauliprop::propagate(gates, s, &truncation)),
    )
    .map_err(value_error)?;
    Ok(result)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PauliSum>()?;
    m.add_class::<Circuit>()?;
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    Ok(())
}
