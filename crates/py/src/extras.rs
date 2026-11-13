//! Gradients, random sampling and surrogates.

use super::{
    check_callable, check_register, picture_of, pool_of, run, truncation_of, value_error, AnySum,
    AsReal, Circuit, PauliSum, Wrap,
};
use pauliprop::{PathLimits, PauliString, Resampling, Rng, State, Surrogate as CoreSurrogate};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Mutex;

/// The state an expectation value is taken in, as Python gives it.
pub(crate) enum StateArg {
    Zero,
    Plus,
    MaxMixed,
    Computational(Vec<usize>),
    Sum(PauliSum),
}

impl<'py> FromPyObject<'py> for StateArg {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(name) = ob.extract::<String>() {
            return match name.as_str() {
                "zero" => Ok(StateArg::Zero),
                "plus" => Ok(StateArg::Plus),
                "mixed" => Ok(StateArg::MaxMixed),
                _ => Err(value_error("a state by name is 'zero', 'plus' or 'mixed'")),
            };
        }
        if let Ok(sum) = ob.extract::<PyRef<'_, PauliSum>>() {
            return Ok(StateArg::Sum(sum.clone()));
        }
        ob.extract::<Vec<usize>>()
            .map(StateArg::Computational)
            .map_err(|_| {
                PyTypeError::new_err(
                    "state must be 'zero', 'plus', 'mixed', the qubits that are 1, or a PauliSum",
                )
            })
    }
}

/// Calls `f` with the state for sums of width `W`.
fn with_state<const W: usize, T>(
    state: &StateArg,
    f: impl FnOnce(&State<'_, W>) -> T,
) -> PyResult<T>
where
    AnySum: AsReal<W>,
{
    Ok(match state {
        StateArg::Zero => f(&State::Zero),
        StateArg::Plus => f(&State::Plus),
        StateArg::MaxMixed => f(&State::MaxMixed),
        StateArg::Computational(ones) => f(&State::Computational(ones)),
        StateArg::Sum(sum) => match sum.inner.as_real() {
            Some(sum) => f(&State::Sum(sum)),
            None => {
                return Err(value_error(
                    "the state must have real coefficients and the qubits of the observable",
                ))
            }
        },
    })
}

fn rng_of(seed: Option<u64>) -> Rng {
    seed.map_or_else(Rng::from_entropy, Rng::new)
}

fn method_of(name: Option<&str>) -> PyResult<Option<Resampling>> {
    match name {
        None => Ok(None),
        Some("multinomial") => Ok(Some(Resampling::Multinomial)),
        Some("systematic") => Ok(Some(Resampling::systematic())),
        Some("semideterministic") => Ok(Some(Resampling::SemiDeterministic)),
        Some(_) => Err(value_error(
            "method must be 'multinomial', 'systematic' or 'semideterministic'",
        )),
    }
}

/// A rule that asks `callback(label)` about a Pauli string, and keeps the first failure.
struct LabelRule<'a> {
    callback: &'a Py<PyAny>,
    num_qubits: usize,
    failure: Mutex<Option<PyErr>>,
}

impl<'a> LabelRule<'a> {
    fn new(
        py: Python<'_>,
        callback: Option<&'a Py<PyAny>>,
        num_qubits: usize,
    ) -> PyResult<Option<Self>> {
        check_callable(py, callback)?;
        Ok(callback.map(|callback| Self {
            callback,
            num_qubits,
            failure: Mutex::new(None),
        }))
    }

    fn drops<const W: usize>(&self, string: &PauliString<W>) -> bool {
        let mut failure = self.failure.lock().expect("the lock is not poisoned");
        if failure.is_some() {
            return true;
        }
        let answer = Python::with_gil(|py| {
            self.callback
                .call1(py, (string.label(self.num_qubits),))
                .and_then(|answer| answer.is_truthy(py))
        });
        answer.unwrap_or_else(|err| {
            *failure = Some(err);
            true
        })
    }
}

fn finish(rule: Option<LabelRule<'_>>) -> PyResult<()> {
    match rule.and_then(|rule| rule.failure.into_inner().expect("the lock is not poisoned")) {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// The expectation value of `observable` after `circuit` has acted on `state`, and its
/// derivatives with respect to the angles of the rotations, as `(value, derivatives)`.
///
/// There is one derivative for every rotation, in the order of `circuit.angles`. All of
/// them together take one pass back through the circuit and one pass forward. `state` is
/// `'zero'`, `'plus'`, `'mixed'`, the list of the qubits that are 1 in a computational basis
/// state, or a `PauliSum`.
///
/// Without truncation the gradient is exact. With truncation it is close to the gradient of
/// the truncated expectation value. The circuit must not have noise channels, and
/// `max_freq` and `max_sins` are not available.
#[pyfunction]
#[pyo3(signature = (
    circuit,
    observable,
    *,
    state=StateArg::Zero,
    max_weight=None,
    min_abs_coeff=1e-10,
    min_rel_coeff=None,
    weight_damping=None,
    threads=None,
))]
#[allow(clippy::too_many_arguments)]
fn gradient(
    py: Python<'_>,
    circuit: &Circuit,
    observable: &PauliSum,
    state: StateArg,
    max_weight: Option<u32>,
    min_abs_coeff: f64,
    min_rel_coeff: Option<f64>,
    weight_damping: Option<(f64, f64)>,
    threads: Option<usize>,
) -> PyResult<(f64, Vec<f64>)> {
    check_register(circuit, observable)?;
    let truncation = truncation_of(
        max_weight,
        min_abs_coeff,
        min_rel_coeff,
        None,
        None,
        weight_damping,
    )?;
    let pool = threads.map(pool_of).transpose()?;
    let pool = pool.as_deref();
    let gates = &circuit.gates;
    dispatch_real!(&observable.inner, s => {
        with_state(&state, |state| {
            let compute = || pauliprop::gradient(gates, s, &truncation, state);
            py.allow_threads(|| match pool {
                Some(pool) => pool.install(compute),
                None => compute(),
            })
        })?
        .map_err(value_error)
    })
}

/// A sum of about `target_size` terms that is `sum` on average: terms are dropped at
/// random and the others are made larger.
///
/// `method` is `'semideterministic'` (the default: the largest terms are kept as they are),
/// `'systematic'` or `'multinomial'`. With `squared=True` the terms are chosen by their
/// squared coefficients, the sum of the squares is kept instead, and the method is
/// `'multinomial'` unless given. `seed` makes the result reproducible.
#[pyfunction]
#[pyo3(signature = (sum, target_size, *, method=None, squared=false, seed=None))]
fn resample(
    sum: &PauliSum,
    target_size: usize,
    method: Option<&str>,
    squared: bool,
    seed: Option<u64>,
) -> PyResult<PauliSum> {
    let method = method_of(method)?;
    let mut rng = rng_of(seed);
    let mut result = sum.clone();
    dispatch!(&mut result.inner, s => pauliprop::resample(s, target_size, method, squared, &mut rng))
        .map_err(value_error)?;
    Ok(result)
}

/// The mean of `samples` random paths of every term of `sum` through `circuit`: where a
/// rotation would split a term in two, one of the two is kept at random.
///
/// The result is the propagated sum on average, without truncation, and it never holds more
/// than `samples` terms per term of `sum`. With `squared=True` a path takes a branch with
/// the probability of its squared factor and keeps its coefficient, so the coefficient of a
/// term is spread over the Pauli strings in proportion to their squared factors.
///
/// The circuit must consist of Clifford gates and rotations.
#[pyfunction]
#[pyo3(signature = (circuit, sum, samples, *, squared=false, heisenberg=true, seed=None))]
fn mcsample(
    py: Python<'_>,
    circuit: &Circuit,
    sum: &PauliSum,
    samples: usize,
    squared: bool,
    heisenberg: bool,
    seed: Option<u64>,
) -> PyResult<PauliSum> {
    check_register(circuit, sum)?;
    let mut rng = rng_of(seed);
    let picture = picture_of(heisenberg);
    let gates = &circuit.gates;
    let inner = dispatch!(&sum.inner, s => {
        py.allow_threads(|| pauliprop::sample_sum(gates, s, samples, picture, squared, &mut rng))
            .map(Wrap::wrap)
    });
    Ok(PauliSum {
        inner: inner.map_err(value_error)?,
    })
}

/// Propagates like `propagate`, and whenever a gate leaves the sum with more than
/// `max_size` terms, resamples it down to about `resampling_size` terms (by default half of
/// `max_size`), see `resample`.
///
/// The sum never holds more than twice `max_size` terms, whatever the circuit, and the
/// result is the truncated propagation on average. `max_freq` and `max_sins` are not
/// available. The gates are applied one at a time, on one thread.
#[pyfunction]
#[pyo3(signature = (
    circuit,
    observable,
    *,
    max_size,
    resampling_size=None,
    method=None,
    squared=false,
    max_weight=None,
    min_abs_coeff=1e-10,
    min_rel_coeff=None,
    weight_damping=None,
    custom_truncation=None,
    heisenberg=true,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
fn mcpropagate(
    py: Python<'_>,
    circuit: &Circuit,
    observable: &PauliSum,
    max_size: usize,
    resampling_size: Option<usize>,
    method: Option<&str>,
    squared: bool,
    max_weight: Option<u32>,
    min_abs_coeff: f64,
    min_rel_coeff: Option<f64>,
    weight_damping: Option<(f64, f64)>,
    custom_truncation: Option<Py<PyAny>>,
    heisenberg: bool,
    seed: Option<u64>,
) -> PyResult<PauliSum> {
    check_register(circuit, observable)?;
    let truncation = truncation_of(
        max_weight,
        min_abs_coeff,
        min_rel_coeff,
        None,
        None,
        weight_damping,
    )?;
    check_callable(py, custom_truncation.as_ref())?;
    let method = method_of(method)?;
    let resampling_size = resampling_size.unwrap_or(max_size.div_ceil(2));
    let picture = picture_of(heisenberg);
    let mut rng = rng_of(seed);

    let mut result = observable.clone();
    let gates = &circuit.gates;
    let callback = custom_truncation.as_ref();
    dispatch!(&mut result.inner, s => run(py, s, callback, None, |sum, custom| {
        pauliprop::propagate_resampling(
            gates, sum, &truncation, picture, custom, max_size, resampling_size, method, squared, &mut rng,
        )
    }))?;
    Ok(result)
}

/// An estimate of the mean squared error that a truncation makes in the expectation value
/// of `observable`, which must be a single Pauli string, after `circuit`.
///
/// The error is averaged over the angles of the rotations: the angle of every rotation is
/// taken as uniformly distributed in `[-thetas, thetas]`, whatever angle the circuit gives
/// it. `thetas` is one number for all rotations, pi by default, or one for every rotation
/// in the order of `circuit.angles`. The estimate is made from `samples` random paths.
///
/// `max_weight`, `max_freq` and `max_sins` are the truncations of `propagate`, and
/// `custom_truncation(label)` drops a path where it returns true. A truncation by the size
/// of coefficients has no error of this kind. The circuit must consist of Clifford gates
/// and rotations.
#[pyfunction]
#[pyo3(signature = (
    circuit,
    observable,
    samples,
    *,
    thetas=None,
    max_weight=None,
    max_freq=None,
    max_sins=None,
    custom_truncation=None,
    state=StateArg::Zero,
    heisenberg=true,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
fn estimate_mse(
    py: Python<'_>,
    circuit: &Circuit,
    observable: &PauliSum,
    samples: usize,
    thetas: Option<&Bound<'_, PyAny>>,
    max_weight: Option<u32>,
    max_freq: Option<u32>,
    max_sins: Option<u32>,
    custom_truncation: Option<Py<PyAny>>,
    state: StateArg,
    heisenberg: bool,
    seed: Option<u64>,
) -> PyResult<f64> {
    check_register(circuit, observable)?;
    let radii = match thetas {
        None => vec![std::f64::consts::PI; circuit.num_rotations()],
        Some(thetas) => match thetas.extract::<f64>() {
            Ok(radius) => vec![radius; circuit.num_rotations()],
            Err(_) => thetas.extract::<Vec<f64>>()?,
        },
    };
    let limits = PathLimits {
        max_weight,
        max_freq,
        max_sins,
    };
    let num_qubits = observable.num_qubits();
    let rule = LabelRule::new(py, custom_truncation.as_ref(), num_qubits)?;
    let picture = picture_of(heisenberg);
    let mut rng = rng_of(seed);
    let gates = &circuit.gates;
    let estimate = dispatch_real!(&observable.inner, s => {
        let terms = s.sorted_terms();
        let [(string, coeff)] = terms[..] else {
            return Err(value_error("the observable must be a single Pauli string"));
        };
        let drops = |string: &PauliString<_>| rule.as_ref().is_some_and(|rule| rule.drops(string));
        let custom: Option<&(dyn Fn(&PauliString<_>) -> bool + Sync)> =
            rule.is_some().then_some(&drops);
        with_state(&state, |state| {
            py.allow_threads(|| {
                pauliprop::estimate_mse(
                    gates, num_qubits, &string, coeff, samples, &radii, &limits, custom, state, picture, &mut rng,
                )
            })
        })?
    });
    finish(rule)?;
    estimate.map_err(value_error)
}

enum AnySurrogate {
    W1(CoreSurrogate<1>),
    W2(CoreSurrogate<2>),
    W4(CoreSurrogate<4>),
    W8(CoreSurrogate<8>),
}

macro_rules! each_width {
    ($surrogate:expr, $s:ident => $body:expr) => {
        match $surrogate {
            AnySurrogate::W1($s) => $body,
            AnySurrogate::W2($s) => $body,
            AnySurrogate::W4($s) => $body,
            AnySurrogate::W8($s) => $body,
        }
    };
}

/// The propagation of an observable through a circuit of Clifford gates and rotations for
/// any angles of the rotations.
///
/// Which Pauli strings a propagation reaches does not depend on the angles, nor which of
/// them `max_weight`, `max_freq`, `max_sins` and `custom_truncation(label)` drop. The
/// surrogate records once how every coefficient is made of sines and cosines, and
/// `evaluate(angles)` then gives the propagated sum for any angles, one for every rotation
/// in the order of `circuit.angles`. It is what `propagate` returns for the circuit with
/// those angles and the same truncations with `min_abs_coeff=0`.
#[pyclass(module = "paulipropagation_qdk")]
struct Surrogate {
    inner: AnySurrogate,
}

#[pymethods]
impl Surrogate {
    #[new]
    #[pyo3(signature = (
        circuit,
        observable,
        *,
        max_weight=None,
        max_freq=None,
        max_sins=None,
        custom_truncation=None,
    ))]
    fn new(
        py: Python<'_>,
        circuit: &Circuit,
        observable: &PauliSum,
        max_weight: Option<u32>,
        max_freq: Option<u32>,
        max_sins: Option<u32>,
        custom_truncation: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        check_register(circuit, observable)?;
        let limits = PathLimits {
            max_weight,
            max_freq,
            max_sins,
        };
        let rule = LabelRule::new(py, custom_truncation.as_ref(), observable.num_qubits())?;
        let gates = &circuit.gates;
        macro_rules! build {
            ($sum:expr, $width:ident) => {{
                let drops =
                    |string: &PauliString<_>| rule.as_ref().is_some_and(|rule| rule.drops(string));
                let custom: Option<&dyn Fn(&PauliString<_>) -> bool> =
                    rule.is_some().then_some(&drops);
                CoreSurrogate::new(gates, $sum, &limits, custom).map(AnySurrogate::$width)
            }};
        }
        let inner = match &observable.inner {
            AnySum::R1(sum) => build!(sum, W1),
            AnySum::R2(sum) => build!(sum, W2),
            AnySum::R4(sum) => build!(sum, W4),
            AnySum::R8(sum) => build!(sum, W8),
            _ => {
                return Err(value_error(
                    "a surrogate takes a sum with real coefficients",
                ))
            }
        };
        finish(rule)?;
        Ok(Self {
            inner: inner.map_err(value_error)?,
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        each_width!(&self.inner, s => s.num_qubits())
    }

    /// The number of rotations of the circuit, each of which has an angle.
    #[getter]
    fn num_parameters(&self) -> usize {
        each_width!(&self.inner, s => s.num_parameters())
    }

    /// The number of sums of a cosine and a sine part that an evaluation computes.
    #[getter]
    fn num_nodes(&self) -> usize {
        each_width!(&self.inner, s => s.num_nodes())
    }

    /// The number of terms of the propagated sum.
    fn __len__(&self) -> usize {
        each_width!(&self.inner, s => s.len())
    }

    /// The propagated sum for the given angles.
    fn evaluate(&self, py: Python<'_>, angles: Vec<f64>) -> PyResult<PauliSum> {
        let inner = each_width!(&self.inner, s => {
            py.allow_threads(|| s.evaluate(&angles)).map(Wrap::wrap)
        });
        Ok(PauliSum {
            inner: inner.map_err(value_error)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Surrogate(num_qubits={}, parameters={}, terms={}, nodes={})",
            self.num_qubits(),
            self.num_parameters(),
            self.__len__(),
            self.num_nodes()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Surrogate>()?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(mcsample, m)?)?;
    m.add_function(wrap_pyfunction!(mcpropagate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_mse, m)?)?;
    Ok(())
}
