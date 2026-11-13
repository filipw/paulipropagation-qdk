//! Propagation of a Pauli sum through a circuit.

use crate::coeff::{Path, Tracked};
use crate::{engine, Coefficient, Gate, PauliString, PauliSum};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    QubitOutOfRange {
        qubit: usize,
        num_qubits: usize,
    },
    DuplicateQubit {
        qubit: usize,
    },
    TooManyQubits {
        num_qubits: usize,
        capacity: usize,
    },
    NonFiniteAngle,
    InvalidNoise(&'static str),
    InvalidMatrix(&'static str),
    /// A request that cannot be met, with the reason.
    Invalid(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::QubitOutOfRange { qubit, num_qubits } => {
                write!(f, "qubit {qubit} is out of range for {num_qubits} qubits")
            }
            Error::DuplicateQubit { qubit } => write!(f, "qubit {qubit} is given more than once"),
            Error::TooManyQubits {
                num_qubits,
                capacity,
            } => write!(
                f,
                "{num_qubits} qubits do not fit a Pauli string of {capacity} qubits"
            ),
            Error::NonFiniteAngle => write!(f, "a rotation angle is not finite"),
            Error::InvalidNoise(what) | Error::InvalidMatrix(what) | Error::Invalid(what) => {
                write!(f, "{what}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Which terms to drop after every gate.
///
/// The rules and their defaults are those of PauliPropagation.jl. A Clifford gate changes
/// neither a coefficient's size nor the rotations behind it, so after a Clifford gate terms
/// are dropped only if a rule reads the Pauli string: `max_weight`, `weight_damping` or a
/// custom rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truncation {
    /// Drop the Pauli strings with more than this many non-identity operators.
    pub max_weight: Option<u32>,
    /// Drop the terms whose coefficient is smaller than this in absolute value.
    pub min_abs_coeff: f64,
    /// Drop the terms whose coefficient is smaller than this fraction of the largest
    /// absolute coefficient in the sum after the gate.
    pub min_rel_coeff: Option<f64>,
    /// Drop the terms that came through more than this many rotations that split the
    /// Pauli string, counted from the start of the propagation. Where several ways lead to
    /// one Pauli string, the smallest count is kept.
    pub max_freq: Option<u32>,
    /// Drop the terms that took the sine branch of more than this many rotations, counted
    /// like `max_freq`. This keeps the low orders of an expansion in small angles.
    pub max_sins: Option<u32>,
    /// Drop the terms whose coefficient, damped by the weight of the Pauli string, is small.
    pub weight_damping: Option<WeightDamping>,
}

/// Drops a term with coefficient `c` and a Pauli string of weight `w` if
/// `|c| * 10^(-gamma * w) < min_abs_coeff`, as `truncatedampingcoeff` of PauliPropagation.jl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightDamping {
    pub gamma: f64,
    pub min_abs_coeff: f64,
}

impl Default for Truncation {
    /// `min_abs_coeff = 1e-10` and no other rule, as in PauliPropagation.jl.
    fn default() -> Self {
        Self {
            min_abs_coeff: 1e-10,
            ..Self::none()
        }
    }
}

impl Truncation {
    /// Keeps every term, including those with a coefficient of zero.
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_weight: None,
            min_abs_coeff: 0.0,
            min_rel_coeff: None,
            max_freq: None,
            max_sins: None,
            weight_damping: None,
        }
    }
}

/// Which way a Pauli sum goes through a circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Picture {
    /// The sum is an observable `O`, replaced by `U† O U`: the gates are applied last to
    /// first, each by its action on observables.
    #[default]
    Heisenberg,
    /// The sum is a state `rho`, replaced by `U rho U†`: the gates are applied first to
    /// last, each by its action on states.
    Schrodinger,
}

/// A rule that drops the term with the given Pauli string and coefficient if it returns `true`.
pub type CustomTruncation<'a, const W: usize, C> = dyn Fn(&PauliString<W>, C) -> bool + Sync + 'a;

/// Replaces the observable `sum` by `U† sum U`, where `U` is the unitary of `circuit`. With
/// noise channels in the circuit, this is the adjoint of the channel that the circuit applies
/// to states, so expectation values are those of the noisy circuit.
///
/// `circuit` lists the gates in the order they act on a state, so they are applied to the
/// observable last to first. After every gate the terms that `truncation` selects are dropped,
/// which makes the result an approximation unless nothing is dropped.
pub fn propagate<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &mut PauliSum<W, C>,
    truncation: &Truncation,
) -> Result<(), Error> {
    propagate_with(circuit, sum, truncation, Picture::Heisenberg, None)
}

/// Propagates `sum` through `circuit` in the given picture, see [`Picture`], dropping after
/// every gate the terms that `truncation` or `custom` selects.
///
/// In both pictures `circuit` lists the gates in the order they act on a state. The
/// expectation value of an observable `O` in the state `rho` after the circuit is
/// `2^n` times the scalar product of `O` propagated in the Heisenberg picture with `rho`, and
/// equally of `O` with `rho` propagated in the Schrödinger picture.
pub fn propagate_with<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &mut PauliSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    custom: Option<&CustomTruncation<'_, W, C>>,
) -> Result<(), Error> {
    propagate_in_zones(circuit, sum, truncation, picture, custom, &Zones::default())
}

/// How the terms of a sum are spread over zones, which are worked on by different threads.
///
/// The result of a propagation does not depend on the zones. They only decide how much of
/// the work can be done at the same time, and how much it costs to hand it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zones {
    /// The number of zones doubles when the sum has more than this many terms per zone.
    pub terms: usize,
    /// The largest number of zones. It is rounded down to a power of two.
    pub max: usize,
}

impl Default for Zones {
    /// A single zone on one thread, and otherwise enough zones for all the threads of the
    /// current rayon pool to find work in a large sum.
    fn default() -> Self {
        let threads = rayon::current_num_threads();
        Self {
            terms: 1024,
            max: if threads == 1 { 1 } else { 8 * threads },
        }
    }
}

impl Zones {
    /// A single zone, which is the whole sum on the thread of the caller. It is the fastest
    /// choice for one gate at a time: dividing a sum into zones costs more than a gate does.
    pub const ONE: Self = Self {
        terms: 1024,
        max: 1,
    };
}

/// [`propagate_with`] with the zones chosen by the caller.
pub fn propagate_in_zones<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &mut PauliSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    custom: Option<&CustomTruncation<'_, W, C>>,
    zones: &Zones,
) -> Result<(), Error> {
    for gate in circuit {
        gate.validate(sum.num_qubits())?;
    }
    let rules = Rules { truncation, custom };
    let num_qubits = sum.num_qubits();
    let (strings, coeffs) = sum.take_ordered();

    // The counters behind `max_freq` and `max_sins` are kept only when one of them is set.
    let (strings, coeffs) = if rules.counts() {
        let tracked: Vec<Tracked<C>> = coeffs.into_iter().map(Tracked::start).collect();
        let (strings, tracked) = engine::run(
            circuit, num_qubits, strings, tracked, &rules, picture, zones,
        );
        (strings, tracked.iter().map(Path::coeff).collect())
    } else {
        engine::run(circuit, num_qubits, strings, coeffs, &rules, picture, zones)
    };
    sum.set_ordered(strings, coeffs);
    Ok(())
}

/// The rules that decide which terms are dropped.
pub(crate) struct Rules<'a, const W: usize, C> {
    pub(crate) truncation: &'a Truncation,
    pub(crate) custom: Option<&'a CustomTruncation<'a, W, C>>,
}

impl<const W: usize, C: Coefficient> Rules<'_, W, C> {
    /// Whether a rule reads the Pauli string, so that a Clifford gate can make it drop a term.
    pub(crate) fn reads_string(&self) -> bool {
        self.truncation.max_weight.is_some()
            || self.truncation.weight_damping.is_some()
            || self.custom.is_some()
    }

    /// Whether a rule reads the counters of the rotations behind a term.
    pub(crate) fn counts(&self) -> bool {
        self.truncation.max_freq.is_some() || self.truncation.max_sins.is_some()
    }

    /// Whether the smallest absolute coefficient that is kept depends on the sum.
    pub(crate) fn is_relative(&self) -> bool {
        self.truncation.min_rel_coeff.is_some()
    }

    /// The smallest absolute coefficient that is kept, where `largest` is the largest
    /// absolute coefficient of the sum as it is after a gate.
    pub(crate) fn threshold(&self, largest: impl FnOnce() -> f64) -> f64 {
        let min_abs = self.truncation.min_abs_coeff;
        match self.truncation.min_rel_coeff {
            None => min_abs,
            Some(min_rel) => (min_rel * largest()).max(min_abs),
        }
    }

    #[inline]
    pub(crate) fn drops<P: Path<Coeff = C>>(
        &self,
        string: &PauliString<W>,
        path: &P,
        min_abs: f64,
    ) -> bool {
        let truncation = self.truncation;
        let coeff = path.coeff();
        if coeff.is_below(min_abs)
            || truncation
                .max_weight
                .is_some_and(|max| string.weight() > max)
            || truncation.max_freq.is_some_and(|max| path.freq() > max)
            || truncation.max_sins.is_some_and(|max| path.sins() > max)
        {
            return true;
        }
        if let Some(damping) = truncation.weight_damping {
            let damped = coeff.abs() * 10f64.powf(-damping.gamma * f64::from(string.weight()));
            if damped < damping.min_abs_coeff {
                return true;
            }
        }
        self.custom.is_some_and(|custom| custom(string, coeff))
    }
}
