//! Propagation of a Pauli sum through a circuit.

use crate::{Gate, PauliString, PauliSum};
use rustc_hash::FxHashMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    DuplicateQubit { qubit: usize },
    TooManyQubits { num_qubits: usize, capacity: usize },
    NonFiniteAngle,
    InvalidNoise(&'static str),
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
            Error::InvalidNoise(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for Error {}

/// Which terms to drop after every gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truncation {
    /// Drop the Pauli strings with more than this many non-identity operators.
    pub max_weight: Option<u32>,
    /// Drop the terms whose coefficient is smaller than this in absolute value.
    pub min_abs_coeff: f64,
}

impl Default for Truncation {
    /// No weight truncation and `min_abs_coeff = 1e-10`, as in PauliPropagation.jl.
    fn default() -> Self {
        Self {
            max_weight: None,
            min_abs_coeff: 1e-10,
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
        }
    }

    #[inline]
    fn drops<const W: usize>(&self, string: &PauliString<W>, coeff: f64) -> bool {
        coeff.abs() < self.min_abs_coeff || self.max_weight.is_some_and(|max| string.weight() > max)
    }
}

/// Replaces the observable `sum` by `U† sum U`, where `U` is the unitary of `circuit`. With
/// noise channels in the circuit, this is the adjoint of the channel that the circuit applies
/// to states, so expectation values are those of the noisy circuit.
///
/// `circuit` lists the gates in the order they act on a state, so they are applied to the
/// observable last to first. After every gate the terms that `truncation` selects are dropped,
/// which makes the result an approximation unless nothing is dropped.
pub fn propagate<const W: usize>(
    circuit: &[Gate],
    sum: &mut PauliSum<W>,
    truncation: &Truncation,
) -> Result<(), Error> {
    for gate in circuit {
        gate.validate(sum.num_qubits())?;
    }

    // Reused across gates: the next sum under a Clifford gate, the new terms under a rotation.
    let mut spare: FxHashMap<PauliString<W>, f64> = FxHashMap::default();
    let mut created: Vec<(PauliString<W>, f64)> = Vec::new();

    for gate in circuit.iter().rev() {
        match gate {
            Gate::Rotation { generator, theta } => {
                apply_rotation(
                    &PauliString::from_ops(generator),
                    *theta,
                    &mut sum.terms,
                    &mut created,
                );
                sum.terms.retain(|p, c| !truncation.drops(p, *c));
            }
            Gate::PauliChannel { qubits, factors } => {
                for (p, c) in &mut sum.terms {
                    *c *= factors[Gate::channel_index(qubits, p)];
                }
                sum.terms.retain(|p, c| !truncation.drops(p, *c));
            }
            Gate::AmplitudeDamping { qubit, gamma } => {
                apply_amplitude_damping(*qubit, *gamma, &mut sum.terms, &mut created);
                sum.terms.retain(|p, c| !truncation.drops(p, *c));
            }
            // A Pauli gate only changes signs, which truncation does not see.
            Gate::X(_) | Gate::Y(_) | Gate::Z(_) => {
                for (p, c) in &mut sum.terms {
                    let mut q = *p;
                    if gate.conjugate_clifford(&mut q) {
                        *c = -*c;
                    }
                }
                sum.terms.retain(|p, c| !truncation.drops(p, *c));
            }
            // Any other Clifford gate is a bijection of the Pauli strings.
            _ => {
                spare.clear();
                spare.reserve(sum.terms.len());
                for (mut p, mut c) in sum.terms.drain() {
                    if gate.conjugate_clifford(&mut p) {
                        c = -c;
                    }
                    if !truncation.drops(&p, c) {
                        spare.insert(p, c);
                    }
                }
                std::mem::swap(&mut sum.terms, &mut spare);
            }
        }
    }
    Ok(())
}

/// Applies `P -> U† P U` for `U = exp(-i theta / 2 G)` to every term.
///
/// A string that commutes with `G` is unchanged. One that anticommutes becomes
/// `cos(theta) P + sin(theta) (i G P)`, and `i G P` is a Pauli string up to a sign.
/// `P -> G P` pairs up the anticommuting strings, so every coefficient is the sum of at
/// most two numbers and the result does not depend on the order of the terms.
fn apply_rotation<const W: usize>(
    generator: &PauliString<W>,
    theta: f64,
    terms: &mut FxHashMap<PauliString<W>, f64>,
    created: &mut Vec<(PauliString<W>, f64)>,
) {
    let (sin, cos) = theta.sin_cos();
    created.clear();
    for (p, c) in terms.iter_mut() {
        if !p.commutes(generator) {
            // G P = i^k P' with k odd, so i G P = i^(k + 1) P' = -P' for k = 1 and +P' for k = 3.
            let (product, k) = generator.mul(p);
            debug_assert!(k == 1 || k == 3);
            let sign = if k == 3 { 1.0 } else { -1.0 };
            created.push((product, *c * sin * sign));
            *c *= cos;
        }
    }
    for &(p, c) in created.iter() {
        *terms.entry(p).or_insert(0.0) += c;
    }
}

/// Applies the adjoint of amplitude damping on `qubit` to every term.
///
/// X and Y on the qubit are multiplied by `sqrt(1 - gamma)`. A string with Z there keeps
/// `1 - gamma` of its coefficient and adds `gamma` of it to the string with the identity there.
fn apply_amplitude_damping<const W: usize>(
    qubit: usize,
    gamma: f64,
    terms: &mut FxHashMap<PauliString<W>, f64>,
    created: &mut Vec<(PauliString<W>, f64)>,
) {
    let transverse = (1.0 - gamma).sqrt();
    created.clear();
    for (p, c) in terms.iter_mut() {
        match (p.x_bit(qubit), p.z_bit(qubit)) {
            (false, false) => {}
            (false, true) => {
                let mut without_z = *p;
                without_z.flip_z(qubit);
                created.push((without_z, gamma * *c));
                *c *= 1.0 - gamma;
            }
            _ => *c *= transverse,
        }
    }
    for &(p, c) in created.iter() {
        *terms.entry(p).or_insert(0.0) += c;
    }
}
