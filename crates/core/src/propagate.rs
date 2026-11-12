//! Propagation of a Pauli sum through a circuit.

use crate::coeff::{Path, Tracked};
use crate::{Coefficient, Gate, PauliString, PauliSum};
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    DuplicateQubit { qubit: usize },
    TooManyQubits { num_qubits: usize, capacity: usize },
    NonFiniteAngle,
    InvalidNoise(&'static str),
    InvalidMatrix(&'static str),
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
            Error::InvalidNoise(what) | Error::InvalidMatrix(what) => write!(f, "{what}"),
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
    for gate in circuit {
        gate.validate(sum.num_qubits())?;
    }
    let rules = Rules { truncation, custom };

    // The counters behind `max_freq` and `max_sins` are kept only when one of them is set.
    if truncation.max_freq.is_some() || truncation.max_sins.is_some() {
        let mut tracked: FxHashMap<PauliString<W>, Tracked<C>> = sum
            .terms
            .drain()
            .map(|(p, c)| (p, Tracked::start(c)))
            .collect();
        run(circuit, &mut tracked, &rules, picture);
        sum.terms
            .extend(tracked.into_iter().map(|(p, path)| (p, path.coeff())));
    } else {
        run(circuit, &mut sum.terms, &rules, picture);
    }
    Ok(())
}

struct Rules<'a, const W: usize, C> {
    truncation: &'a Truncation,
    custom: Option<&'a CustomTruncation<'a, W, C>>,
}

impl<const W: usize, C: Coefficient> Rules<'_, W, C> {
    /// Whether a rule reads the Pauli string, so that a Clifford gate can make it drop a term.
    fn reads_string(&self) -> bool {
        self.truncation.max_weight.is_some()
            || self.truncation.weight_damping.is_some()
            || self.custom.is_some()
    }

    /// The smallest absolute coefficient that is kept, with the relative threshold taken
    /// from the sum as it is after a gate.
    fn threshold<P: Path<Coeff = C>>(&self, terms: &FxHashMap<PauliString<W>, P>) -> f64 {
        let min_abs = self.truncation.min_abs_coeff;
        match self.truncation.min_rel_coeff {
            None => min_abs,
            Some(min_rel) => {
                let largest = terms
                    .values()
                    .map(|path| path.coeff().abs())
                    .fold(0.0, f64::max);
                (min_rel * largest).max(min_abs)
            }
        }
    }

    #[inline]
    fn drops<P: Path<Coeff = C>>(&self, string: &PauliString<W>, path: &P, min_abs: f64) -> bool {
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

/// Buffers that are reused across gates.
struct Scratch<const W: usize, P> {
    /// The next sum under a Clifford gate.
    spare: FxHashMap<PauliString<W>, P>,
    /// The new terms under a rotation or amplitude damping.
    created: Vec<(PauliString<W>, P)>,
    /// The images under a transfer map, each with the Pauli string it came from.
    mapped: Vec<(PauliString<W>, PauliString<W>, P)>,
}

fn run<const W: usize, P: Path>(
    circuit: &[Gate],
    terms: &mut FxHashMap<PauliString<W>, P>,
    rules: &Rules<'_, W, P::Coeff>,
    picture: Picture,
) {
    let mut scratch = Scratch {
        spare: FxHashMap::default(),
        created: Vec::new(),
        mapped: Vec::new(),
    };
    match picture {
        Picture::Heisenberg => {
            for gate in circuit.iter().rev() {
                apply(gate, terms, rules, &mut scratch);
            }
        }
        Picture::Schrodinger => {
            for gate in circuit {
                apply(&gate.adjoint(), terms, rules, &mut scratch);
            }
        }
    }
}

/// Applies the action of `gate` on observables to every term, then truncates.
fn apply<const W: usize, P: Path>(
    gate: &Gate,
    terms: &mut FxHashMap<PauliString<W>, P>,
    rules: &Rules<'_, W, P::Coeff>,
    scratch: &mut Scratch<W, P>,
) {
    match gate {
        Gate::Rotation { generator, theta } => {
            apply_rotation(
                &PauliString::from_ops(generator),
                *theta,
                terms,
                &mut scratch.created,
            );
        }
        Gate::PauliChannel { qubits, factors } => {
            for (p, path) in terms.iter_mut() {
                *path = path.scale(factors[Gate::channel_index(qubits, p)]);
            }
        }
        Gate::AmplitudeDamping { qubit, gamma } => {
            apply_amplitude_damping(*qubit, *gamma, terms, &mut scratch.created);
        }
        Gate::TransferMap { qubits, columns } => {
            apply_transfer_map(qubits, columns, terms, &mut scratch.mapped);
        }
        // A Clifford gate is a bijection of the Pauli strings up to signs, which leaves
        // nothing to drop for a rule that reads only the coefficient.
        _ => {
            apply_clifford(gate, terms, rules, &mut scratch.spare);
            return;
        }
    }
    let min_abs = rules.threshold(terms);
    terms.retain(|p, path| !rules.drops(p, path, min_abs));
}

fn apply_clifford<const W: usize, P: Path>(
    gate: &Gate,
    terms: &mut FxHashMap<PauliString<W>, P>,
    rules: &Rules<'_, W, P::Coeff>,
    spare: &mut FxHashMap<PauliString<W>, P>,
) {
    let truncates = rules.reads_string();
    let min_abs = rules.truncation.min_abs_coeff;

    // A Pauli gate only changes signs.
    if matches!(gate, Gate::X(_) | Gate::Y(_) | Gate::Z(_)) {
        for (p, path) in terms.iter_mut() {
            let mut q = *p;
            if gate.conjugate_clifford(&mut q) {
                *path = path.scale(-1.0);
            }
        }
        if truncates {
            terms.retain(|p, path| !rules.drops(p, path, min_abs));
        }
        return;
    }

    spare.clear();
    spare.reserve(terms.len());
    for (mut p, mut path) in terms.drain() {
        if gate.conjugate_clifford(&mut p) {
            path = path.scale(-1.0);
        }
        if !(truncates && rules.drops(&p, &path, min_abs)) {
            spare.insert(p, path);
        }
    }
    std::mem::swap(terms, spare);
}

/// Adds the new terms to the sum, merging each with the term on the same Pauli string.
fn merge_created<const W: usize, P: Path>(
    terms: &mut FxHashMap<PauliString<W>, P>,
    created: &mut Vec<(PauliString<W>, P)>,
) {
    for (p, path) in created.drain(..) {
        match terms.entry(p) {
            Entry::Occupied(mut entry) => {
                let merged = entry.get().merge(path);
                entry.insert(merged);
            }
            Entry::Vacant(entry) => {
                entry.insert(path);
            }
        }
    }
}

/// Applies `P -> U† P U` for `U = exp(-i theta / 2 G)` to every term.
///
/// A string that commutes with `G` is unchanged. One that anticommutes becomes
/// `cos(theta) P + sin(theta) (i G P)`, and `i G P` is a Pauli string up to a sign.
/// `P -> G P` pairs up the anticommuting strings, so every coefficient is the sum of at
/// most two numbers and the result does not depend on the order of the terms.
fn apply_rotation<const W: usize, P: Path>(
    generator: &PauliString<W>,
    theta: f64,
    terms: &mut FxHashMap<PauliString<W>, P>,
    created: &mut Vec<(PauliString<W>, P)>,
) {
    let (sin, cos) = theta.sin_cos();
    created.clear();
    for (p, path) in terms.iter_mut() {
        if !p.commutes(generator) {
            // G P = i^k P' with k odd, so i G P = i^(k + 1) P' = -P' for k = 1 and +P' for k = 3.
            let (product, k) = generator.mul(p);
            debug_assert!(k == 1 || k == 3);
            let sign = if k == 3 { 1.0 } else { -1.0 };
            created.push((product, path.sin_branch(sin, sign)));
            *path = path.cos_branch(cos);
        }
    }
    merge_created(terms, created);
}

/// Applies the adjoint of amplitude damping on `qubit` to every term.
///
/// X and Y on the qubit are multiplied by `sqrt(1 - gamma)`. A string with Z there keeps
/// `1 - gamma` of its coefficient and adds `gamma` of it to the string with the identity there.
fn apply_amplitude_damping<const W: usize, P: Path>(
    qubit: usize,
    gamma: f64,
    terms: &mut FxHashMap<PauliString<W>, P>,
    created: &mut Vec<(PauliString<W>, P)>,
) {
    let transverse = (1.0 - gamma).sqrt();
    created.clear();
    for (p, path) in terms.iter_mut() {
        match (p.x_bit(qubit), p.z_bit(qubit)) {
            (false, false) => {}
            (false, true) => {
                let mut without_z = *p;
                without_z.flip_z(qubit);
                created.push((without_z, path.scale(gamma)));
                *path = path.scale(1.0 - gamma);
            }
            _ => *path = path.scale(transverse),
        }
    }
    merge_created(terms, created);
}

/// Replaces every term by its image under a transfer map on `qubits`.
///
/// Several terms can contribute to one Pauli string. Their contributions are merged in the
/// order of the Pauli strings they come from, so that the result does not depend on the
/// order of the terms.
fn apply_transfer_map<const W: usize, P: Path>(
    qubits: &[usize],
    columns: &[Vec<(usize, f64)>],
    terms: &mut FxHashMap<PauliString<W>, P>,
    mapped: &mut Vec<(PauliString<W>, PauliString<W>, P)>,
) {
    mapped.clear();
    for (p, path) in terms.drain() {
        for &(index, factor) in &columns[Gate::channel_index(qubits, &p)] {
            mapped.push((Gate::with_index(qubits, p, index), p, path.scale(factor)));
        }
    }
    mapped.sort_unstable_by_key(|&(image, source, _)| (image, source));
    for (image, _, path) in mapped.drain(..) {
        match terms.entry(image) {
            Entry::Occupied(mut entry) => {
                let merged = entry.get().merge(path);
                entry.insert(merged);
            }
            Entry::Vacant(entry) => {
                entry.insert(path);
            }
        }
    }
}
