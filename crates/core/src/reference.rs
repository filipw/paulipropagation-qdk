//! The same propagation on a hash map.
//!
//! This applies every gate to one term at a time and looks each new Pauli string up in a
//! hash map. It is slower than the engine behind [`crate::propagate_with`], which works on
//! sorted arrays, and it shares nothing with it but the rules for dropping terms. The two
//! give the same sum, coefficient for coefficient, which is what the tests check.

use crate::coeff::{Path, Tracked};
use crate::propagate::Rules;
use crate::{
    Coefficient, CustomTruncation, Error, Gate, PauliString, PauliSum, Picture, Truncation,
};
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

/// Does what [`crate::propagate_with`] does.
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
    let (strings, coeffs) = sum.take_ordered();
    let start = strings.into_iter().zip(coeffs);

    // The counters behind `max_freq` and `max_sins` are kept only when one of them is set.
    let mut result: Vec<(PauliString<W>, C)> = if rules.counts() {
        let mut terms: FxHashMap<PauliString<W>, Tracked<C>> =
            start.map(|(p, c)| (p, Tracked::start(c))).collect();
        run(circuit, &mut terms, &rules, picture);
        let terms = terms.into_iter();
        terms.map(|(p, path)| (p, path.coeff())).collect()
    } else {
        let mut terms: FxHashMap<PauliString<W>, C> = start.collect();
        run(circuit, &mut terms, &rules, picture);
        terms.into_iter().collect()
    };
    result.sort_unstable_by_key(|term| term.0);
    let (strings, coeffs) = result.into_iter().unzip();
    sum.set_ordered(strings, coeffs);
    Ok(())
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
    let min_abs = rules.threshold(|| {
        let sizes = terms.values().map(|path| path.coeff().abs());
        sizes.fold(0.0, f64::max)
    });
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
