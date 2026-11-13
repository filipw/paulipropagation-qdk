//! Sums and products of Pauli sums, and the merging of terms that a symmetry makes equal.

use crate::{Coefficient, Complex64, Error, PauliString, PauliSum};
use rustc_hash::FxHashMap;

/// `i^power`.
fn phase(power: u8) -> Complex64 {
    match power & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// A sum of the given terms, which may repeat a Pauli string. The coefficients of a string
/// are added up in the order they are given. Coefficients that add up to zero are kept.
fn collected<const W: usize, C: Coefficient>(
    num_qubits: usize,
    terms: impl Iterator<Item = (PauliString<W>, C)>,
) -> PauliSum<W, C> {
    let mut totals: FxHashMap<PauliString<W>, C> = FxHashMap::default();
    for (string, coeff) in terms {
        let total = totals.entry(string).or_insert(C::ZERO);
        *total = total.plus(coeff);
    }
    let mut terms: Vec<(PauliString<W>, C)> = totals.into_iter().collect();
    terms.sort_unstable_by_key(|term| term.0);
    let mut sum = PauliSum::new(num_qubits).expect("the register was valid for the factors");
    let (strings, coeffs) = terms.into_iter().unzip();
    sum.set_ordered(strings, coeffs);
    sum
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    fn same_register(&self, other: &Self) -> Result<(), Error> {
        if self.num_qubits() == other.num_qubits() {
            Ok(())
        } else {
            Err(Error::Invalid(
                "the two sums are on different numbers of qubits",
            ))
        }
    }

    /// Multiplies every coefficient by `factor`.
    pub fn scale(&mut self, factor: C) {
        let (strings, coeffs) = self.take_ordered();
        let coeffs = coeffs.into_iter().map(|c| c.times(factor)).collect();
        self.set_ordered(strings, coeffs);
    }

    /// Adds `factor * other` to the sum.
    pub fn add_scaled(&mut self, other: &Self, factor: C) -> Result<(), Error> {
        self.same_register(other)?;
        for (string, coeff) in other.sorted_terms() {
            self.add_string(string, coeff.times(factor));
        }
        Ok(())
    }

    /// The terms of `self * other`, or of `self * other - other * self`.
    fn products(&self, other: &Self, commutator: bool) -> Result<PauliSum<W, Complex64>, Error> {
        self.same_register(other)?;
        let (left, right) = (self.sorted_terms(), other.sorted_terms());
        let terms = left.iter().flat_map(|(a, ca)| {
            right.iter().filter_map(move |(b, cb)| {
                // Two Pauli strings commute or anticommute, and the commutator of two that
                // anticommute is twice their product.
                if commutator && a.commutes(b) {
                    return None;
                }
                let (string, power) = a.mul(b);
                let factor = if commutator { 2.0 } else { 1.0 };
                let coeff = ca.to_complex() * cb.to_complex() * phase(power) * factor;
                Some((string, coeff))
            })
        });
        Ok(collected(self.num_qubits(), terms))
    }

    /// The product `self * other` of the two operators.
    ///
    /// The product of two Pauli strings is a Pauli string times a power of `i`, so the
    /// coefficients of the product are complex also where those of the factors are real.
    pub fn product(&self, other: &Self) -> Result<PauliSum<W, Complex64>, Error> {
        self.products(other, false)
    }

    /// The commutator `self * other - other * self`.
    pub fn commutator(&self, other: &Self) -> Result<PauliSum<W, Complex64>, Error> {
        self.products(other, true)
    }

    /// Replaces every Pauli string by `representative(string)` and adds up the terms that
    /// have become equal.
    ///
    /// Where a symmetry of the circuit and of the state makes Pauli strings contribute
    /// equally, a function that maps all of them to one of them shrinks the sum without
    /// changing expectation values in that state.
    pub fn merge_by(&mut self, representative: impl Fn(&PauliString<W>) -> PauliString<W>) {
        let terms = self.sorted_terms();
        let mapped = terms.iter().map(|(string, c)| (representative(string), *c));
        *self = collected(self.num_qubits(), mapped);
    }

    /// Merges the Pauli strings that are translations of each other on a ring of the qubits
    /// `0..num_qubits`. Of each set of translations, the string that is lowest in
    /// [`PauliString::cmp_from_highest_qubit`] is kept, as in PauliPropagation.jl.
    pub fn translation_merge(&mut self) {
        let n = self.num_qubits();
        self.merge_by(|string| lowest_image(string, (1..n).map(|shift| move |q| (q + shift) % n)));
    }

    /// Merges the Pauli strings that are translations of each other on a torus of `nx`
    /// columns and `ny` rows, where the qubit in column `x` of row `y` is `y * nx + x`.
    pub fn translation_merge_on_grid(&mut self, nx: usize, ny: usize) -> Result<(), Error> {
        if nx.checked_mul(ny) != Some(self.num_qubits()) {
            return Err(Error::Invalid(
                "the grid does not have as many sites as the sum has qubits",
            ));
        }
        self.merge_by(|string| {
            let shifts = (0..ny).flat_map(|dy| (0..nx).map(move |dx| (dx, dy)));
            lowest_image(
                string,
                shifts.map(|(dx, dy)| move |q| ((q / nx + dy) % ny) * nx + (q % nx + dx) % nx),
            )
        });
        Ok(())
    }
}

/// The lowest of `string` and its images under the given maps of the qubits.
fn lowest_image<const W: usize, F: Fn(usize) -> usize>(
    string: &PauliString<W>,
    moves: impl Iterator<Item = F>,
) -> PauliString<W> {
    let ops = string.ops();
    let mut lowest = *string;
    let mut moved = ops.clone();
    for to in moves {
        for (image, &(q, op)) in moved.iter_mut().zip(&ops) {
            *image = (to(q), op);
        }
        let candidate = PauliString::from_ops(&moved);
        if candidate.cmp_from_highest_qubit(&lowest).is_lt() {
            lowest = candidate;
        }
    }
    lowest
}
