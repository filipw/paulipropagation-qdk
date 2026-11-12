//! A sparse sum of Pauli strings.

use crate::{Coefficient, Error, PauliOp, PauliString};
use rustc_hash::FxHashMap;

/// The operator `sum_k c_k P_k` on `num_qubits` qubits, with distinct Pauli strings `P_k` and
/// real or complex coefficients `c_k`.
#[derive(Clone, Debug)]
pub struct PauliSum<const W: usize, C = f64> {
    num_qubits: usize,
    pub(crate) terms: FxHashMap<PauliString<W>, C>,
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    /// The empty sum on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Result<Self, Error> {
        if num_qubits > PauliString::<W>::CAPACITY {
            return Err(Error::TooManyQubits {
                num_qubits,
                capacity: PauliString::<W>::CAPACITY,
            });
        }
        Ok(Self {
            num_qubits,
            terms: FxHashMap::default(),
        })
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Adds `coeff * P`, where `P` is given as `(qubit, operator)` pairs on distinct qubits.
    pub fn add(&mut self, ops: &[(usize, PauliOp)], coeff: C) -> Result<(), Error> {
        for (i, &(q, _)) in ops.iter().enumerate() {
            if q >= self.num_qubits {
                return Err(Error::QubitOutOfRange {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
            }
            if ops[..i].iter().any(|&(other, _)| other == q) {
                return Err(Error::DuplicateQubit { qubit: q });
            }
        }
        self.add_string(PauliString::from_ops(ops), coeff);
        Ok(())
    }

    /// Adds `coeff * string`. The string must not reach beyond `num_qubits`.
    pub fn add_string(&mut self, string: PauliString<W>, coeff: C) {
        debug_assert!(string.max_qubit().is_none_or(|q| q < self.num_qubits));
        let entry = self.terms.entry(string).or_insert(C::ZERO);
        *entry = entry.plus(coeff);
    }

    /// The coefficient of `string`, zero if it is not in the sum.
    #[must_use]
    pub fn coeff(&self, string: &PauliString<W>) -> C {
        self.terms.get(string).copied().unwrap_or(C::ZERO)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PauliString<W>, C)> {
        self.terms.iter().map(|(p, &c)| (p, c))
    }

    /// The terms ordered by Pauli string, for output that does not depend on hashing.
    #[must_use]
    pub fn sorted_terms(&self) -> Vec<(PauliString<W>, C)> {
        let mut terms: Vec<_> = self.terms.iter().map(|(&p, &c)| (p, c)).collect();
        terms.sort_unstable_by_key(|term| term.0);
        terms
    }

    /// The expectation value `<0...0| O |0...0>`: the sum of the coefficients of the strings
    /// made of `I` and `Z` only.
    #[must_use]
    pub fn overlap_with_zero(&self) -> C {
        sum_in_order(
            self.terms
                .iter()
                .filter(|(p, _)| !p.has_x_or_y())
                .map(|(&p, &c)| (p, c)),
        )
    }

    /// The expectation value `<+...+| O |+...+>`: the sum of the coefficients of the strings
    /// made of `I` and `X` only.
    #[must_use]
    pub fn overlap_with_plus(&self) -> C {
        sum_in_order(
            self.terms
                .iter()
                .filter(|(p, _)| !p.has_y_or_z())
                .map(|(&p, &c)| (p, c)),
        )
    }

    /// The expectation value in the computational basis state whose qubits in `ones` are 1
    /// and whose other qubits are 0.
    pub fn overlap_with_computational(&self, ones: &[usize]) -> Result<C, Error> {
        if let Some(&qubit) = ones.iter().find(|&&q| q >= self.num_qubits) {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        // A string of I and Z has the sign of its number of Z on the ones, and that number
        // is odd exactly if the string anticommutes with X on the ones.
        let flips: Vec<(usize, PauliOp)> = ones.iter().map(|&q| (q, PauliOp::X)).collect();
        let flips = PauliString::from_ops(&flips);
        Ok(sum_in_order(
            self.terms
                .iter()
                .filter(|(p, _)| !p.has_x_or_y())
                .map(|(&p, &c)| (p, if p.commutes(&flips) { c } else { c.scale(-1.0) })),
        ))
    }

    /// The expectation value in the maximally mixed state, `Tr[O] / 2^n`: the coefficient of
    /// the identity.
    #[must_use]
    pub fn overlap_with_max_mixed(&self) -> C {
        self.coeff(&PauliString::identity())
    }

    /// The sum over the Pauli strings of the products of their coefficients in the two sums,
    /// which is `Tr[A B] / 2^n`. Complex coefficients are not conjugated.
    #[must_use]
    pub fn scalar_product(&self, other: &Self) -> C {
        let (small, large) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        sum_in_order(
            small
                .terms
                .iter()
                .filter_map(|(p, &c)| large.terms.get(p).map(|&d| (*p, c.times(d)))),
        )
    }

    /// The sum of the squared absolute values of the coefficients, which unitary gates
    /// preserve when nothing is truncated.
    #[must_use]
    pub fn norm_sqr(&self) -> f64 {
        let mut squares: Vec<f64> = self.terms.values().map(|c| c.norm_sqr()).collect();
        squares.sort_unstable_by(f64::total_cmp);
        squares.iter().sum()
    }

    /// The largest absolute value of a coefficient, zero for the empty sum.
    #[must_use]
    pub fn max_abs_coeff(&self) -> f64 {
        self.terms.values().map(|c| c.abs()).fold(0.0, f64::max)
    }
}

/// Sums the values in the order of their Pauli strings, so that the result does not depend
/// on the order in which a hash map yields them.
fn sum_in_order<const W: usize, C: Coefficient>(
    terms: impl Iterator<Item = (PauliString<W>, C)>,
) -> C {
    let mut terms: Vec<(PauliString<W>, C)> = terms.collect();
    terms.sort_unstable_by_key(|term| term.0);
    terms
        .iter()
        .fold(C::ZERO, |total, &(_, value)| total.plus(value))
}
