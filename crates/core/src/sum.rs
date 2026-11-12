//! A sparse sum of Pauli strings with real coefficients.

use crate::{Error, PauliOp, PauliString};
use rustc_hash::FxHashMap;

/// The operator `sum_k c_k P_k` on `num_qubits` qubits, with real `c_k` and distinct Pauli strings `P_k`.
#[derive(Clone, Debug)]
pub struct PauliSum<const W: usize> {
    num_qubits: usize,
    pub(crate) terms: FxHashMap<PauliString<W>, f64>,
}

impl<const W: usize> PauliSum<W> {
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
    pub fn add(&mut self, ops: &[(usize, PauliOp)], coeff: f64) -> Result<(), Error> {
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
    pub fn add_string(&mut self, string: PauliString<W>, coeff: f64) {
        debug_assert!(string.max_qubit().is_none_or(|q| q < self.num_qubits));
        *self.terms.entry(string).or_insert(0.0) += coeff;
    }

    /// The coefficient of `string`, zero if it is not in the sum.
    #[must_use]
    pub fn coeff(&self, string: &PauliString<W>) -> f64 {
        self.terms.get(string).copied().unwrap_or(0.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PauliString<W>, f64)> {
        self.terms.iter().map(|(p, &c)| (p, c))
    }

    /// The terms ordered by Pauli string, for output that does not depend on hashing.
    #[must_use]
    pub fn sorted_terms(&self) -> Vec<(PauliString<W>, f64)> {
        let mut terms: Vec<_> = self.terms.iter().map(|(&p, &c)| (p, c)).collect();
        terms.sort_unstable_by_key(|term| term.0);
        terms
    }

    /// The expectation value `<0...0| O |0...0>`: the sum of the coefficients of the strings
    /// made of `I` and `Z` only.
    #[must_use]
    pub fn overlap_with_zero(&self) -> f64 {
        // Summed in a fixed order so the result is reproducible.
        let mut coeffs: Vec<(PauliString<W>, f64)> = self
            .terms
            .iter()
            .filter(|(p, _)| !p.has_x_or_y())
            .map(|(&p, &c)| (p, c))
            .collect();
        coeffs.sort_unstable_by_key(|term| term.0);
        coeffs.iter().map(|&(_, c)| c).sum()
    }

    /// The sum of the squared coefficients, which unitary gates preserve when nothing is truncated.
    #[must_use]
    pub fn norm_sqr(&self) -> f64 {
        let mut squares: Vec<f64> = self.terms.values().map(|c| c * c).collect();
        squares.sort_unstable_by(f64::total_cmp);
        squares.iter().sum()
    }
}
