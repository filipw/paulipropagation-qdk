//! A sparse sum of Pauli strings.

use crate::{Coefficient, Error, PauliOp, PauliString};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cmp::Ordering;

/// The operator `sum_k c_k P_k` on `num_qubits` qubits, with distinct Pauli strings `P_k` and
/// real or complex coefficients `c_k`.
///
/// The terms are kept in the order of their Pauli strings, which is the form that
/// propagation works on. Terms that are added one by one are collected on the side and put
/// in order when the sum is propagated or read in order.
#[derive(Clone, Debug)]
pub struct PauliSum<const W: usize, C = f64> {
    num_qubits: usize,
    /// The Pauli strings in ascending order.
    strings: Vec<PauliString<W>>,
    /// The coefficient of each of `strings`.
    coeffs: Vec<C>,
    /// Terms added since the last ordering, on Pauli strings that are not in `strings`.
    added: FxHashMap<PauliString<W>, C>,
}

/// Fewer added terms than this are not worth putting in order.
const MIN_ADDED: usize = 1024;

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
            strings: Vec::new(),
            coeffs: Vec::new(),
            added: FxHashMap::default(),
        })
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len() + self.added.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        if self.added.is_empty() && self.strings.last().is_none_or(|last| *last < string) {
            self.strings.push(string);
            self.coeffs.push(coeff);
            return;
        }
        match self.strings.binary_search(&string) {
            Ok(index) => self.coeffs[index] = self.coeffs[index].plus(coeff),
            Err(_) => {
                let entry = self.added.entry(string).or_insert(C::ZERO);
                *entry = entry.plus(coeff);
                // Ordering costs as much as the sum is long, so it waits until the sum
                // has doubled.
                if self.added.len() > self.strings.len().max(MIN_ADDED) {
                    self.put_in_order();
                }
            }
        }
    }

    /// Moves the added terms into the ordered ones.
    pub(crate) fn put_in_order(&mut self) {
        if self.added.is_empty() {
            return;
        }
        let mut added: Vec<(PauliString<W>, C)> = self.added.drain().collect();
        added.sort_unstable_by_key(|term| term.0);

        let old_strings = std::mem::take(&mut self.strings);
        let old_coeffs = std::mem::take(&mut self.coeffs);
        self.strings.reserve(old_strings.len() + added.len());
        self.coeffs.reserve(old_strings.len() + added.len());
        let mut old = old_strings.into_iter().zip(old_coeffs).peekable();
        let mut new = added.into_iter().peekable();
        // No Pauli string is in both, so the two never tie.
        loop {
            let from_old = match (old.peek(), new.peek()) {
                (Some(a), Some(b)) => a.0 < b.0,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            let (string, coeff) = if from_old { old.next() } else { new.next() }
                .expect("the side that was peeked has a term");
            self.strings.push(string);
            self.coeffs.push(coeff);
        }
    }

    /// The sum with all of its terms in order.
    fn in_order(&self) -> Cow<'_, Self> {
        if self.added.is_empty() {
            Cow::Borrowed(self)
        } else {
            let mut ordered = self.clone();
            ordered.put_in_order();
            Cow::Owned(ordered)
        }
    }

    /// The ordered Pauli strings and their coefficients, which are all the terms.
    pub(crate) fn take_ordered(&mut self) -> (Vec<PauliString<W>>, Vec<C>) {
        self.put_in_order();
        (
            std::mem::take(&mut self.strings),
            std::mem::take(&mut self.coeffs),
        )
    }

    /// Replaces the terms by Pauli strings in ascending order and their coefficients.
    pub(crate) fn set_ordered(&mut self, strings: Vec<PauliString<W>>, coeffs: Vec<C>) {
        debug_assert!(strings.len() == coeffs.len());
        debug_assert!(strings.windows(2).all(|pair| pair[0] < pair[1]));
        self.strings = strings;
        self.coeffs = coeffs;
        self.added.clear();
    }

    /// The coefficient of `string`, zero if it is not in the sum.
    #[must_use]
    pub fn coeff(&self, string: &PauliString<W>) -> C {
        match self.strings.binary_search(string) {
            Ok(index) => self.coeffs[index],
            Err(_) => self.added.get(string).copied().unwrap_or(C::ZERO),
        }
    }

    /// The terms, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&PauliString<W>, C)> {
        let ordered = self.strings.iter().zip(self.coeffs.iter().copied());
        ordered.chain(self.added.iter().map(|(p, &c)| (p, c)))
    }

    /// The terms ordered by Pauli string.
    #[must_use]
    pub fn sorted_terms(&self) -> Vec<(PauliString<W>, C)> {
        let ordered = self.in_order();
        let strings = ordered.strings.iter().copied();
        strings.zip(ordered.coeffs.iter().copied()).collect()
    }

    /// The sum of `value(string, coeff)` over the terms that it is given for, added up in
    /// the order of the Pauli strings.
    fn sum_in_order(&self, value: impl Fn(&PauliString<W>, C) -> Option<C>) -> C {
        let ordered = self.in_order();
        let terms = ordered.strings.iter().zip(&ordered.coeffs);
        terms
            .filter_map(|(string, &coeff)| value(string, coeff))
            .fold(C::ZERO, C::plus)
    }

    /// The expectation value `<0...0| O |0...0>`: the sum of the coefficients of the strings
    /// made of `I` and `Z` only.
    #[must_use]
    pub fn overlap_with_zero(&self) -> C {
        self.sum_in_order(|p, c| (!p.has_x_or_y()).then_some(c))
    }

    /// The expectation value `<+...+| O |+...+>`: the sum of the coefficients of the strings
    /// made of `I` and `X` only.
    #[must_use]
    pub fn overlap_with_plus(&self) -> C {
        self.sum_in_order(|p, c| (!p.has_y_or_z()).then_some(c))
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
        Ok(self.sum_in_order(|p, c| {
            (!p.has_x_or_y()).then(|| if p.commutes(&flips) { c } else { c.scale(-1.0) })
        }))
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
        let (a, b) = (self.in_order(), other.in_order());
        let (mut i, mut j) = (0, 0);
        let mut total = C::ZERO;
        while i < a.strings.len() && j < b.strings.len() {
            match a.strings[i].cmp(&b.strings[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    total = total.plus(a.coeffs[i].times(b.coeffs[j]));
                    i += 1;
                    j += 1;
                }
            }
        }
        total
    }

    /// The sum of the squared absolute values of the coefficients, which unitary gates
    /// preserve when nothing is truncated.
    #[must_use]
    pub fn norm_sqr(&self) -> f64 {
        let mut squares: Vec<f64> = self.iter().map(|(_, c)| c.norm_sqr()).collect();
        squares.sort_unstable_by(f64::total_cmp);
        squares.iter().sum()
    }

    /// The largest absolute value of a coefficient, zero for the empty sum.
    #[must_use]
    pub fn max_abs_coeff(&self) -> f64 {
        self.iter().map(|(_, c)| c.abs()).fold(0.0, f64::max)
    }
}
