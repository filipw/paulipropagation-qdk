//! A propagation with the angles of the rotations left open, to be evaluated for many
//! choices of the angles.

use crate::montecarlo::PathLimits;
use crate::{Error, Gate, PauliString, PauliSum};
use rustc_hash::FxHashMap;

/// No node.
const NONE: u32 = u32::MAX;

/// The coefficient of a Pauli string after a rotation, as a function of the coefficients
/// before it: `cos(theta) * cos_sign * [cos_from] + sin(theta) * sin_sign * [sin_from]`,
/// where `theta` is the angle of the rotation and either part may be missing.
#[derive(Clone, Copy, Debug)]
struct Node {
    parameter: u32,
    cos_from: u32,
    sin_from: u32,
    cos_sign: f64,
    sin_sign: f64,
}

/// A term of the sum while the surrogate is built.
#[derive(Clone, Copy)]
struct Term {
    node: u32,
    /// The sign that Clifford gates have given the term since its node was made.
    sign: f64,
    freq: u32,
    sins: u32,
}

/// The propagation of an observable through a circuit of Clifford gates and rotations, for
/// any angles of the rotations.
///
/// Which Pauli strings a propagation reaches, and which of them a truncation by weight or
/// by the number of rotations drops, does not depend on the angles. A surrogate records
/// how every coefficient is made of sines and cosines, once, and [`Surrogate::evaluate`]
/// then gives the propagated sum for given angles without propagating again. It agrees with
/// [`crate::propagate`] under the same `max_weight`, `max_freq` and `max_sins` and no
/// truncation by coefficients.
#[derive(Clone, Debug)]
pub struct Surrogate<const W: usize> {
    num_qubits: usize,
    num_parameters: usize,
    /// The coefficients of the observable, which are the values of the first nodes.
    leaves: Vec<f64>,
    /// The other nodes, each after the nodes it is made of.
    nodes: Vec<Node>,
    /// The Pauli strings of the result in ascending order, with the node and the sign of
    /// the coefficient of each.
    terms: Vec<(PauliString<W>, u32, f64)>,
}

impl<const W: usize> Surrogate<W> {
    /// Propagates `observable` through `circuit` as an observable, last gate first, for all
    /// angles at once. The angles that the rotations of `circuit` have are not used.
    ///
    /// After every gate the terms that `limits` or `custom` select are dropped. `custom`
    /// reads the Pauli string only.
    pub fn new(
        circuit: &[Gate],
        observable: &PauliSum<W>,
        limits: &PathLimits,
        custom: Option<&dyn Fn(&PauliString<W>) -> bool>,
    ) -> Result<Self, Error> {
        let num_qubits = observable.num_qubits();
        let mut parameters = Vec::with_capacity(circuit.len());
        let mut num_parameters = 0;
        for gate in circuit {
            gate.validate(num_qubits)?;
            if gate.is_noise() || matches!(gate, Gate::TransferMap { .. }) {
                return Err(Error::Invalid(
                    "a surrogate takes only Clifford gates and Pauli rotations",
                ));
            }
            parameters.push(num_parameters);
            if matches!(gate, Gate::Rotation { .. }) {
                num_parameters += 1;
            }
        }
        let dropped = |string: &PauliString<W>, term: &Term| {
            limits.max_weight.is_some_and(|max| string.weight() > max)
                || limits.max_freq.is_some_and(|max| term.freq > max)
                || limits.max_sins.is_some_and(|max| term.sins > max)
                || custom.is_some_and(|custom| custom(string))
        };

        let start = observable.sorted_terms();
        let leaves: Vec<f64> = start.iter().map(|term| term.1).collect();
        let mut nodes: Vec<Node> = Vec::new();
        let mut terms: FxHashMap<PauliString<W>, Term> = FxHashMap::default();
        for (index, (string, _)) in start.iter().enumerate() {
            let term = Term {
                node: index as u32,
                sign: 1.0,
                freq: 0,
                sins: 0,
            };
            terms.insert(*string, term);
        }

        for (gate, &parameter) in circuit.iter().zip(&parameters).rev() {
            if let Gate::Rotation { generator, .. } = gate {
                let generator = PauliString::<W>::from_ops(generator);
                let split: Vec<(PauliString<W>, Term)> = terms
                    .iter()
                    .filter(|(string, _)| !string.commutes(&generator))
                    .map(|(string, term)| (*string, *term))
                    .collect();
                if nodes.len() + leaves.len() + 2 * split.len() >= NONE as usize {
                    return Err(Error::Invalid("the surrogate has too many nodes"));
                }
                let new_node = |nodes: &mut Vec<Node>| {
                    nodes.push(Node {
                        parameter: parameter as u32,
                        cos_from: NONE,
                        sin_from: NONE,
                        cos_sign: 0.0,
                        sin_sign: 0.0,
                    });
                    (leaves.len() + nodes.len() - 1) as u32
                };
                // The cosine branch stays on the string.
                for (string, old) in &split {
                    let node = new_node(&mut nodes);
                    let made = &mut nodes[node as usize - leaves.len()];
                    made.cos_from = old.node;
                    made.cos_sign = old.sign;
                    let term = Term {
                        node,
                        sign: 1.0,
                        freq: old.freq + 1,
                        sins: old.sins,
                    };
                    terms.insert(*string, term);
                }
                // The sine branch goes to the string times the generator, where it meets the
                // cosine branch of that string if the sum had it.
                for (string, old) in &split {
                    // G P = i^k P' with k odd, and i G P is -P' for k = 1 and +P' for k = 3.
                    let (product, k) = generator.mul(string);
                    let product_sign = if k == 3 { 1.0 } else { -1.0 };
                    let (freq, sins) = (old.freq + 1, old.sins + 1);
                    let term = match terms.get(&product) {
                        Some(met) => Term {
                            freq: met.freq.min(freq),
                            sins: met.sins.min(sins),
                            ..*met
                        },
                        None => Term {
                            node: new_node(&mut nodes),
                            sign: 1.0,
                            freq,
                            sins,
                        },
                    };
                    let made = &mut nodes[term.node as usize - leaves.len()];
                    made.sin_from = old.node;
                    made.sin_sign = old.sign * product_sign;
                    terms.insert(product, term);
                }
            } else {
                let moved = terms.drain().map(|(mut string, mut term)| {
                    if gate.conjugate_clifford(&mut string) {
                        term.sign = -term.sign;
                    }
                    (string, term)
                });
                terms = moved.collect();
            }
            terms.retain(|string, term| !dropped(string, term));
        }

        let mut terms: Vec<(PauliString<W>, u32, f64)> = terms
            .into_iter()
            .map(|(string, term)| (string, term.node, term.sign))
            .collect();
        terms.sort_unstable_by_key(|term| term.0);
        let mut surrogate = Self {
            num_qubits,
            num_parameters,
            leaves,
            nodes,
            terms,
        };
        surrogate.drop_unused_nodes();
        Ok(surrogate)
    }

    /// Removes the nodes that no term of the result is made of, which are those of the
    /// terms that were dropped on the way.
    fn drop_unused_nodes(&mut self) {
        let first = self.leaves.len();
        let mut used = vec![false; first + self.nodes.len()];
        for term in &self.terms {
            used[term.1 as usize] = true;
        }
        for (index, node) in self.nodes.iter().enumerate().rev() {
            if used[first + index] {
                for from in [node.cos_from, node.sin_from] {
                    if from != NONE {
                        used[from as usize] = true;
                    }
                }
            }
        }
        // The leaves keep their places, so that they stay the coefficients of the observable.
        let mut renumbered = vec![NONE; used.len()];
        let mut next = first as u32;
        for (index, slot) in renumbered.iter_mut().enumerate() {
            if index < first {
                *slot = index as u32;
            } else if used[index] {
                *slot = next;
                next += 1;
            }
        }
        let renumber = |from: u32| {
            if from == NONE {
                NONE
            } else {
                renumbered[from as usize]
            }
        };
        let mut index = first;
        self.nodes.retain_mut(|node| {
            node.cos_from = renumber(node.cos_from);
            node.sin_from = renumber(node.sin_from);
            index += 1;
            used[index - 1]
        });
        for term in &mut self.terms {
            term.1 = renumbered[term.1 as usize];
        }
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The number of rotations of the circuit, each of which has an angle.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.num_parameters
    }

    /// The number of terms of the propagated sum.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// The number of sums of a cosine and a sine part that an evaluation computes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The propagated sum for the given angles, one for every rotation of the circuit in
    /// the order of the circuit.
    pub fn evaluate(&self, thetas: &[f64]) -> Result<PauliSum<W>, Error> {
        if thetas.len() != self.num_parameters {
            return Err(Error::Invalid(
                "there must be one angle for every rotation of the circuit",
            ));
        }
        if thetas.iter().any(|theta| !theta.is_finite()) {
            return Err(Error::NonFiniteAngle);
        }
        let trig: Vec<(f64, f64)> = thetas.iter().map(|theta| theta.sin_cos()).collect();
        let mut values = Vec::with_capacity(self.leaves.len() + self.nodes.len());
        values.extend_from_slice(&self.leaves);
        for node in &self.nodes {
            let (sin, cos) = trig[node.parameter as usize];
            let mut value = 0.0;
            if node.cos_from != NONE {
                value = values[node.cos_from as usize] * node.cos_sign * cos;
            }
            if node.sin_from != NONE {
                let part = values[node.sin_from as usize] * node.sin_sign * sin;
                value = if node.cos_from == NONE {
                    part
                } else {
                    value + part
                };
            }
            values.push(value);
        }
        let mut sum = PauliSum::new(self.num_qubits)?;
        let strings = self.terms.iter().map(|term| term.0).collect();
        let coeffs = self
            .terms
            .iter()
            .map(|term| values[term.1 as usize] * term.2);
        sum.set_ordered(strings, coeffs.collect());
        Ok(sum)
    }
}
