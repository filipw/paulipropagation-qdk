//! The gradient of an expectation value with respect to the angles of the rotations.

use crate::{
    propagate_in_zones, propagate_with, Error, Gate, PauliOp, PauliString, PauliSum, Picture,
    Truncation, Zones,
};
use std::cmp::Ordering;

/// A state that the expectation value of a Pauli sum is taken in.
#[derive(Clone, Copy, Debug)]
pub enum State<'a, const W: usize> {
    /// Every qubit in the state 0.
    Zero,
    /// Every qubit in the state `(|0> + |1>) / sqrt(2)`.
    Plus,
    /// The computational basis state whose qubits listed here are 1 and whose others are 0.
    Computational(&'a [usize]),
    /// The maximally mixed state.
    MaxMixed,
    /// A state given as a Pauli sum, whose coefficient of the Pauli string `P` is
    /// `Tr[rho P] / 2^n`.
    Sum(&'a PauliSum<W>),
}

impl<const W: usize> State<'_, W> {
    /// The expectation value of a Pauli string on `num_qubits` qubits in the state.
    #[must_use]
    pub fn expectation_of(&self, string: &PauliString<W>, num_qubits: usize) -> f64 {
        match self {
            State::Zero => f64::from(u8::from(!string.has_x_or_y())),
            State::Plus => f64::from(u8::from(!string.has_y_or_z())),
            State::Computational(ones) => {
                if string.has_x_or_y() {
                    0.0
                } else if ones.iter().filter(|&&q| string.z_bit(q)).count() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            State::MaxMixed => f64::from(u8::from(*string == PauliString::identity())),
            State::Sum(state) => state.coeff(string) * 2f64.powi(num_qubits as i32),
        }
    }

    fn validate(&self, num_qubits: usize) -> Result<(), Error> {
        match self {
            State::Computational(ones) => match ones.iter().find(|&&q| q >= num_qubits) {
                Some(&qubit) => Err(Error::QubitOutOfRange { qubit, num_qubits }),
                None => Ok(()),
            },
            State::Sum(state) if state.num_qubits() != num_qubits => Err(Error::Invalid(
                "the state and the observable are on different numbers of qubits",
            )),
            _ => Ok(()),
        }
    }
}

/// Whether applying the gate to a state and then to an observable gives the observable back.
fn is_undone_by_its_adjoint(gate: &Gate) -> bool {
    match gate {
        Gate::PauliChannel { .. } | Gate::AmplitudeDamping { .. } => false,
        // A transfer map is undone by its transpose exactly if its matrix is orthogonal,
        // which the map of a unitary gate is.
        Gate::TransferMap { columns, .. } => {
            let size = columns.len();
            let mut dense = vec![0.0; size * size];
            for (j, column) in columns.iter().enumerate() {
                for &(i, factor) in column {
                    dense[i * size + j] = factor;
                }
            }
            (0..size).all(|j| {
                (0..size).all(|k| {
                    let dot: f64 = (0..size)
                        .map(|i| dense[i * size + j] * dense[i * size + k])
                        .sum();
                    (dot - f64::from(u8::from(j == k))).abs() < 1e-9
                })
            })
        }
        _ => true,
    }
}

/// The expectation value of `observable` after `circuit` has acted on `state`, and its
/// derivatives with respect to the angle of every rotation of the circuit, in the order of
/// the rotations in the circuit.
///
/// The observable is propagated through the circuit once, with the truncation, and then
/// taken back through it gate by gate together with the state, which gives the derivative
/// at every rotation: two passes through the circuit, whatever the number of rotations.
/// Only the first of them is divided among threads. Without truncation the gradient is
/// exact. With truncation it is the gradient of the truncated propagation only
/// approximately, because the terms that were dropped on the way do not come back.
///
/// The circuit must be unitary: noise channels cannot be taken back. `max_freq` and
/// `max_sins` are not supported.
pub fn gradient<const W: usize>(
    circuit: &[Gate],
    observable: &PauliSum<W>,
    truncation: &Truncation,
    state: &State<'_, W>,
) -> Result<(f64, Vec<f64>), Error> {
    let num_qubits = observable.num_qubits();
    state.validate(num_qubits)?;
    if truncation.max_freq.is_some() || truncation.max_sins.is_some() {
        return Err(Error::Invalid(
            "a gradient cannot be truncated by max_freq or max_sins",
        ));
    }
    if !circuit.iter().all(is_undone_by_its_adjoint) {
        return Err(Error::Invalid(
            "a gradient needs a unitary circuit: noise channels cannot be undone",
        ));
    }

    let mut operator = observable.clone();
    propagate_with(
        circuit,
        &mut operator,
        truncation,
        Picture::Heisenberg,
        None,
    )?;

    // The state as the value it gives to every Pauli string of the propagated observable.
    let (strings, coeffs) = operator.take_ordered();
    let values: Vec<f64> = strings
        .iter()
        .map(|string| state.expectation_of(string, num_qubits))
        .collect();
    let value = coeffs.iter().zip(&values).map(|(c, v)| c * v).sum();
    let mut dual = PauliSum::new(num_qubits)?;
    dual.set_ordered(strings.clone(), values);
    operator.set_ordered(strings, coeffs);

    // Going through the circuit from its first gate, the observable is taken back to where
    // it was before each gate and the state is taken forward through it.
    let mut derivatives = Vec::new();
    for gate in circuit {
        if let Gate::Rotation { generator, .. } = gate {
            derivatives.push(commutator_value(generator, &operator, &dual));
        }
        let gate = std::slice::from_ref(gate);
        let back = Picture::Schrodinger;
        propagate_in_zones(gate, &mut operator, truncation, back, None, &Zones::ONE)?;
        let all = Truncation::none();
        propagate_in_zones(gate, &mut dual, &all, back, None, &Zones::ONE)?;
        keep_strings_of(&mut dual, &mut operator);
    }
    Ok((value, derivatives))
}

/// `(i / 2) * dual([G, operator])`, the derivative with respect to the angle of the rotation
/// `exp(-i theta / 2 G)` that the operator has just been propagated through.
fn commutator_value<const W: usize>(
    generator: &[(usize, PauliOp)],
    operator: &PauliSum<W>,
    dual: &PauliSum<W>,
) -> f64 {
    let generator = PauliString::<W>::from_ops(generator);
    let mut total = 0.0;
    // A propagated sum has its terms in the order of the Pauli strings.
    for (string, coeff) in operator.iter() {
        if string.commutes(&generator) {
            continue;
        }
        // [G, P] = 2 G P = 2 i^power Q with an odd power, so (i / 2) [G, P] is -Q or +Q.
        let (product, power) = generator.mul(string);
        let sign = if power == 1 { -1.0 } else { 1.0 };
        total += sign * coeff * dual.coeff(&product);
    }
    total
}

/// Drops the terms of `dual` whose Pauli strings are not in `operator`.
fn keep_strings_of<const W: usize>(dual: &mut PauliSum<W>, operator: &mut PauliSum<W>) {
    let (kept, coeffs) = operator.take_ordered();
    let (strings, values) = dual.take_ordered();
    if strings == kept {
        dual.set_ordered(strings, values);
        operator.set_ordered(kept, coeffs);
        return;
    }
    let mut shared = (Vec::new(), Vec::new());
    let mut next = 0;
    for (string, value) in strings.into_iter().zip(values) {
        while next < kept.len() && kept[next].cmp(&string) == Ordering::Less {
            next += 1;
        }
        if next < kept.len() && kept[next] == string {
            shared.0.push(string);
            shared.1.push(value);
        }
    }
    dual.set_ordered(shared.0, shared.1);
    operator.set_ordered(kept, coeffs);
}
