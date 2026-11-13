//! Checks gradients against the parameter-shift rule evaluated with dense matrices: the
//! derivative of an expectation value with respect to the angle of a Pauli rotation is half
//! the difference of the values with the angle shifted by `pi / 2` either way.

mod common;

use common::*;
use pauliprop::{gradient, propagate, Gate, PauliOp, PauliSum, State, Truncation};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::FRAC_PI_2;

fn matrix_of(sum: &PauliSum<1>) -> Matrix {
    let n = sum.num_qubits();
    let mut matrix = zeros(1 << n);
    for (string, coeff) in sum.sorted_terms() {
        let ops: Vec<PauliOp> = (0..n).map(|q| string.get(q)).collect();
        for (row, string_row) in matrix.iter_mut().zip(&string_matrix(&ops)) {
            for (value, string_value) in row.iter_mut().zip(string_row) {
                *value += string_value * coeff;
            }
        }
    }
    matrix
}

fn random_sum(rng: &mut StdRng, n: usize, terms: usize) -> PauliSum<1> {
    let mut sum = PauliSum::<1>::new(n).unwrap();
    for _ in 0..terms {
        let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
        sum.add(&sparse(&ops), rng.gen_range(-1.0..1.0)).unwrap();
    }
    sum
}

/// The density matrix of a state.
fn density(state: &State<'_, 1>, n: usize) -> Matrix {
    let dim = 1usize << n;
    let mut rho = zeros(dim);
    match state {
        State::Zero => rho[0][0] = c(1.0, 0.0),
        State::Plus => rho = vec![vec![c(1.0 / dim as f64, 0.0); dim]; dim],
        State::Computational(ones) => {
            let index = (0..dim)
                .find(|&i| (0..n).all(|q| (bit(i, q, n) == 1) == ones.contains(&q)))
                .unwrap();
            rho[index][index] = c(1.0, 0.0);
        }
        State::MaxMixed => rho = scaled(&identity(dim), 1.0 / dim as f64),
        State::Sum(sum) => rho = matrix_of(sum),
    }
    rho
}

/// `Tr[rho U† O U]` with dense matrices.
fn dense_value(circuit: &[Gate], observable: &Matrix, rho: &Matrix, n: usize) -> f64 {
    let u = circuit_matrix(circuit, n);
    let evolved = mul(&mul(&dagger(&u), observable), &u);
    trace_of_product(rho, &evolved).re
}

fn shifted(circuit: &[Gate], rotation: usize, by: f64) -> Vec<Gate> {
    let mut circuit = circuit.to_vec();
    let rotations = circuit.iter_mut().filter_map(|gate| match gate {
        Gate::Rotation { theta, .. } => Some(theta),
        _ => None,
    });
    *rotations.into_iter().nth(rotation).unwrap() += by;
    circuit
}

fn assert_gradient(circuit: &[Gate], observable: &PauliSum<1>, state: &State<'_, 1>, what: &str) {
    let n = observable.num_qubits();
    let (value, derivatives) = gradient(circuit, observable, &Truncation::none(), state).unwrap();
    let (o, rho) = (matrix_of(observable), density(state, n));

    let expected = dense_value(circuit, &o, &rho, n);
    assert!(
        (value - expected).abs() < 1e-11,
        "{what}: value {value}, dense {expected}"
    );

    let rotations = circuit
        .iter()
        .filter(|g| matches!(g, Gate::Rotation { .. }))
        .count();
    assert_eq!(derivatives.len(), rotations, "{what}");
    for (k, derivative) in derivatives.iter().enumerate() {
        let up = dense_value(&shifted(circuit, k, FRAC_PI_2), &o, &rho, n);
        let down = dense_value(&shifted(circuit, k, -FRAC_PI_2), &o, &rho, n);
        let expected = (up - down) / 2.0;
        assert!(
            (derivative - expected).abs() < 1e-11,
            "{what}: derivative {k} is {derivative}, parameter shift gives {expected}"
        );
    }
}

#[test]
fn gradients_of_random_circuits_match_the_parameter_shift_rule() {
    let mut rng = StdRng::seed_from_u64(31);
    let mut nonzero = 0;
    for round in 0..40 {
        let n = 2 + round % 3;
        let circuit: Vec<Gate> = (0..rng.gen_range(1..14))
            .map(|_| random_gate(&mut rng, n))
            .collect();
        let observable = random_sum(&mut rng, n, 1 + round % 5);
        let ones: Vec<usize> = (0..n).filter(|_| rng.gen_bool(0.5)).collect();
        let mixture = random_sum(&mut rng, n, 6);
        let states = [
            State::Zero,
            State::Plus,
            State::Computational(&ones),
            State::MaxMixed,
            State::Sum(&mixture),
        ];
        for state in &states {
            assert_gradient(
                &circuit,
                &observable,
                state,
                &format!("round {round}, {state:?}"),
            );
            let (_, derivatives) =
                gradient(&circuit, &observable, &Truncation::none(), state).unwrap();
            nonzero += derivatives.iter().filter(|d| d.abs() > 1e-3).count();
        }
    }
    assert!(nonzero > 100, "only {nonzero} derivatives were not zero");
}

#[test]
fn gradients_through_gates_given_as_matrices_match_the_parameter_shift_rule() {
    let mut rng = StdRng::seed_from_u64(32);
    let n = 3;
    for round in 0..10 {
        let mut circuit = Vec::new();
        let mut dense = Vec::new();
        for _ in 0..8 {
            let gate = random_gate(&mut rng, n);
            dense.push(gate.clone());
            // Every other gate that is not a rotation goes in as its matrix on two qubits.
            if !matches!(gate, Gate::Rotation { .. }) && rng.gen_bool(0.5) {
                let u = circuit_matrix(std::slice::from_ref(&gate), n);
                let flat: Vec<_> = u.into_iter().flatten().collect();
                circuit.push(Gate::unitary(&[0, 1, 2], &flat).unwrap());
            } else {
                circuit.push(gate);
            }
        }
        let observable = random_sum(&mut rng, n, 3);
        let (value, derivatives) =
            gradient(&circuit, &observable, &Truncation::none(), &State::Zero).unwrap();
        let (expected_value, expected) =
            gradient(&dense, &observable, &Truncation::none(), &State::Zero).unwrap();
        assert_gradient(&dense, &observable, &State::Zero, &format!("round {round}"));
        assert!((value - expected_value).abs() < 1e-11);
        for (a, b) in derivatives.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-11);
        }
    }
}

#[test]
fn the_value_is_that_of_the_truncated_propagation() {
    let mut rng = StdRng::seed_from_u64(33);
    let n = 8;
    let truncation = Truncation {
        max_weight: Some(3),
        min_abs_coeff: 1e-3,
        ..Truncation::none()
    };
    let mut circuit = Vec::new();
    for _ in 0..4 {
        circuit.extend((0..n - 1).map(|q| Gate::rzz(rng.gen_range(0.2..0.6), q, q + 1)));
        circuit.extend((0..n).map(|q| Gate::rx(rng.gen_range(0.2..0.6), q)));
        circuit.push(Gate::Cx(2, 5));
    }
    let mut observable = PauliSum::<1>::new(n).unwrap();
    observable.add(&[(3, PauliOp::Z)], 1.0).unwrap();

    let (value, derivatives) = gradient(&circuit, &observable, &truncation, &State::Zero).unwrap();
    let mut propagated = observable.clone();
    propagate(&circuit, &mut propagated, &truncation).unwrap();
    assert_eq!(value, propagated.overlap_with_zero());

    // The truncation drops terms, and the gradient stays close to the exact one.
    let mut exact = observable.clone();
    propagate(&circuit, &mut exact, &Truncation::none()).unwrap();
    assert!(propagated.len() < exact.len());
    let (_, exact_derivatives) =
        gradient(&circuit, &observable, &Truncation::none(), &State::Zero).unwrap();
    let error: f64 = derivatives
        .iter()
        .zip(&exact_derivatives)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let size: f64 = exact_derivatives.iter().map(|d| d.abs()).sum();
    assert!(size > 0.1 && error < 0.2 * size, "error {error} of {size}");
}

#[test]
fn circuits_that_cannot_be_undone_are_rejected() {
    let mut observable = PauliSum::<1>::new(2).unwrap();
    observable.add(&[(0, PauliOp::Z)], 1.0).unwrap();
    let none = Truncation::none();
    let rotation = Gate::rx(0.3, 0);

    let noisy = [rotation.clone(), Gate::depolarizing(0.1, 0).unwrap()];
    assert!(gradient(&noisy, &observable, &none, &State::Zero).is_err());
    let damped = [rotation.clone(), Gate::amplitude_damping(0.1, 0).unwrap()];
    assert!(gradient(&damped, &observable, &none, &State::Zero).is_err());
    // A transfer map that is not that of a unitary: Z is halved.
    let mut matrix = vec![0.0; 16];
    for (i, factor) in [1.0, 1.0, 1.0, 0.5].into_iter().enumerate() {
        matrix[i * 4 + i] = factor;
    }
    let shrinking = [rotation.clone(), Gate::transfer_map(&[0], &matrix).unwrap()];
    assert!(gradient(&shrinking, &observable, &none, &State::Zero).is_err());
    // A transfer map that keeps the size of every Pauli operator and still is not that of
    // a unitary: X and Y both become X.
    let mut matrix = vec![0.0; 16];
    for (to, from) in [(0, 0), (1, 1), (1, 2), (3, 3)] {
        matrix[to * 4 + from] = 1.0;
    }
    let folding = [rotation.clone(), Gate::transfer_map(&[0], &matrix).unwrap()];
    assert!(gradient(&folding, &observable, &none, &State::Zero).is_err());

    let counted = Truncation {
        max_freq: Some(3),
        ..none
    };
    assert!(gradient(
        std::slice::from_ref(&rotation),
        &observable,
        &counted,
        &State::Zero
    )
    .is_err());
    let counted = Truncation {
        max_sins: Some(3),
        ..none
    };
    assert!(gradient(
        std::slice::from_ref(&rotation),
        &observable,
        &counted,
        &State::Zero
    )
    .is_err());
    assert!(gradient(
        std::slice::from_ref(&rotation),
        &observable,
        &none,
        &State::Computational(&[2])
    )
    .is_err());
    let other = PauliSum::<1>::new(3).unwrap();
    assert!(gradient(&[rotation], &observable, &none, &State::Sum(&other)).is_err());
}
