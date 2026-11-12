//! Checks noise channels against their Kraus operators.
//!
//! A channel acts on a state as `rho -> sum_k K_k rho K_k†`, so it acts on an observable as
//! `O -> sum_k K_k† O K_k`. The reference builds the Kraus operators as dense matrices from
//! the description of the channel and never from the table the engine derives.

mod common;

use common::*;
use pauliprop::{propagate, Gate, PauliOp, PauliString, PauliSum, Truncation};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Pauli errors as `(string, probability)` pairs.
type Errors = Vec<(Vec<PauliOp>, f64)>;

/// One step of a circuit: the gate for the engine and its Kraus operators for the reference.
struct Step {
    gate: Gate,
    kraus: Vec<Matrix>,
}

fn scaled(matrix: &Matrix, factor: f64) -> Matrix {
    matrix
        .iter()
        .map(|row| row.iter().map(|&v| v * factor).collect())
        .collect()
}

fn unitary_step(gate: Gate, n: usize) -> Step {
    let kraus = vec![gate_matrix(&gate, n)];
    Step { gate, kraus }
}

/// A random Pauli error: the string `errors[k].0` on `qubits` with probability `errors[k].1`.
fn pauli_channel_step(qubits: &[usize], errors: &[(Vec<PauliOp>, f64)], n: usize) -> Step {
    let total: f64 = errors.iter().map(|(_, p)| p).sum();
    let mut kraus = vec![scaled(&identity(1 << n), (1.0 - total).sqrt())];
    for (ops, probability) in errors {
        let mut full = vec![PauliOp::I; n];
        for (&q, &op) in qubits.iter().zip(ops) {
            full[q] = op;
        }
        kraus.push(scaled(&string_matrix(&full), probability.sqrt()));
    }
    Step {
        gate: Gate::pauli_channel(qubits, errors).unwrap(),
        kraus,
    }
}

fn amplitude_damping_step(gamma: f64, q: usize, n: usize) -> Step {
    let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
    let k0 = vec![vec![l, o], vec![o, c((1.0 - gamma).sqrt(), 0.0)]];
    let k1 = vec![vec![o, c(gamma.sqrt(), 0.0)], vec![o, o]];
    Step {
        gate: Gate::amplitude_damping(gamma, q).unwrap(),
        kraus: vec![embed(&k0, &[q], n), embed(&k1, &[q], n)],
    }
}

/// Random probabilities for `count` errors that sum to less than one.
fn random_probabilities(rng: &mut StdRng, count: usize) -> Vec<f64> {
    let weights: Vec<f64> = (0..count).map(|_| rng.gen_range(0.0..1.0)).collect();
    let scale = rng.gen_range(0.1..0.9) / weights.iter().sum::<f64>();
    weights.iter().map(|w| w * scale).collect()
}

/// A random Pauli channel on `qubits` with an error probability for every Pauli string.
fn random_pauli_channel_step(rng: &mut StdRng, qubits: &[usize], n: usize) -> Step {
    let k = qubits.len();
    let strings: Vec<Vec<PauliOp>> = (1..1 << (2 * k)).map(|i| ops_of_index(i, k)).collect();
    let probabilities = random_probabilities(rng, strings.len());
    let errors: Vec<_> = strings.into_iter().zip(probabilities).collect();
    pauli_channel_step(qubits, &errors, n)
}

/// Asserts that the engine maps every Pauli string on `n` qubits as the Kraus operators do.
fn assert_matches_kraus(steps: &[Step], n: usize, what: &str) {
    let dim = (1usize << n) as f64;
    let basis: Vec<Matrix> = (0..1 << (2 * n))
        .map(|index| string_matrix(&ops_of_index(index, n)))
        .collect();
    let circuit: Vec<Gate> = steps.iter().map(|step| step.gate.clone()).collect();

    for input in 0..1 << (2 * n) {
        let input_ops = ops_of_index(input, n);

        let mut sum = PauliSum::<1>::new(n).unwrap();
        sum.add(&sparse(&input_ops), 1.0).unwrap();
        propagate(&circuit, &mut sum, &Truncation::none()).unwrap();

        // The last step acts on the observable first.
        let mut evolved = basis[input].clone();
        for step in steps.iter().rev() {
            let mut next = zeros(1 << n);
            for k in &step.kraus {
                let term = mul(&mul(&dagger(k), &evolved), k);
                for (row, term_row) in next.iter_mut().zip(&term) {
                    for (value, term_value) in row.iter_mut().zip(term_row) {
                        *value += term_value;
                    }
                }
            }
            evolved = next;
        }

        for (output, q_matrix) in basis.iter().enumerate() {
            let expected = trace_of_product(q_matrix, &evolved) / dim;
            assert!(
                expected.im.abs() < TOL,
                "{what}: reference coefficient is not real"
            );
            let string = PauliString::<1>::from_ops(&sparse(&ops_of_index(output, n)));
            let actual = sum.coeff(&string);
            assert!(
                (actual - expected.re).abs() < TOL,
                "{what}: {:?} -> {:?}: engine {actual}, reference {}",
                input_ops,
                ops_of_index(output, n),
                expected.re
            );
        }
    }
}

#[test]
fn pauli_channels_match_their_kraus_operators() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(11);
    // One, two and three qubits, in every order, so that the order of an error string is checked.
    let supports: [&[usize]; 12] = [
        &[0],
        &[1],
        &[2],
        &[0, 1],
        &[1, 0],
        &[0, 2],
        &[2, 0],
        &[1, 2],
        &[2, 1],
        &[0, 1, 2],
        &[2, 0, 1],
        &[1, 2, 0],
    ];
    for qubits in supports {
        for _ in 0..3 {
            let step = random_pauli_channel_step(&mut rng, qubits, n);
            assert_matches_kraus(&[step], n, &format!("Pauli channel on {qubits:?}"));
        }
    }

    // A single correlated error.
    let step = pauli_channel_step(&[2, 0], &[(vec![PauliOp::X, PauliOp::Z], 0.3)], n);
    assert_matches_kraus(&[step], n, "XZ error on qubits 2 and 0");
}

#[test]
fn named_noise_channels_match_their_kraus_operators() {
    use PauliOp::{X, Y, Z};
    let n = 2;
    for q in 0..n {
        for strength in [0.0, 0.07, 0.5, 1.0] {
            let cases: [(Gate, Errors, &str); 4] = [
                (
                    Gate::depolarizing(strength, q).unwrap(),
                    vec![
                        (vec![X], strength / 4.0),
                        (vec![Y], strength / 4.0),
                        (vec![Z], strength / 4.0),
                    ],
                    "depolarizing",
                ),
                (
                    Gate::pauli_x_noise(strength, q).unwrap(),
                    vec![(vec![X], strength / 2.0)],
                    "Pauli X noise",
                ),
                (
                    Gate::pauli_y_noise(strength, q).unwrap(),
                    vec![(vec![Y], strength / 2.0)],
                    "Pauli Y noise",
                ),
                (
                    Gate::pauli_z_noise(strength, q).unwrap(),
                    vec![(vec![Z], strength / 2.0)],
                    "Pauli Z noise",
                ),
            ];
            for (gate, errors, what) in cases {
                let reference = pauli_channel_step(&[q], &errors, n);
                let step = Step {
                    gate,
                    kraus: reference.kraus,
                };
                assert_matches_kraus(&[step], n, what);
            }
            let step = amplitude_damping_step(strength, q, n);
            assert_matches_kraus(&[step], n, "amplitude damping");
        }
    }
}

#[test]
fn noisy_random_circuits_match_their_kraus_operators() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(31);
    for trial in 0..10 {
        let steps: Vec<Step> = (0..30)
            .map(|_| {
                let q = rng.gen_range(0..n);
                let mut other = rng.gen_range(0..n - 1);
                if other >= q {
                    other += 1;
                }
                match rng.gen_range(0..6) {
                    0 => random_pauli_channel_step(&mut rng, &[q], n),
                    1 => random_pauli_channel_step(&mut rng, &[q, other], n),
                    2 => amplitude_damping_step(rng.gen_range(0.0..1.0), q, n),
                    _ => unitary_step(random_gate(&mut rng, n), n),
                }
            })
            .collect();
        assert!(steps.iter().any(|step| step.gate.is_noise()));
        assert_matches_kraus(&steps, n, &format!("noisy random circuit {trial}"));
    }
}

#[test]
fn depolarizing_noise_shrinks_the_norm_of_an_observable() {
    let n = 5;
    let mut rng = StdRng::seed_from_u64(3);
    let mut circuit = Vec::new();
    for _ in 0..60 {
        circuit.push(random_gate(&mut rng, n));
        let q = rng.gen_range(0..n);
        circuit.push(Gate::depolarizing(rng.gen_range(0.0..0.2), q).unwrap());
    }
    let noiseless: Vec<Gate> = circuit.iter().filter(|g| !g.is_noise()).cloned().collect();

    let mut noisy = PauliSum::<1>::new(n).unwrap();
    noisy.add(&[(2, PauliOp::Z)], 1.0).unwrap();
    let mut clean = noisy.clone();
    propagate(&circuit, &mut noisy, &Truncation::none()).unwrap();
    propagate(&noiseless, &mut clean, &Truncation::none()).unwrap();

    assert!((clean.norm_sqr() - 1.0).abs() < 1e-11);
    assert!(noisy.norm_sqr() < clean.norm_sqr());
}

#[test]
fn invalid_noise_is_rejected() {
    use PauliOp::{X, Z};
    for bad in [-0.1, 1.1, f64::NAN] {
        assert!(Gate::depolarizing(bad, 0).is_err());
        assert!(Gate::pauli_x_noise(bad, 0).is_err());
        assert!(Gate::pauli_y_noise(bad, 0).is_err());
        assert!(Gate::pauli_z_noise(bad, 0).is_err());
        assert!(Gate::amplitude_damping(bad, 0).is_err());
        assert!(Gate::pauli_channel(&[0], &[(vec![X], bad)]).is_err());
    }
    // More operators than qubits, probabilities above one, no qubits.
    assert!(Gate::pauli_channel(&[0], &[(vec![X, Z], 0.1)]).is_err());
    assert!(Gate::pauli_channel(&[0], &[(vec![X], 0.6), (vec![Z], 0.6)]).is_err());
    assert!(Gate::pauli_channel(&[], &[]).is_err());

    let mut sum = PauliSum::<1>::new(2).unwrap();
    sum.add(&[(0, Z)], 1.0).unwrap();
    let none = Truncation::none();
    let out_of_range = Gate::depolarizing(0.1, 2).unwrap();
    assert!(propagate(&[out_of_range], &mut sum, &none).is_err());
    let repeated = Gate::pauli_channel(&[1, 1], &[(vec![X, X], 0.1)]).unwrap();
    assert!(propagate(&[repeated], &mut sum, &none).is_err());
}
