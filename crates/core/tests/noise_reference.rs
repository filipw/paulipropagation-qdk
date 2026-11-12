//! Checks noise channels against their Kraus operators.
//!
//! A channel acts on a state as `rho -> sum_k K_k rho K_k†`, so it acts on an observable as
//! `O -> sum_k K_k† O K_k`. The reference builds the Kraus operators as dense matrices from
//! the description of the channel and never from the table the engine derives.

mod common;

use common::*;
use pauliprop::{propagate, Gate, PauliOp, PauliSum, Truncation};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Pauli errors as `(string, probability)` pairs.
type Errors = Vec<(Vec<PauliOp>, f64)>;

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
        let steps = random_noisy_steps(&mut rng, n, 30);
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
