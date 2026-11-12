//! Checks the propagation of states against dense matrices.
//!
//! A state `rho` goes through a circuit as `rho -> sum_k K_k rho K_k†`, gate by gate in the
//! order of the circuit. The reference applies the Kraus operators as dense matrices.

mod common;

use common::*;
use pauliprop::{
    propagate, propagate_with, Gate, PauliOp, PauliString, PauliSum, Picture, Truncation,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn forward(circuit: &[Gate], sum: &mut PauliSum<1>) {
    propagate_with(
        circuit,
        sum,
        &Truncation::none(),
        Picture::Schrodinger,
        None,
    )
    .unwrap();
}

fn random_sum(rng: &mut StdRng, n: usize, terms: usize) -> PauliSum<1> {
    let mut sum = PauliSum::<1>::new(n).unwrap();
    for _ in 0..terms {
        let ops = ops_of_index(rng.gen_range(0..1 << (2 * n)), n);
        sum.add(&sparse(&ops), rng.gen_range(-1.0..1.0)).unwrap();
    }
    sum
}

#[test]
fn every_gate_acts_on_a_state_as_its_matrix() {
    let n = 3;
    let mut gates = vec![
        Gate::H(1),
        Gate::X(0),
        Gate::Y(2),
        Gate::Z(1),
        Gate::S(0),
        Gate::Sdg(2),
        Gate::Sx(1),
        Gate::Sxdg(0),
        Gate::Cx(0, 2),
        Gate::Cx(2, 0),
        Gate::Cy(1, 2),
        Gate::Cy(2, 1),
        Gate::Cz(0, 1),
        Gate::Swap(0, 2),
        Gate::t(1),
        Gate::tdg(2),
        Gate::rx(0.37, 0),
        Gate::ry(-1.2, 1),
        Gate::rz(2.4, 2),
        Gate::rzz(0.9, 2, 0),
    ];
    gates.extend(Gate::ccx(2, 0, 1));
    for gate in gates {
        let what = format!("{gate:?}");
        let step = unitary_step(gate, n);
        assert_matches_kraus_in(Picture::Schrodinger, &[step], n, &what);
    }
}

#[test]
fn noise_channels_act_on_a_state_as_their_kraus_operators() {
    let n = 2;
    let mut rng = StdRng::seed_from_u64(41);
    for q in 0..n {
        for gamma in [0.0, 0.2, 0.75, 1.0] {
            let step = amplitude_damping_step(gamma, q, n);
            let what = format!("amplitude damping {gamma} on {q}");
            assert_matches_kraus_in(Picture::Schrodinger, &[step], n, &what);
        }
        let step = random_pauli_channel_step(&mut rng, &[q], n);
        assert_matches_kraus_in(Picture::Schrodinger, &[step], n, "Pauli channel");
    }
    let step = random_pauli_channel_step(&mut rng, &[1, 0], n);
    assert_matches_kraus_in(Picture::Schrodinger, &[step], n, "two-qubit Pauli channel");
}

#[test]
fn noisy_random_circuits_act_on_a_state_as_their_kraus_operators() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(42);
    for trial in 0..10 {
        let steps = random_noisy_steps(&mut rng, n, 30);
        let what = format!("noisy random circuit {trial}");
        assert_matches_kraus_in(Picture::Schrodinger, &steps, n, &what);
    }
}

#[test]
fn amplitude_damping_takes_the_one_state_towards_the_zero_state() {
    // |1><1| = (I - Z) / 2 becomes (1 - gamma) |1><1| + gamma |0><0| = (I - (1 - 2 gamma) Z) / 2.
    let gamma = 0.3;
    let mut state = PauliSum::<1>::new(1).unwrap();
    state.add(&[], 0.5).unwrap();
    state.add(&[(0, PauliOp::Z)], -0.5).unwrap();
    forward(&[Gate::amplitude_damping(gamma, 0).unwrap()], &mut state);
    assert_eq!(state.len(), 2);
    assert!((state.coeff(&PauliString::identity()) - 0.5).abs() < TOL);
    let z = PauliString::from_ops(&[(0, PauliOp::Z)]);
    assert!((state.coeff(&z) + 0.5 * (1.0 - 2.0 * gamma)).abs() < TOL);
}

#[test]
fn an_expectation_value_is_the_same_in_both_pictures() {
    // Tr[O E(rho)] = Tr[E†(O) rho], where a sum propagated by a part of the circuit in one
    // picture meets the other sum propagated by the rest in the other picture.
    let n = 3;
    let mut rng = StdRng::seed_from_u64(43);
    for _ in 0..20 {
        let circuit: Vec<Gate> = random_noisy_steps(&mut rng, n, 9)
            .into_iter()
            .map(|step| step.gate)
            .collect();
        let state = random_sum(&mut rng, n, 40);
        let observable = random_sum(&mut rng, n, 40);

        let mut values = Vec::new();
        for split in [0, 4, 9] {
            let mut evolved_state = state.clone();
            forward(&circuit[..split], &mut evolved_state);
            let mut evolved_observable = observable.clone();
            propagate(
                &circuit[split..],
                &mut evolved_observable,
                &Truncation::none(),
            )
            .unwrap();
            values.push(evolved_state.scalar_product(&evolved_observable));
        }
        assert!(
            values[0].abs() > 1e-3,
            "the check needs a value that is not zero"
        );
        assert!((values[0] - values[1]).abs() < 1e-12);
        assert!((values[0] - values[2]).abs() < 1e-12);
    }
}

#[test]
fn propagating_back_undoes_a_unitary_circuit() {
    let n = 4;
    let mut rng = StdRng::seed_from_u64(44);
    for _ in 0..20 {
        let circuit: Vec<Gate> = (0..40).map(|_| random_gate(&mut rng, n)).collect();
        let start = random_sum(&mut rng, n, 5);

        for first in [Picture::Schrodinger, Picture::Heisenberg] {
            let second = match first {
                Picture::Schrodinger => Picture::Heisenberg,
                Picture::Heisenberg => Picture::Schrodinger,
            };
            let mut sum = start.clone();
            propagate_with(&circuit, &mut sum, &Truncation::none(), first, None).unwrap();
            propagate_with(&circuit, &mut sum, &Truncation::none(), second, None).unwrap();
            for (string, coeff) in sum.iter() {
                assert!((coeff - start.coeff(string)).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn the_adjoint_of_the_adjoint_is_the_gate() {
    let mut rng = StdRng::seed_from_u64(45);
    for _ in 0..100 {
        let gate = random_gate(&mut rng, 3);
        assert_eq!(gate.adjoint().adjoint(), gate);
    }
    let channel = Gate::depolarizing(0.2, 1).unwrap();
    assert_eq!(channel.adjoint(), channel);
    assert_eq!(Gate::rz(0.4, 2).adjoint(), Gate::rz(-0.4, 2));
    assert_eq!(Gate::S(0).adjoint(), Gate::Sdg(0));
}
