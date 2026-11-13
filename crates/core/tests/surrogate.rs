//! Checks that a surrogate, evaluated for any angles, is the propagation with those angles.

mod common;

use common::*;
use pauliprop::{
    propagate_with, Gate, PathLimits, PauliOp, PauliString, PauliSum, Picture, Surrogate,
    Truncation,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_sum(rng: &mut StdRng, n: usize, terms: usize) -> PauliSum<1> {
    let mut sum = PauliSum::<1>::new(n).unwrap();
    for _ in 0..terms {
        let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
        sum.add(&sparse(&ops), rng.gen_range(-1.0..1.0)).unwrap();
    }
    sum
}

/// The circuit with new angles for its rotations, and those angles.
fn with_new_angles(rng: &mut StdRng, circuit: &[Gate]) -> (Vec<Gate>, Vec<f64>) {
    let mut thetas = Vec::new();
    let circuit = circuit
        .iter()
        .map(|gate| match gate {
            Gate::Rotation { generator, .. } => {
                // Now and then an angle at which a branch vanishes.
                let theta = match rng.gen_range(0..8) {
                    0 => 0.0,
                    1 => std::f64::consts::FRAC_PI_2,
                    _ => rng.gen_range(-3.0..3.0),
                };
                thetas.push(theta);
                Gate::rotation(generator, theta)
            }
            other => other.clone(),
        })
        .collect();
    (circuit, thetas)
}

fn limits_of(round: usize, n: usize) -> PathLimits {
    PathLimits {
        max_weight: (round % 2 == 1).then_some(1 + (round as u32 / 2) % n as u32),
        max_freq: (round % 3 == 1).then_some(2 + round as u32 % 5),
        max_sins: (round % 5 == 1).then_some(1 + round as u32 % 3),
    }
}

#[test]
fn a_surrogate_is_the_propagation_for_any_angles() {
    let mut rng = StdRng::seed_from_u64(61);
    let (mut dropped, mut largest) = (0, 0);
    for round in 0..120 {
        let n = 2 + round % 5;
        let circuit: Vec<Gate> = (0..rng.gen_range(1..40))
            .map(|_| random_gate(&mut rng, n))
            .collect();
        let observable = random_sum(&mut rng, n, 1 + round % 4);
        let limits = limits_of(round, n);
        let truncation = Truncation {
            max_weight: limits.max_weight,
            max_freq: limits.max_freq,
            max_sins: limits.max_sins,
            ..Truncation::none()
        };
        let surrogate = Surrogate::new(&circuit, &observable, &limits, None).unwrap();
        let rotations = circuit
            .iter()
            .filter(|g| matches!(g, Gate::Rotation { .. }))
            .count();
        assert_eq!(surrogate.num_parameters(), rotations);
        assert_eq!(surrogate.num_qubits(), n);

        for _ in 0..3 {
            let (circuit, thetas) = with_new_angles(&mut rng, &circuit);
            let mut expected = observable.clone();
            propagate_with(
                &circuit,
                &mut expected,
                &truncation,
                Picture::Heisenberg,
                None,
            )
            .unwrap();
            let evaluated = surrogate.evaluate(&thetas).unwrap();
            // The same strings, and every coefficient made of the same products and sums.
            assert_eq!(
                evaluated.sorted_terms(),
                expected.sorted_terms(),
                "round {round}"
            );
            assert_eq!(surrogate.len(), expected.len());

            let mut exact = observable.clone();
            propagate_with(
                &circuit,
                &mut exact,
                &Truncation::none(),
                Picture::Heisenberg,
                None,
            )
            .unwrap();
            dropped += usize::from(exact.len() > expected.len());
            largest = largest.max(expected.len());
        }
    }
    assert!(
        dropped > 100 && largest > 500,
        "{dropped} truncated, largest sum {largest}"
    );
}

#[test]
fn a_custom_rule_drops_terms_of_a_surrogate() {
    let mut rng = StdRng::seed_from_u64(62);
    let n = 5;
    let rule = |string: &PauliString<1>| string.x_bit(0) && string.weight() > 2;
    let mut differed = 0;
    for _ in 0..30 {
        let circuit: Vec<Gate> = (0..30).map(|_| random_gate(&mut rng, n)).collect();
        let observable = random_sum(&mut rng, n, 2);
        let surrogate =
            Surrogate::new(&circuit, &observable, &PathLimits::default(), Some(&rule)).unwrap();
        let unruled = Surrogate::new(&circuit, &observable, &PathLimits::default(), None).unwrap();
        differed += usize::from(unruled.len() != surrogate.len());
        assert!(surrogate.num_nodes() <= unruled.num_nodes());

        let (circuit, thetas) = with_new_angles(&mut rng, &circuit);
        let mut expected = observable.clone();
        let custom = |string: &PauliString<1>, _: f64| rule(string);
        propagate_with(
            &circuit,
            &mut expected,
            &Truncation::none(),
            Picture::Heisenberg,
            Some(&custom),
        )
        .unwrap();
        assert_eq!(
            surrogate.evaluate(&thetas).unwrap().sorted_terms(),
            expected.sorted_terms()
        );
    }
    assert!(differed > 10);
}

#[test]
fn a_surrogate_rejects_what_it_cannot_hold() {
    let mut observable = PauliSum::<1>::new(2).unwrap();
    observable.add(&[(0, PauliOp::Z)], 1.0).unwrap();
    let limits = PathLimits::default();
    let noisy = [Gate::rx(0.3, 0), Gate::depolarizing(0.1, 0).unwrap()];
    assert!(Surrogate::new(&noisy, &observable, &limits, None).is_err());
    let damped = [Gate::amplitude_damping(0.1, 0).unwrap()];
    assert!(Surrogate::new(&damped, &observable, &limits, None).is_err());
    let out_of_range = [Gate::rx(0.3, 2)];
    assert!(Surrogate::new(&out_of_range, &observable, &limits, None).is_err());

    let circuit = [Gate::rx(0.3, 0), Gate::H(1), Gate::rzz(0.2, 0, 1)];
    let surrogate = Surrogate::new(&circuit, &observable, &limits, None).unwrap();
    assert!(surrogate.evaluate(&[0.1]).is_err());
    assert!(surrogate.evaluate(&[0.1, f64::NAN]).is_err());
    assert!(surrogate.evaluate(&[0.1, 0.2]).is_ok());
    assert!(!surrogate.is_empty());
}
