//! Checks the truncation rules against a propagation written out term by term.
//!
//! The reference keeps a Pauli string as a list of operators, multiplies operators by the
//! table of the Pauli group, and carries the two counters next to each coefficient. It
//! shares only the definition of the rules with the engine.

mod common;

use common::*;
use pauliprop::{
    propagate, propagate_with, Gate, PauliOp, PauliString, PauliSum, Picture, Truncation,
    WeightDamping,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
struct Term {
    coeff: f64,
    freq: u32,
    sins: u32,
}

type Reference = BTreeMap<Vec<PauliOp>, Term>;

/// `a b = i^k c` for single-qubit Pauli operators, as `(c, k)`.
fn product(a: PauliOp, b: PauliOp) -> (PauliOp, u8) {
    use PauliOp::{I, X, Y, Z};
    match (a, b) {
        (I, other) | (other, I) => (other, 0),
        (X, X) | (Y, Y) | (Z, Z) => (I, 0),
        (X, Y) => (Z, 1),
        (Y, Z) => (X, 1),
        (Z, X) => (Y, 1),
        (Y, X) => (Z, 3),
        (Z, Y) => (X, 3),
        (X, Z) => (Y, 3),
    }
}

fn weight(ops: &[PauliOp]) -> u32 {
    ops.iter().filter(|&&op| op != PauliOp::I).count() as u32
}

fn merge(sum: &mut Reference, ops: Vec<PauliOp>, term: Term) {
    sum.entry(ops)
        .and_modify(|old| {
            old.coeff += term.coeff;
            old.freq = old.freq.min(term.freq);
            old.sins = old.sins.min(term.sins);
        })
        .or_insert(term);
}

/// `O -> U† O U` for `U = exp(-i theta / 2 G)`: a string `P` that anticommutes with `G`
/// becomes `cos(theta) P + sin(theta) i G P`.
fn rotate(sum: &Reference, generator: &[PauliOp], theta: f64) -> Reference {
    let mut next = Reference::new();
    for (ops, term) in sum {
        let (mut image, mut phase) = (Vec::new(), 0u8);
        for (&g, &p) in generator.iter().zip(ops) {
            let (op, k) = product(g, p);
            image.push(op);
            phase = (phase + k) % 4;
        }
        if phase % 2 == 0 {
            merge(&mut next, ops.clone(), *term);
            continue;
        }
        // i G P = i^(phase + 1) P'
        let sign = if phase == 3 { 1.0 } else { -1.0 };
        let kept = Term {
            coeff: term.coeff * theta.cos(),
            freq: term.freq + 1,
            sins: term.sins,
        };
        let created = Term {
            coeff: term.coeff * theta.sin() * sign,
            freq: term.freq + 1,
            sins: term.sins + 1,
        };
        merge(&mut next, ops.clone(), kept);
        merge(&mut next, image, created);
    }
    next
}

fn truncate(sum: &mut Reference, truncation: &Truncation) {
    let largest = sum.values().map(|t| t.coeff.abs()).fold(0.0, f64::max);
    let min_abs = match truncation.min_rel_coeff {
        Some(min_rel) => (min_rel * largest).max(truncation.min_abs_coeff),
        None => truncation.min_abs_coeff,
    };
    sum.retain(|ops, term| {
        let w = weight(ops);
        let damped_away = truncation.weight_damping.is_some_and(|damping| {
            term.coeff.abs() * 10f64.powf(-damping.gamma * f64::from(w)) < damping.min_abs_coeff
        });
        !(term.coeff.abs() < min_abs
            || truncation.max_weight.is_some_and(|max| w > max)
            || truncation.max_freq.is_some_and(|max| term.freq > max)
            || truncation.max_sins.is_some_and(|max| term.sins > max)
            || damped_away)
    });
}

fn dense_generator(gate: &Gate, n: usize) -> (Vec<PauliOp>, f64) {
    match gate {
        Gate::Rotation { generator, theta } => {
            let mut ops = vec![PauliOp::I; n];
            for &(q, op) in generator {
                ops[q] = op;
            }
            (ops, *theta)
        }
        other => panic!("expected a rotation, got {other:?}"),
    }
}

/// Rotations by small and large angles on one to three qubits.
fn random_rotations(rng: &mut StdRng, n: usize, count: usize) -> Vec<Gate> {
    (0..count)
        .map(|_| {
            let k = rng.gen_range(1..=3);
            let mut qubits: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                qubits.swap(i, j);
            }
            let ops: Vec<(usize, PauliOp)> = qubits[..k]
                .iter()
                .map(|&q| (q, OPS[rng.gen_range(1..4)]))
                .collect();
            Gate::rotation(&ops, rng.gen_range(-1.5..1.5))
        })
        .collect()
}

/// Propagates `Z` on qubit 2 and a second term through the circuit both ways and compares.
fn assert_matches_reference(circuit: &[Gate], n: usize, truncation: &Truncation, what: &str) {
    let start = [
        (vec![(2, PauliOp::Z)], 1.0),
        (vec![(0, PauliOp::X), (3, PauliOp::Y)], -0.4),
    ];

    let mut sum = PauliSum::<1>::new(n).unwrap();
    let mut reference = Reference::new();
    for (ops, coeff) in &start {
        sum.add(ops, *coeff).unwrap();
        let mut dense = vec![PauliOp::I; n];
        for &(q, op) in ops {
            dense[q] = op;
        }
        let term = Term {
            coeff: *coeff,
            freq: 0,
            sins: 0,
        };
        reference.insert(dense, term);
    }

    propagate(circuit, &mut sum, truncation).unwrap();
    for gate in circuit.iter().rev() {
        let (generator, theta) = dense_generator(gate, n);
        reference = rotate(&reference, &generator, theta);
        truncate(&mut reference, truncation);
    }

    assert_eq!(sum.len(), reference.len(), "{what}: number of terms");
    for (ops, term) in &reference {
        let actual = sum.coeff(&PauliString::from_ops(&sparse(ops)));
        assert!(
            (actual - term.coeff).abs() < 1e-13,
            "{what}: {ops:?}: engine {actual}, reference {}",
            term.coeff
        );
    }
}

#[test]
fn frequency_and_sine_truncations_match_the_reference() {
    let n = 5;
    let mut rng = StdRng::seed_from_u64(51);
    let mut dropped_something = false;
    for trial in 0..6 {
        let circuit = random_rotations(&mut rng, n, 30);
        let mut exact = PauliSum::<1>::new(n).unwrap();
        exact.add(&[(2, PauliOp::Z)], 1.0).unwrap();
        exact
            .add(&[(0, PauliOp::X), (3, PauliOp::Y)], -0.4)
            .unwrap();
        propagate(&circuit, &mut exact, &Truncation::none()).unwrap();

        for (max_freq, max_sins) in [
            (Some(0), None),
            (Some(2), None),
            (Some(5), None),
            (Some(9), None),
            (None, Some(0)),
            (None, Some(1)),
            (None, Some(3)),
            (None, Some(6)),
            (Some(6), Some(2)),
            (Some(3), Some(3)),
        ] {
            let truncation = Truncation {
                max_freq,
                max_sins,
                ..Truncation::none()
            };
            let what = format!("trial {trial}, max_freq {max_freq:?}, max_sins {max_sins:?}");
            assert_matches_reference(&circuit, n, &truncation, &what);

            let mut sum = PauliSum::<1>::new(n).unwrap();
            sum.add(&[(2, PauliOp::Z)], 1.0).unwrap();
            sum.add(&[(0, PauliOp::X), (3, PauliOp::Y)], -0.4).unwrap();
            propagate(&circuit, &mut sum, &truncation).unwrap();
            dropped_something |= sum.len() < exact.len();
        }
    }
    assert!(dropped_something, "the truncations never dropped a term");
}

#[test]
fn relative_and_damped_truncations_match_the_reference() {
    let n = 5;
    let mut rng = StdRng::seed_from_u64(52);
    for trial in 0..6 {
        let circuit = random_rotations(&mut rng, n, 30);
        let rules = [
            Truncation {
                min_rel_coeff: Some(1e-3),
                ..Truncation::default()
            },
            Truncation {
                min_rel_coeff: Some(5e-2),
                min_abs_coeff: 1e-2,
                ..Truncation::none()
            },
            Truncation {
                min_rel_coeff: Some(0.3),
                max_weight: Some(3),
                ..Truncation::none()
            },
            Truncation {
                weight_damping: Some(WeightDamping {
                    gamma: 0.5,
                    min_abs_coeff: 1e-3,
                }),
                ..Truncation::none()
            },
            Truncation {
                weight_damping: Some(WeightDamping {
                    gamma: 0.1,
                    min_abs_coeff: 1e-2,
                }),
                min_rel_coeff: Some(1e-2),
                max_sins: Some(4),
                ..Truncation::default()
            },
        ];
        for (index, truncation) in rules.iter().enumerate() {
            let what = format!("trial {trial}, rule {index}");
            assert_matches_reference(&circuit, n, truncation, &what);
        }
    }
}

fn start_sum(n: usize) -> PauliSum<1> {
    let mut sum = PauliSum::<1>::new(n).unwrap();
    sum.add(&[(2, PauliOp::Z)], 1.0).unwrap();
    sum
}

#[test]
fn rules_that_drop_nothing_leave_the_result_as_it_is() {
    let n = 5;
    let mut rng = StdRng::seed_from_u64(53);
    let circuit: Vec<Gate> = (0..60).map(|_| random_gate(&mut rng, n)).collect();
    let mut plain = start_sum(n);
    propagate(&circuit, &mut plain, &Truncation::default()).unwrap();

    let loose = [
        Truncation {
            max_freq: Some(1000),
            max_sins: Some(1000),
            ..Truncation::default()
        },
        Truncation {
            min_rel_coeff: Some(0.0),
            ..Truncation::default()
        },
        Truncation {
            weight_damping: Some(WeightDamping {
                gamma: 0.0,
                min_abs_coeff: 1e-10,
            }),
            ..Truncation::default()
        },
        Truncation {
            max_weight: Some(n as u32),
            ..Truncation::default()
        },
    ];
    for truncation in loose {
        let mut sum = start_sum(n);
        propagate(&circuit, &mut sum, &truncation).unwrap();
        assert_eq!(sum.sorted_terms(), plain.sorted_terms(), "{truncation:?}");
    }
}

#[test]
fn a_custom_rule_drops_the_terms_it_selects() {
    let n = 5;
    let mut rng = StdRng::seed_from_u64(54);
    let circuit: Vec<Gate> = (0..60).map(|_| random_gate(&mut rng, n)).collect();

    // A rule on the Pauli string is the weight truncation written by hand.
    let by_weight = Truncation {
        max_weight: Some(3),
        ..Truncation::default()
    };
    let mut expected = start_sum(n);
    propagate(&circuit, &mut expected, &by_weight).unwrap();
    let heavy = |string: &PauliString<1>, _: f64| string.weight() > 3;
    let mut sum = start_sum(n);
    propagate_with(
        &circuit,
        &mut sum,
        &Truncation::default(),
        Picture::Heisenberg,
        Some(&heavy),
    )
    .unwrap();
    assert_eq!(sum.sorted_terms(), expected.sorted_terms());
    let mut exact = start_sum(n);
    propagate(&circuit, &mut exact, &Truncation::default()).unwrap();
    assert!(1 < sum.len() && sum.len() < exact.len());

    // A rule on the coefficient is the coefficient truncation, on rotations alone: after a
    // Clifford gate the coefficient rule is asked only because there is a custom rule.
    let rotations = random_rotations(&mut rng, n, 40);
    let by_coeff = Truncation {
        min_abs_coeff: 1e-2,
        ..Truncation::none()
    };
    let mut expected = start_sum(n);
    propagate(&rotations, &mut expected, &by_coeff).unwrap();
    let small = |_: &PauliString<1>, coeff: f64| coeff.abs() < 1e-2;
    let mut sum = start_sum(n);
    propagate_with(
        &rotations,
        &mut sum,
        &Truncation::none(),
        Picture::Heisenberg,
        Some(&small),
    )
    .unwrap();
    assert_eq!(sum.sorted_terms(), expected.sorted_terms());
}

#[test]
fn a_clifford_gate_drops_terms_only_by_rules_on_the_pauli_string() {
    let tiny = PauliString::<1>::from_ops(&[(1, PauliOp::X)]);
    let start = || {
        let mut sum = PauliSum::<1>::new(3).unwrap();
        sum.add(&[(0, PauliOp::Z)], 1.0).unwrap();
        sum.add(&[(1, PauliOp::Z)], 1e-12).unwrap();
        sum
    };

    // Below the default threshold, but a Clifford gate does not ask about coefficients.
    let mut sum = start();
    propagate(&[Gate::H(1)], &mut sum, &Truncation::default()).unwrap();
    assert_eq!(sum.len(), 2);
    assert_eq!(sum.coeff(&tiny), 1e-12);

    // With a rule on the Pauli string, every rule is asked.
    let with_weight = Truncation {
        max_weight: Some(3),
        ..Truncation::default()
    };
    let mut sum = start();
    propagate(&[Gate::H(1)], &mut sum, &with_weight).unwrap();
    assert_eq!(sum.len(), 1);

    // Any other gate asks every rule about every term, also those it leaves alone.
    let mut sum = start();
    propagate(&[Gate::rz(0.3, 2)], &mut sum, &Truncation::default()).unwrap();
    assert_eq!(sum.len(), 1);
    let mut sum = start();
    let noise = Gate::depolarizing(0.1, 2).unwrap();
    propagate(&[noise], &mut sum, &Truncation::default()).unwrap();
    assert_eq!(sum.len(), 1);
}

#[test]
fn no_sine_branches_leave_the_cosine_path() {
    // With max_sins = 0 the only term left is the start, with one cosine per rotation
    // that does not commute with it.
    let n = 5;
    let mut rng = StdRng::seed_from_u64(55);
    let circuit = random_rotations(&mut rng, n, 25);
    let start = PauliString::<1>::from_ops(&[(2, PauliOp::Z)]);
    let mut expected = 1.0;
    for gate in circuit.iter().rev() {
        if let Gate::Rotation { generator, theta } = gate {
            if !start.commutes(&PauliString::from_ops(generator)) {
                expected *= theta.cos();
            }
        }
    }
    let truncation = Truncation {
        max_sins: Some(0),
        ..Truncation::none()
    };
    let mut sum = start_sum(n);
    propagate(&circuit, &mut sum, &truncation).unwrap();
    assert_eq!(sum.len(), 1);
    assert!((sum.coeff(&start) - expected).abs() < 1e-14);
    assert!(expected.abs() < 0.999);
}
