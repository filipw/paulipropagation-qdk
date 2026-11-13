//! Checks the propagation with a largest number of terms against a propagation that is
//! written out: one gate at a time on the hash-map propagation, and after each gate a sort
//! of all the terms by size.

mod common;

use common::*;
use pauliprop::{
    propagate_capped, propagate_with, reference, Coefficient, Complex64, Gate, PauliOp,
    PauliString, PauliSum, Picture, Truncation, WeightDamping,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn any_gate(rng: &mut StdRng, n: usize) -> Gate {
    let q = rng.gen_range(0..n);
    match rng.gen_range(0..16) {
        0..=9 => random_gate(rng, n),
        10 | 11 => Gate::amplitude_damping(rng.gen_range(0.0..0.6), q).unwrap(),
        12 => Gate::depolarizing(rng.gen_range(0.0..0.3), q).unwrap(),
        13 | 14 => {
            // A map that sends a Pauli string to itself and a few others.
            let other = (q + 1 + rng.gen_range(0..n - 1)) % n;
            let qubits = if rng.gen_bool(0.5) {
                vec![q]
            } else {
                vec![q, other]
            };
            let dim = 1 << (2 * qubits.len());
            let matrix: Vec<f64> = (0..dim * dim)
                .map(|entry| {
                    if entry % (dim + 1) == 0 {
                        rng.gen_range(0.5..1.0)
                    } else if rng.gen_bool(2.0 / dim as f64) {
                        rng.gen_range(-1.0..1.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            Gate::transfer_map(&qubits, &matrix).unwrap()
        }
        _ => {
            let other = (q + 1 + rng.gen_range(0..n - 1)) % n;
            let inner = [random_gate(rng, 2), random_gate(rng, 2)];
            let matrix: Vec<Complex64> = circuit_matrix(&inner, 2).into_iter().flatten().collect();
            Gate::unitary(&[q, other], &matrix).unwrap()
        }
    }
}

/// A sum of a few Pauli strings whose coefficients are equally large, so that the choice
/// among equally large terms is exercised.
fn start<const W: usize, C: Coefficient>(
    rng: &mut StdRng,
    n: usize,
    coeff: impl Fn(&mut StdRng) -> C,
) -> PauliSum<W, C> {
    let mut sum = PauliSum::new(n).unwrap();
    for _ in 0..rng.gen_range(1..=4) {
        let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
        sum.add(&sparse(&ops), coeff(rng)).unwrap();
    }
    sum
}

fn sign(rng: &mut StdRng) -> f64 {
    if rng.gen_bool(0.5) {
        1.0
    } else {
        -1.0
    }
}

fn phase(rng: &mut StdRng) -> Complex64 {
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-0.6, 0.8),
        Complex64::new(0.8, -0.6),
    ][rng.gen_range(0..4)]
}

/// The largest `count` of the terms, which are in the order of their Pauli strings, by a
/// sort of all of them that keeps equally large terms in that order.
fn largest<const W: usize, C: Coefficient>(
    mut terms: Vec<(PauliString<W>, C)>,
    count: usize,
) -> Vec<(PauliString<W>, C)> {
    terms.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    terms.truncate(count);
    terms.sort_by_key(|term| term.0);
    terms
}

/// What the written-out propagation gives, how often it had to drop terms, and the most
/// terms that a gate left.
fn written_out<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &PauliSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    max_terms: usize,
) -> (Vec<(PauliString<W>, C)>, usize, usize) {
    let order: Vec<&Gate> = match picture {
        Picture::Heisenberg => circuit.iter().rev().collect(),
        Picture::Schrodinger => circuit.iter().collect(),
    };
    let mut sum = sum.clone();
    let (mut dropped, mut most) = (0, 0);
    for gate in order {
        let gate = std::slice::from_ref(gate);
        reference::propagate_with(gate, &mut sum, truncation, picture, None).unwrap();
        most = most.max(sum.len());
        if sum.len() > max_terms {
            dropped += 1;
            let kept = largest(sum.sorted_terms(), max_terms);
            sum = PauliSum::new(sum.num_qubits()).unwrap();
            for (string, coeff) in kept {
                sum.add_string(string, coeff);
            }
        }
    }
    (sum.sorted_terms(), dropped, most)
}

fn truncations() -> Vec<Truncation> {
    let none = Truncation::none();
    vec![
        none,
        Truncation {
            max_weight: Some(3),
            ..none
        },
        Truncation {
            min_abs_coeff: 1e-3,
            ..none
        },
        Truncation {
            min_rel_coeff: Some(1e-2),
            weight_damping: Some(WeightDamping {
                gamma: 0.3,
                min_abs_coeff: 1e-3,
            }),
            ..none
        },
    ]
}

/// Compares the two on random circuits, and returns how often terms were dropped for
/// their number and how often a circuit went through without that.
fn sweep<const W: usize, C: Coefficient>(
    seed: u64,
    qubits: std::ops::Range<usize>,
    coeff: impl Fn(&mut StdRng) -> C + Copy,
) -> (usize, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut dropped, mut untouched) = (0, 0);
    for round in 0..40 {
        let n = rng.gen_range(qubits.clone());
        let circuit: Vec<Gate> = (0..50).map(|_| any_gate(&mut rng, n)).collect();
        let sum = start::<W, C>(&mut rng, n, coeff);
        let truncation = truncations()[round % 4];
        for picture in [Picture::Heisenberg, Picture::Schrodinger] {
            for max_terms in [1, 7, 60, 400, 5000] {
                let (expected, drops, _) =
                    written_out(&circuit, &sum, &truncation, picture, max_terms);
                let mut capped = sum.clone();
                propagate_capped(&circuit, &mut capped, &truncation, picture, None, max_terms)
                    .unwrap();
                assert_eq!(
                    capped.sorted_terms(),
                    expected,
                    "round {round}, {picture:?}, at most {max_terms} terms"
                );
                assert!(capped.len() <= max_terms);
                dropped += drops;
                untouched += usize::from(drops == 0);
            }
        }
    }
    (dropped, untouched)
}

#[test]
fn the_largest_terms_are_kept_after_every_gate() {
    let (dropped, untouched) = sweep::<1, f64>(1, 3..7, sign);
    assert!(dropped > 3000, "dropped {dropped} times");
    assert!(untouched > 20, "{untouched} circuits stayed small");
}

#[test]
fn the_largest_complex_terms_are_kept_after_every_gate() {
    let (dropped, _) = sweep::<1, Complex64>(2, 3..7, phase);
    assert!(dropped > 3000, "dropped {dropped} times");
}

#[test]
fn the_largest_terms_are_kept_on_a_register_of_several_words() {
    let (dropped, _) = sweep::<2, f64>(3, 66..70, sign);
    assert!(dropped > 3000, "dropped {dropped} times");
}

#[test]
fn a_number_of_terms_that_is_never_reached_changes_nothing() {
    let mut rng = StdRng::seed_from_u64(4);
    let mut divided = 0;
    for round in 0..30 {
        let n = rng.gen_range(4..8);
        let circuit: Vec<Gate> = (0..80).map(|_| any_gate(&mut rng, n)).collect();
        let sum = start::<1, f64>(&mut rng, n, sign);
        let truncation = truncations()[round % 4];
        for picture in [Picture::Heisenberg, Picture::Schrodinger] {
            let mut plain = sum.clone();
            propagate_with(&circuit, &mut plain, &truncation, picture, None).unwrap();
            // The most terms that a gate leaves, and numbers above it that make the gates
            // go in runs of different lengths.
            let (_, _, most) = written_out(&circuit, &sum, &truncation, picture, usize::MAX);
            if most < 2 {
                continue;
            }
            for max_terms in [most, 2 * most + 1, 37 * most, 1 << 40, usize::MAX] {
                let mut capped = sum.clone();
                propagate_capped(&circuit, &mut capped, &truncation, picture, None, max_terms)
                    .unwrap();
                assert_eq!(capped.sorted_terms(), plain.sorted_terms());
            }
            // One term fewer is reached.
            let mut capped = sum.clone();
            propagate_capped(&circuit, &mut capped, &truncation, picture, None, most - 1).unwrap();
            let (expected, drops, _) = written_out(&circuit, &sum, &truncation, picture, most - 1);
            assert_eq!(capped.sorted_terms(), expected);
            assert!(drops > 0);
            divided += usize::from(capped.sorted_terms() != plain.sorted_terms());
        }
    }
    assert!(
        divided > 20,
        "{divided} results differ from the propagation"
    );
}

#[test]
fn equally_large_terms_are_kept_in_the_order_of_their_strings() {
    let string = |ops: &[PauliOp]| PauliString::<1>::from_ops(&sparse(ops));
    let (i, x, y, z) = (PauliOp::I, PauliOp::X, PauliOp::Y, PauliOp::Z);
    let mut strings = [
        string(&[z, z, i]),
        string(&[x, i, i]),
        string(&[i, y, z]),
        string(&[i, i, z]),
        string(&[y, x, x]),
    ];
    strings.sort();
    let coeffs = [0.5, -0.5, 0.25, 0.5, -0.5];
    for count in 0..=6 {
        let mut sum = PauliSum::<1>::new(3).unwrap();
        for (string, coeff) in strings.iter().zip(coeffs) {
            sum.add_string(*string, coeff);
        }
        sum.keep_largest(count);
        // The term of 0.25 is the first to go, then the equally large ones from the last.
        let expected: Vec<(PauliString<1>, f64)> = match count {
            0 => vec![],
            1 => vec![(strings[0], 0.5)],
            2 => vec![(strings[0], 0.5), (strings[1], -0.5)],
            3 => vec![(strings[0], 0.5), (strings[1], -0.5), (strings[3], 0.5)],
            4 => (0..5)
                .filter(|&k| k != 2)
                .map(|k| (strings[k], coeffs[k]))
                .collect(),
            _ => (0..5).map(|k| (strings[k], coeffs[k])).collect(),
        };
        assert_eq!(sum.sorted_terms(), expected, "the largest {count}");
    }

    // A complex coefficient counts by its absolute value.
    let mut sum = PauliSum::<1, Complex64>::new(3).unwrap();
    sum.add_string(strings[0], Complex64::new(0.3, 0.0));
    sum.add_string(strings[1], Complex64::new(0.2, -0.3));
    sum.add_string(strings[2], Complex64::new(0.0, 0.35));
    sum.keep_largest(2);
    let kept: Vec<PauliString<1>> = sum.sorted_terms().iter().map(|term| term.0).collect();
    assert_eq!(kept, vec![strings[1], strings[2]]);
}

#[test]
fn the_largest_of_random_sums() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..200 {
        let n = rng.gen_range(1..6);
        let mut sum = PauliSum::<1>::new(n).unwrap();
        for _ in 0..rng.gen_range(0..300) {
            let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
            // A few sizes, so that many terms are equally large.
            let size = [1.0, 0.5, 0.5, 1e-3, 1e-200, 0.0][rng.gen_range(0..6)];
            let string = PauliString::from_ops(&sparse(&ops));
            if sum.coeff(&string) == 0.0 {
                sum.add_string(string, size * sign(&mut rng));
            }
        }
        let count = rng.gen_range(0..sum.len() + 3);
        let expected = largest(sum.sorted_terms(), count);
        sum.keep_largest(count);
        assert_eq!(sum.sorted_terms(), expected);
    }
}

#[test]
fn what_cannot_be_done_is_rejected() {
    let circuit = [Gate::rx(0.3, 0)];
    let mut sum = PauliSum::<1>::new(2).unwrap();
    sum.add(&[(0, PauliOp::Z)], 1.0).unwrap();
    let none = Truncation::none();
    let picture = Picture::Heisenberg;
    assert!(propagate_capped(&circuit, &mut sum, &none, picture, None, 0).is_err());
    for counted in [
        Truncation {
            max_freq: Some(3),
            ..none
        },
        Truncation {
            max_sins: Some(3),
            ..none
        },
    ] {
        assert!(propagate_capped(&circuit, &mut sum, &counted, picture, None, 5).is_err());
    }
    let beyond = [Gate::rx(0.3, 2)];
    assert!(propagate_capped(&beyond, &mut sum, &none, picture, None, 5).is_err());
    assert_eq!(sum.len(), 1);
    // A gate that cannot be applied leaves the sum as it was, also where gates that change
    // the sum come before it.
    let before = sum.sorted_terms();
    let last_to_act = [Gate::rx(0.3, 2), Gate::rzz(0.4, 0, 1), Gate::rx(0.2, 0)];
    assert!(propagate_capped(&last_to_act, &mut sum, &none, picture, None, 2).is_err());
    assert_eq!(sum.sorted_terms(), before);
    let last_to_act = [Gate::rx(0.2, 0), Gate::rzz(0.4, 0, 1), Gate::rx(0.3, 2)];
    let state = Picture::Schrodinger;
    assert!(propagate_capped(&last_to_act, &mut sum, &none, state, None, 2).is_err());
    assert_eq!(sum.sorted_terms(), before);
    let mut changed = sum.clone();
    propagate_capped(&last_to_act[..2], &mut changed, &none, state, None, 2).unwrap();
    assert_ne!(changed.sorted_terms(), before);

    // A custom rule is applied like the other truncations.
    let rule = |string: &PauliString<1>, _: f64| string.weight() > 1;
    let circuit = [Gate::rx(0.3, 0), Gate::rzz(0.4, 0, 1), Gate::rx(0.2, 1)];
    let mut capped = sum.clone();
    propagate_capped(&circuit, &mut capped, &none, picture, Some(&rule), 100).unwrap();
    let mut plain = sum.clone();
    propagate_with(&circuit, &mut plain, &none, picture, Some(&rule)).unwrap();
    assert_eq!(capped.sorted_terms(), plain.sorted_terms());
    assert!(!plain.is_empty());
}
