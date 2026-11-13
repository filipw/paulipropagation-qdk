//! Checks sums, products and commutators of Pauli sums against their matrices, and the
//! merging of translated Pauli strings against translations written out on the operators.

mod common;

use common::*;
use num_complex::Complex64;
use pauliprop::{Coefficient, PauliOp, PauliString, PauliSum};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeMap;

/// The matrix of a sum, from its terms.
fn matrix_of<C: Coefficient>(sum: &PauliSum<1, C>) -> Matrix {
    let n = sum.num_qubits();
    let mut matrix = zeros(1 << n);
    for (string, coeff) in sum.sorted_terms() {
        let ops: Vec<PauliOp> = (0..n).map(|q| string.get(q)).collect();
        for (row, string_row) in matrix.iter_mut().zip(&string_matrix(&ops)) {
            for (value, string_value) in row.iter_mut().zip(string_row) {
                *value += string_value * coeff.to_complex();
            }
        }
    }
    matrix
}

fn random_sum<C: Coefficient>(
    rng: &mut StdRng,
    n: usize,
    terms: usize,
    coeff: impl Fn(&mut StdRng) -> C,
) -> PauliSum<1, C> {
    let mut sum = PauliSum::new(n).unwrap();
    for _ in 0..terms {
        let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
        sum.add(&sparse(&ops), coeff(rng)).unwrap();
    }
    sum
}

fn assert_close(a: &Matrix, b: &Matrix, what: &str) {
    for (row_a, row_b) in a.iter().zip(b) {
        for (x, y) in row_a.iter().zip(row_b) {
            assert!((x - y).norm() < 1e-11, "{what}: {x} and {y}");
        }
    }
}

fn combined(a: &Matrix, b: &Matrix, factor: Complex64) -> Matrix {
    let rows = a.iter().zip(b);
    rows.map(|(row_a, row_b)| {
        row_a
            .iter()
            .zip(row_b)
            .map(|(x, y)| x + y * factor)
            .collect()
    })
    .collect()
}

fn check_products<C: Coefficient>(seed: u64, coeff: impl Fn(&mut StdRng) -> C + Copy) {
    let mut rng = StdRng::seed_from_u64(seed);
    for round in 0..20 {
        let n = 1 + round % 4;
        let a = random_sum(&mut rng, n, 1 + round, coeff);
        let b = random_sum(&mut rng, n, 1 + 2 * round, coeff);
        let (ma, mb) = (matrix_of(&a), matrix_of(&b));
        let (ab, ba) = (mul(&ma, &mb), mul(&mb, &ma));

        assert_close(&matrix_of(&a.product(&b).unwrap()), &ab, "product");
        assert_close(&matrix_of(&b.product(&a).unwrap()), &ba, "product");
        let commutator = a.commutator(&b).unwrap();
        assert_close(
            &matrix_of(&commutator),
            &combined(&ab, &ba, c(-1.0, 0.0)),
            "commutator",
        );
        // Pairs that commute contribute no term, not a term of zero.
        assert!(a
            .commutator(&a)
            .unwrap()
            .sorted_terms()
            .iter()
            .all(|t| t.1.norm() < 1e-12));

        let factor = coeff(&mut rng);
        let mut total = a.clone();
        total.add_scaled(&b, factor).unwrap();
        assert_close(
            &matrix_of(&total),
            &combined(&ma, &mb, factor.to_complex()),
            "sum",
        );
        let mut scaled = a.clone();
        scaled.scale(factor);
        let expected = combined(&zeros(1 << n), &ma, factor.to_complex());
        assert_close(&matrix_of(&scaled), &expected, "scaled");
    }
}

#[test]
fn products_and_commutators_of_real_sums_match_their_matrices() {
    check_products(3, |rng| rng.gen_range(-1.0..1.0));
}

#[test]
fn products_and_commutators_of_complex_sums_match_their_matrices() {
    check_products(4, |rng| {
        Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    });
}

#[test]
fn products_of_pauli_operators() {
    let single = |op| {
        let mut sum = PauliSum::<1>::new(1).unwrap();
        sum.add(&[(0, op)], 1.0).unwrap();
        sum
    };
    let (x, y, z) = (single(PauliOp::X), single(PauliOp::Y), single(PauliOp::Z));
    let only = |sum: PauliSum<1, Complex64>| {
        let terms = sum.sorted_terms();
        assert_eq!(terms.len(), 1);
        (terms[0].0.get(0), terms[0].1)
    };
    assert_eq!(only(x.product(&y).unwrap()), (PauliOp::Z, c(0.0, 1.0)));
    assert_eq!(only(y.product(&x).unwrap()), (PauliOp::Z, c(0.0, -1.0)));
    assert_eq!(only(z.product(&z).unwrap()), (PauliOp::I, c(1.0, 0.0)));
    assert_eq!(only(x.commutator(&y).unwrap()), (PauliOp::Z, c(0.0, 2.0)));
    assert!(x.commutator(&x).unwrap().is_empty());
}

#[test]
fn sums_on_different_registers_are_rejected() {
    let a = PauliSum::<1>::new(2).unwrap();
    let mut b = PauliSum::<1>::new(3).unwrap();
    assert!(a.product(&b).is_err());
    assert!(a.commutator(&b).is_err());
    assert!(b.add_scaled(&a, 1.0).is_err());
    assert!(b.translation_merge_on_grid(2, 2).is_err());
}

/// The operators of a Pauli string from the highest qubit down, which compare like the
/// integers that PauliPropagation.jl holds Pauli strings in.
fn from_the_top<const W: usize>(string: &PauliString<W>, n: usize) -> Vec<PauliOp> {
    (0..n).rev().map(|q| string.get(q)).collect()
}

/// Merges by translations written out on the operators: of all images of a string under the
/// given maps of the qubits, the lowest is kept.
fn merged_by_hand<const W: usize>(
    sum: &PauliSum<W>,
    moves: &[Vec<usize>],
) -> BTreeMap<Vec<PauliOp>, f64> {
    let n = sum.num_qubits();
    let mut merged = BTreeMap::new();
    for (string, coeff) in sum.sorted_terms() {
        let images = moves.iter().map(|to| {
            let mut image = vec![PauliOp::I; n];
            for q in 0..n {
                image[n - 1 - to[q]] = string.get(q);
            }
            image
        });
        *merged.entry(images.min().unwrap()).or_insert(0.0) += coeff;
    }
    merged
}

fn assert_merged<const W: usize>(sum: &PauliSum<W>, expected: &BTreeMap<Vec<PauliOp>, f64>) {
    let n = sum.num_qubits();
    let terms = sum.sorted_terms();
    assert_eq!(terms.len(), expected.len());
    for (string, coeff) in terms {
        let wanted = expected[&from_the_top(&string, n)];
        assert!((coeff - wanted).abs() < 1e-12, "{}", string.label(n));
    }
}

fn random_sparse_sum<const W: usize>(rng: &mut StdRng, n: usize, terms: usize) -> PauliSum<W> {
    let mut sum = PauliSum::<W>::new(n).unwrap();
    for _ in 0..terms {
        let weight = rng.gen_range(0..4);
        let ops: Vec<(usize, PauliOp)> = (0..weight)
            .map(|_| (rng.gen_range(0..n), OPS[rng.gen_range(1..4)]))
            .collect();
        sum.add_string(PauliString::from_ops(&ops), rng.gen_range(-1.0..1.0));
    }
    sum
}

#[test]
fn translations_on_a_ring_are_merged() {
    let mut rng = StdRng::seed_from_u64(5);
    for n in [1, 2, 5, 6, 9] {
        let moves: Vec<Vec<usize>> = (0..n)
            .map(|s| (0..n).map(|q| (q + s) % n).collect())
            .collect();
        let sum = random_sparse_sum::<1>(&mut rng, n, 300);
        let mut merged = sum.clone();
        merged.translation_merge();
        assert_merged(&merged, &merged_by_hand(&sum, &moves));
        assert!(merged.len() < sum.len() || n == 1);
    }
}

#[test]
fn translations_on_a_ring_of_several_words_are_merged() {
    let mut rng = StdRng::seed_from_u64(6);
    let n = 150;
    let moves: Vec<Vec<usize>> = (0..n)
        .map(|s| (0..n).map(|q| (q + s) % n).collect())
        .collect();
    let sum = random_sparse_sum::<4>(&mut rng, n, 400);
    let mut merged = sum.clone();
    merged.translation_merge();
    assert_merged(&merged, &merged_by_hand(&sum, &moves));
    // Every string of one operator is a translation of that operator on the first qubit.
    for op in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
        let mut ring = PauliSum::<4>::new(n).unwrap();
        for q in 0..n {
            ring.add(&[(q, op)], 1.0).unwrap();
        }
        ring.translation_merge();
        assert_eq!(
            ring.sorted_terms(),
            vec![(PauliString::from_ops(&[(0, op)]), n as f64)]
        );
    }
}

#[test]
fn translations_on_a_torus_are_merged() {
    let mut rng = StdRng::seed_from_u64(7);
    for (nx, ny) in [(2, 3), (3, 3), (4, 2), (1, 5), (5, 1)] {
        let n = nx * ny;
        let mut moves = Vec::new();
        for dy in 0..ny {
            for dx in 0..nx {
                let to = |q: usize| ((q / nx + dy) % ny) * nx + (q % nx + dx) % nx;
                moves.push((0..n).map(to).collect());
            }
        }
        let sum = random_sparse_sum::<1>(&mut rng, n, 300);
        let mut merged = sum.clone();
        merged.translation_merge_on_grid(nx, ny).unwrap();
        assert_merged(&merged, &merged_by_hand(&sum, &moves));
    }
}

#[test]
fn a_custom_symmetry_is_merged() {
    // Reflection of a chain: a string and its mirror image are merged into the lower of the two.
    let mut rng = StdRng::seed_from_u64(8);
    let n = 7;
    let moves = vec![(0..n).collect::<Vec<_>>(), (0..n).rev().collect()];
    let sum = random_sparse_sum::<1>(&mut rng, n, 300);
    let mut merged = sum.clone();
    merged.merge_by(|string| {
        let mirrored: Vec<(usize, PauliOp)> = string
            .ops()
            .iter()
            .map(|&(q, op)| (n - 1 - q, op))
            .collect();
        let mirrored = PauliString::from_ops(&mirrored);
        if mirrored.cmp_from_highest_qubit(string).is_lt() {
            mirrored
        } else {
            *string
        }
    });
    assert_merged(&merged, &merged_by_hand(&sum, &moves));
}
