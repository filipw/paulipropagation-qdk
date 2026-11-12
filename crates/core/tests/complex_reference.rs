//! Checks sums with complex coefficients against dense matrices.

mod common;

use common::*;
use pauliprop::{propagate, Complex64, Gate, PauliOp, PauliString, PauliSum, Truncation};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A random sum with complex coefficients, with its real part, its imaginary part and its matrix.
fn random_sums(
    rng: &mut StdRng,
    n: usize,
    terms: usize,
) -> (PauliSum<1, Complex64>, PauliSum<1>, PauliSum<1>, Matrix) {
    let mut sum = PauliSum::<1, Complex64>::new(n).unwrap();
    let mut real = PauliSum::<1>::new(n).unwrap();
    let mut imaginary = PauliSum::<1>::new(n).unwrap();
    let mut matrix = zeros(1 << n);
    for _ in 0..terms {
        let ops = ops_of_index(rng.gen_range(0..1 << (2 * n)), n);
        let coeff = c(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        sum.add(&sparse(&ops), coeff).unwrap();
        real.add(&sparse(&ops), coeff.re).unwrap();
        imaginary.add(&sparse(&ops), coeff.im).unwrap();
        for (row, string_row) in matrix.iter_mut().zip(&string_matrix(&ops)) {
            for (value, string_value) in row.iter_mut().zip(string_row) {
                *value += coeff * string_value;
            }
        }
    }
    (sum, real, imaginary, matrix)
}

#[test]
fn complex_sums_match_the_matrix_of_the_circuit() {
    let n = 3;
    let dim = (1usize << n) as f64;
    let mut rng = StdRng::seed_from_u64(61);
    let basis: Vec<Matrix> = (0..1 << (2 * n))
        .map(|index| string_matrix(&ops_of_index(index, n)))
        .collect();
    for _ in 0..10 {
        let circuit: Vec<Gate> = (0..30).map(|_| random_gate(&mut rng, n)).collect();
        let unitary = circuit_matrix(&circuit, n);
        let (mut sum, _, _, matrix) = random_sums(&mut rng, n, 6);
        propagate(&circuit, &mut sum, &Truncation::none()).unwrap();

        let evolved = mul(&mul(&dagger(&unitary), &matrix), &unitary);
        for (index, string_matrix) in basis.iter().enumerate() {
            let expected = trace_of_product(string_matrix, &evolved) / dim;
            let string = PauliString::<1>::from_ops(&sparse(&ops_of_index(index, n)));
            assert!((sum.coeff(&string) - expected).norm() < TOL);
        }
    }
}

#[test]
fn a_complex_sum_propagates_as_its_real_and_imaginary_parts() {
    let n = 4;
    let mut rng = StdRng::seed_from_u64(62);
    for _ in 0..10 {
        let mut circuit: Vec<Gate> = (0..40).map(|_| random_gate(&mut rng, n)).collect();
        circuit.push(Gate::depolarizing(0.1, 1).unwrap());
        circuit.push(Gate::amplitude_damping(0.2, 3).unwrap());
        circuit.extend((0..10).map(|_| random_gate(&mut rng, n)));

        let (mut sum, mut real, mut imaginary, _) = random_sums(&mut rng, n, 5);
        propagate(&circuit, &mut sum, &Truncation::none()).unwrap();
        propagate(&circuit, &mut real, &Truncation::none()).unwrap();
        propagate(&circuit, &mut imaginary, &Truncation::none()).unwrap();

        // The gates multiply by real numbers and add, so the parts agree bit for bit.
        assert!(sum.len() > 20);
        for (string, coeff) in sum.iter() {
            assert_eq!(coeff.re, real.coeff(string));
            assert_eq!(coeff.im, imaginary.coeff(string));
        }
    }
}

#[test]
fn overlaps_of_complex_sums_match_their_matrices() {
    let n = 3;
    let dim = 1usize << n;
    let mut rng = StdRng::seed_from_u64(63);
    for _ in 0..20 {
        let (sum, _, _, matrix) = random_sums(&mut rng, n, 12);

        assert!((sum.overlap_with_zero() - matrix[0][0]).norm() < TOL);
        let plus: Complex64 = matrix.iter().flatten().sum::<Complex64>() / dim as f64;
        assert!((sum.overlap_with_plus() - plus).norm() < TOL);
        // Qubit 0 is the most significant bit of a basis state.
        let ones = [0, 2];
        let state = 0b101;
        let computational = sum.overlap_with_computational(&ones).unwrap();
        assert!((computational - matrix[state][state]).norm() < TOL);
        let trace: Complex64 = (0..dim).map(|i| matrix[i][i]).sum();
        assert!((sum.overlap_with_max_mixed() - trace / dim as f64).norm() < TOL);

        let (other, _, _, other_matrix) = random_sums(&mut rng, n, 12);
        let expected = trace_of_product(&matrix, &other_matrix) / dim as f64;
        assert!((sum.scalar_product(&other) - expected).norm() < TOL);

        let norm: f64 = sum.iter().map(|(_, coeff)| coeff.norm_sqr()).sum();
        assert!((sum.norm_sqr() - norm).abs() < TOL);
        let largest = sum
            .iter()
            .map(|(_, coeff)| coeff.norm())
            .fold(0.0, f64::max);
        assert_eq!(sum.max_abs_coeff(), largest);
    }
}

#[test]
fn complex_coefficients_are_truncated_by_their_absolute_value() {
    let truncation = Truncation {
        min_abs_coeff: 0.09,
        ..Truncation::none()
    };

    // 0.06 and 0.08 are both below the threshold, their absolute value 0.1 is not. The
    // rotation commutes with both terms and so only truncates.
    let mut sum = PauliSum::<1, Complex64>::new(3).unwrap();
    sum.add(&[(0, PauliOp::Z)], c(0.06, 0.08)).unwrap();
    sum.add(&[(1, PauliOp::Z)], c(0.05, 0.05)).unwrap();
    propagate(&[Gate::rz(0.3, 2)], &mut sum, &truncation).unwrap();
    assert_eq!(sum.len(), 1);
    assert_eq!(
        sum.coeff(&PauliString::from_ops(&[(0, PauliOp::Z)])),
        c(0.06, 0.08)
    );

    // The relative threshold is taken from the largest absolute value.
    let mut sum = PauliSum::<1, Complex64>::new(3).unwrap();
    sum.add(&[(0, PauliOp::Z)], c(3.0, 4.0)).unwrap();
    sum.add(&[(1, PauliOp::Z)], c(0.0, 0.6)).unwrap();
    sum.add(&[(1, PauliOp::X)], c(0.4, 0.0)).unwrap();
    let relative = Truncation {
        min_rel_coeff: Some(0.1),
        ..Truncation::none()
    };
    propagate(&[Gate::rz(0.3, 2)], &mut sum, &relative).unwrap();
    assert_eq!(sum.len(), 2);
}
