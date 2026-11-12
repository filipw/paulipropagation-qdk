//! Checks the overlaps of a Pauli sum with states against the matrix of the sum.

mod common;

use common::*;
use pauliprop::{PauliOp, PauliSum};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A random Pauli sum on `n` qubits and its matrix.
fn random_sum(rng: &mut StdRng, n: usize, terms: usize) -> (PauliSum<1>, Matrix) {
    let mut sum = PauliSum::<1>::new(n).unwrap();
    let mut matrix = zeros(1 << n);
    for _ in 0..terms {
        let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
        let coeff = rng.gen_range(-1.0..1.0);
        sum.add(&sparse(&ops), coeff).unwrap();
        for (row, string_row) in matrix.iter_mut().zip(&string_matrix(&ops)) {
            for (value, string_value) in row.iter_mut().zip(string_row) {
                *value += string_value * coeff;
            }
        }
    }
    (sum, matrix)
}

#[test]
fn overlaps_with_states_match_the_matrix() {
    let n = 4;
    let dim = 1usize << n;
    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..20 {
        // Enough terms that strings of every kind are present, and some repeat.
        let (sum, matrix) = random_sum(&mut rng, n, 200);

        assert!((sum.overlap_with_zero() - matrix[0][0].re).abs() < TOL);

        // <+|O|+> is the mean of all matrix elements.
        let plus: f64 = matrix.iter().flatten().map(|v| v.re).sum::<f64>() / dim as f64;
        assert!((sum.overlap_with_plus() - plus).abs() < TOL);

        let trace: f64 = (0..dim).map(|i| matrix[i][i].re).sum();
        assert!((sum.overlap_with_max_mixed() - trace / dim as f64).abs() < TOL);

        for (state, row) in matrix.iter().enumerate() {
            let ones: Vec<usize> = (0..n).filter(|&q| bit(state, q, n) == 1).collect();
            let overlap = sum.overlap_with_computational(&ones).unwrap();
            assert!((overlap - row[state].re).abs() < TOL, "state {state:04b}");
        }

        let (other, other_matrix) = random_sum(&mut rng, n, 200);
        let expected = trace_of_product(&matrix, &other_matrix).re / dim as f64;
        assert!((sum.scalar_product(&other) - expected).abs() < TOL);
        assert_eq!(sum.scalar_product(&other), other.scalar_product(&sum));
    }
}

#[test]
fn overlap_with_a_computational_state_rejects_qubits_out_of_range() {
    let sum = PauliSum::<1>::new(3).unwrap();
    assert!(sum.overlap_with_computational(&[3]).is_err());
    assert_eq!(sum.overlap_with_computational(&[0, 2]).unwrap(), 0.0);
}
