//! Checks gates given by a matrix against that matrix.
//!
//! A unitary gate is compared with its own matrix, and a transfer map with the Kraus
//! operators it was computed from, both as they act on observables and on states. The
//! transfer matrices are computed here by traces of dense matrices.

mod common;

use common::*;
use pauliprop::{propagate, Complex64, Gate, PauliOp, PauliSum, Picture, Truncation};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Every ordered choice of one, two or three of three qubits.
const SUPPORTS: [&[usize]; 15] = [
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
    &[0, 2, 1],
    &[1, 0, 2],
    &[1, 2, 0],
    &[2, 0, 1],
    &[2, 1, 0],
];

fn flat(matrix: &Matrix) -> Vec<Complex64> {
    matrix.iter().flatten().copied().collect()
}

/// A random unitary, from the Gram-Schmidt process on the columns of a random matrix.
fn random_unitary(rng: &mut StdRng, dim: usize) -> Matrix {
    let mut columns: Vec<Vec<Complex64>> = Vec::new();
    while columns.len() < dim {
        let mut v: Vec<Complex64> = (0..dim)
            .map(|_| c(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
            .collect();
        for u in &columns {
            let overlap: Complex64 = u.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
            for (x, a) in v.iter_mut().zip(u) {
                *x -= overlap * a;
            }
        }
        let norm = v.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        if norm > 0.1 {
            columns.push(v.iter().map(|x| x / norm).collect());
        }
    }
    let mut m = zeros(dim);
    for (j, column) in columns.iter().enumerate() {
        for (i, value) in column.iter().enumerate() {
            m[i][j] = *value;
        }
    }
    m
}

/// The transfer matrix of `O -> sum_k K_k† O K_k` on `k` qubits, row by row: the entry in row
/// `i` and column `j` is the factor of the Pauli string `i` in the image of the string `j`.
fn transfer_matrix(kraus: &[Matrix], k: usize) -> Vec<f64> {
    let strings = 1usize << (2 * k);
    let basis: Vec<Matrix> = (0..strings)
        .map(|index| string_matrix(&ops_of_index(index, k)))
        .collect();
    let mut matrix = vec![0.0; strings * strings];
    for j in 0..strings {
        let mut image = zeros(1 << k);
        for kraus_operator in kraus {
            let term = mul(&mul(&dagger(kraus_operator), &basis[j]), kraus_operator);
            for (row, term_row) in image.iter_mut().zip(&term) {
                for (value, term_value) in row.iter_mut().zip(term_row) {
                    *value += term_value;
                }
            }
        }
        for i in 0..strings {
            let factor = trace_of_product(&basis[i], &image) / (1usize << k) as f64;
            assert!(factor.im.abs() < TOL);
            matrix[i * strings + j] = factor.re;
        }
    }
    matrix
}

fn columns_of(gate: &Gate) -> &[Vec<(usize, f64)>] {
    match gate {
        Gate::TransferMap { columns, .. } => columns,
        other => panic!("expected a transfer map, got {other:?}"),
    }
}

/// The transfer map as a dense matrix, row by row.
fn dense(gate: &Gate) -> Vec<Vec<f64>> {
    let columns = columns_of(gate);
    let mut matrix = vec![vec![0.0; columns.len()]; columns.len()];
    for (j, column) in columns.iter().enumerate() {
        for &(i, factor) in column {
            matrix[i][j] = factor;
        }
    }
    matrix
}

#[test]
fn unitary_gates_match_their_matrices_on_observables_and_states() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(21);
    for qubits in SUPPORTS {
        let small = random_unitary(&mut rng, 1 << qubits.len());
        let step = Step {
            gate: Gate::unitary(qubits, &flat(&small)).unwrap(),
            kraus: vec![embed(&small, qubits, n)],
        };
        let what = format!("unitary on {qubits:?}");
        assert_matches_kraus_in(Picture::Heisenberg, std::slice::from_ref(&step), n, &what);
        assert_matches_kraus_in(Picture::Schrodinger, std::slice::from_ref(&step), n, &what);
    }
}

#[test]
fn transfer_matrices_of_rotations_have_the_known_form() {
    let theta: f64 = 0.731;
    let (sin, cos) = theta.sin_cos();

    // exp(-i theta / 2 Y) takes X to cos X + sin Z and Z to cos Z - sin X on an observable.
    let ry = gate_matrix(&Gate::ry(theta, 0), 1);
    let expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos, 0.0, -sin],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, sin, 0.0, cos],
    ];
    let gate = Gate::unitary(&[0], &flat(&ry)).unwrap();
    let (on_observables, on_states) = (dense(&gate), dense(&gate.adjoint()));
    for (i, row) in expected.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            assert!((on_observables[i][j] - value).abs() < TOL);
            assert!((on_states[j][i] - value).abs() < TOL);
        }
    }

    // The T gate, diag(1, exp(i pi / 4)).
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let t = vec![vec![c(1.0, 0.0), c(0.0, 0.0)], vec![c(0.0, 0.0), c(s, s)]];
    let expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.0],
        [0.0, -s, s, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let gate = Gate::unitary(&[0], &flat(&t)).unwrap();
    let on_observables = dense(&gate);
    for (i, row) in expected.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            assert!((on_observables[i][j] - value).abs() < TOL);
        }
    }
    // Entries that vanish are not stored.
    assert_eq!(columns_of(&gate).iter().map(Vec::len).sum::<usize>(), 6);
}

#[test]
fn unitary_of_a_gate_propagates_like_the_gate() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(22);
    for _ in 0..40 {
        let gate = random_gate(&mut rng, n);
        let all: Vec<usize> = (0..n).collect();
        let as_matrix = Gate::unitary(&all, &flat(&gate_matrix(&gate, n))).unwrap();

        for input in 0..1 << (2 * n) {
            let ops = sparse(&ops_of_index(input, n));
            let mut direct = PauliSum::<1>::new(n).unwrap();
            direct.add(&ops, 1.0).unwrap();
            let mut through_matrix = direct.clone();
            propagate(
                std::slice::from_ref(&gate),
                &mut direct,
                &Truncation::none(),
            )
            .unwrap();
            propagate(
                std::slice::from_ref(&as_matrix),
                &mut through_matrix,
                &Truncation::default(),
            )
            .unwrap();
            for (string, coeff) in through_matrix.iter() {
                assert!((coeff - direct.coeff(string)).abs() < TOL, "{gate:?}");
            }
            for (string, coeff) in direct.iter() {
                assert!(
                    (coeff - through_matrix.coeff(string)).abs() < TOL,
                    "{gate:?}"
                );
            }
        }
    }
}

#[test]
fn transfer_maps_match_the_kraus_operators_they_come_from() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(23);
    for qubits in &SUPPORTS[..9] {
        let k = qubits.len();
        let dim = 1 << k;

        // A mixture of two random unitaries followed by amplitude damping on the first qubit
        // of the support: neither unitary nor diagonal in the Pauli basis, and not unital.
        let p: f64 = rng.gen_range(0.2..0.8);
        let gamma: f64 = rng.gen_range(0.1..0.6);
        let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
        let k0 = vec![vec![l, o], vec![o, c((1.0 - gamma).sqrt(), 0.0)]];
        let k1 = vec![vec![o, c(gamma.sqrt(), 0.0)], vec![o, o]];
        let unitaries = [random_unitary(&mut rng, dim), random_unitary(&mut rng, dim)];
        let mut kraus = Vec::new();
        for damping in [&k0, &k1] {
            for (unitary, weight) in unitaries.iter().zip([p, 1.0 - p]) {
                let product = mul(&embed(damping, &[0], k), unitary);
                kraus.push(scaled(&product, weight.sqrt()));
            }
        }

        let step = Step {
            gate: Gate::transfer_map(qubits, &transfer_matrix(&kraus, k)).unwrap(),
            kraus: kraus.iter().map(|m| embed(m, qubits, n)).collect(),
        };
        let what = format!("transfer map on {qubits:?}");
        assert_matches_kraus_in(Picture::Heisenberg, std::slice::from_ref(&step), n, &what);
        assert_matches_kraus_in(Picture::Schrodinger, std::slice::from_ref(&step), n, &what);
    }
}

#[test]
fn transfer_map_of_amplitude_damping_propagates_like_the_channel() {
    let n = 2;
    let gamma = 0.3;
    let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
    let k0 = vec![vec![l, o], vec![o, c((1.0f64 - gamma).sqrt(), 0.0)]];
    let k1 = vec![vec![o, c(gamma.sqrt(), 0.0)], vec![o, o]];
    let as_map = Gate::transfer_map(&[1], &transfer_matrix(&[k0, k1], 1)).unwrap();
    let native = Gate::amplitude_damping(gamma, 1).unwrap();

    for input in 0..1 << (2 * n) {
        let ops = sparse(&ops_of_index(input, n));
        let mut a = PauliSum::<1>::new(n).unwrap();
        a.add(&ops, 1.0).unwrap();
        let mut b = a.clone();
        propagate(std::slice::from_ref(&native), &mut a, &Truncation::none()).unwrap();
        propagate(std::slice::from_ref(&as_map), &mut b, &Truncation::none()).unwrap();
        assert_eq!(a.len(), b.len());
        for (string, coeff) in a.iter() {
            assert!((coeff - b.coeff(string)).abs() < TOL);
        }
    }
}

#[test]
fn a_map_without_an_image_drops_the_term() {
    // Only the identity survives: the map of the channel that replaces the state of the
    // qubit by the maximally mixed state.
    let mut matrix = vec![0.0; 16];
    matrix[0] = 1.0;
    let gate = Gate::transfer_map(&[0], &matrix).unwrap();
    let mut sum = PauliSum::<1>::new(2).unwrap();
    sum.add(&[(0, PauliOp::X), (1, PauliOp::Z)], 0.5).unwrap();
    sum.add(&[(1, PauliOp::Y)], 0.25).unwrap();
    propagate(&[gate], &mut sum, &Truncation::none()).unwrap();
    assert_eq!(sum.len(), 1);
    assert_eq!(sum.sorted_terms()[0].1, 0.25);
}

#[test]
fn invalid_matrices_are_rejected() {
    let identity2 = flat(&identity(2));
    assert!(Gate::unitary(&[0], &identity2).is_ok());
    // The size does not match the number of qubits.
    assert!(Gate::unitary(&[0, 1], &identity2).is_err());
    assert!(Gate::unitary(&[], &[]).is_err());
    assert!(Gate::unitary(&[0, 1, 2, 3, 4, 5], &flat(&identity(64))).is_err());
    // Not unitary, and not a number.
    let mut stretched = identity2.clone();
    stretched[3] = c(1.001, 0.0);
    assert!(Gate::unitary(&[0], &stretched).is_err());
    let mut broken = identity2.clone();
    broken[1] = c(f64::NAN, 0.0);
    assert!(Gate::unitary(&[0], &broken).is_err());

    assert!(Gate::transfer_map(&[0], &[0.0; 16]).is_ok());
    assert!(Gate::transfer_map(&[0], &[0.0; 4]).is_err());
    assert!(Gate::transfer_map(&[0, 1], &[0.0; 16]).is_err());
    let mut infinite = vec![0.0; 16];
    infinite[5] = f64::INFINITY;
    assert!(Gate::transfer_map(&[0], &infinite).is_err());

    // The qubits are checked against the sum when the gate is used.
    let mut sum = PauliSum::<1>::new(2).unwrap();
    sum.add(&[(0, PauliOp::Z)], 1.0).unwrap();
    let outside = Gate::unitary(&[2], &identity2).unwrap();
    assert!(propagate(&[outside], &mut sum, &Truncation::none()).is_err());
    let twice = Gate::unitary(&[1, 1], &flat(&identity(4))).unwrap();
    assert!(propagate(&[twice], &mut sum, &Truncation::none()).is_err());
}
