//! Checks the propagation rules of unitary gates against dense matrices.
//!
//! The reference shares nothing with the engine: `U† P U` is computed by matrix
//! multiplication and expanded in the Pauli basis by traces.

mod common;

use common::*;
use pauliprop::{propagate, Gate, PauliOp, PauliString, PauliSum, Truncation};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn product_of_pauli_strings_matches_matrices() {
    let n = 2;
    for a in 0..16 {
        for b in 0..16 {
            let (ops_a, ops_b) = (ops_of_index(a, n), ops_of_index(b, n));
            let pa = PauliString::<1>::from_ops(&sparse(&ops_a));
            let pb = PauliString::<1>::from_ops(&sparse(&ops_b));
            let (product, k) = pa.mul(&pb);
            let phase = [c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0), c(0.0, -1.0)][k as usize];

            let expected = mul(&string_matrix(&ops_a), &string_matrix(&ops_b));
            let product_ops: Vec<PauliOp> = (0..n).map(|q| product.get(q)).collect();
            let actual = string_matrix(&product_ops);
            for i in 0..4 {
                for j in 0..4 {
                    assert!((phase * actual[i][j] - expected[i][j]).norm() < TOL);
                }
            }
            // Two strings commute exactly when their matrices do.
            let reversed = mul(&string_matrix(&ops_b), &string_matrix(&ops_a));
            let commute =
                (0..4).all(|i| (0..4).all(|j| (expected[i][j] - reversed[i][j]).norm() < TOL));
            assert_eq!(pa.commutes(&pb), commute);
        }
    }
}

#[test]
fn every_clifford_gate_matches_its_matrix_on_every_pauli_string() {
    let n = 3;
    let one_qubit: [fn(usize) -> Gate; 8] = [
        Gate::H,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::S,
        Gate::Sdg,
        Gate::Sx,
        Gate::Sxdg,
    ];
    for make in one_qubit {
        for q in 0..n {
            let gate = make(q);
            assert_matches_matrices(std::slice::from_ref(&gate), n, &format!("{gate:?}"));
        }
    }
    let two_qubit: [fn(usize, usize) -> Gate; 4] = [Gate::Cx, Gate::Cy, Gate::Cz, Gate::Swap];
    for make in two_qubit {
        for a in 0..n {
            for b in 0..n {
                if a != b {
                    let gate = make(a, b);
                    assert_matches_matrices(std::slice::from_ref(&gate), n, &format!("{gate:?}"));
                }
            }
        }
    }
}

#[test]
fn every_pauli_rotation_matches_its_matrix_on_every_pauli_string() {
    let n = 3;
    let mut rng = StdRng::seed_from_u64(7);
    // Every one of the 63 generators on three qubits, at a random angle and at special ones.
    for generator in 1..1 << (2 * n) {
        let ops = sparse(&ops_of_index(generator, n));
        for theta in [
            rng.gen_range(-7.0..7.0),
            0.0,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::PI,
        ] {
            let gate = Gate::rotation(&ops, theta);
            assert_matches_matrices(std::slice::from_ref(&gate), n, &format!("{gate:?}"));
        }
    }
}

#[test]
fn named_rotations_and_t_gates_match_their_matrices() {
    let n = 2;
    let theta = 0.73;
    let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
    let phase = |angle: f64| c(angle.cos(), angle.sin());

    for q in 0..n {
        let t = embed(
            &vec![vec![l, o], vec![o, phase(std::f64::consts::FRAC_PI_4)]],
            &[q],
            n,
        );
        assert_matches_unitary(&[Gate::t(q)], &t, n, "T");
        assert_matches_unitary(&[Gate::tdg(q)], &dagger(&t), n, "T†");

        // RZ(theta) = diag(e^{-i theta / 2}, e^{i theta / 2})
        let rz = embed(
            &vec![vec![phase(-theta / 2.0), o], vec![o, phase(theta / 2.0)]],
            &[q],
            n,
        );
        assert_matches_unitary(&[Gate::rz(theta, q)], &rz, n, "RZ");

        let (sin, cos) = (theta / 2.0).sin_cos();
        let rx = embed(
            &vec![
                vec![c(cos, 0.0), c(0.0, -sin)],
                vec![c(0.0, -sin), c(cos, 0.0)],
            ],
            &[q],
            n,
        );
        assert_matches_unitary(&[Gate::rx(theta, q)], &rx, n, "RX");
        let ry = embed(
            &vec![
                vec![c(cos, 0.0), c(-sin, 0.0)],
                vec![c(sin, 0.0), c(cos, 0.0)],
            ],
            &[q],
            n,
        );
        assert_matches_unitary(&[Gate::ry(theta, q)], &ry, n, "RY");
    }

    // RZZ(theta) = diag(e^{-i theta / 2}, e^{i theta / 2}, e^{i theta / 2}, e^{-i theta / 2})
    let mut rzz = zeros(4);
    for (i, sign) in [-1.0, 1.0, 1.0, -1.0].into_iter().enumerate() {
        rzz[i][i] = phase(sign * theta / 2.0);
    }
    assert_matches_unitary(&[Gate::rzz(theta, 0, 1)], &rzz, n, "RZZ");
    assert_matches_unitary(&[Gate::rzz(theta, 1, 0)], &rzz, n, "RZZ reversed");
}

#[test]
fn toffoli_decomposition_matches_the_toffoli_matrix() {
    let n = 3;
    for (c0, c1, t) in [(0, 1, 2), (2, 0, 1), (1, 2, 0), (1, 0, 2)] {
        let mut toffoli = zeros(8);
        for (j, flipped) in (0..8usize)
            .map(|j| j ^ ((bit(j, c0, n) & bit(j, c1, n)) << (n - 1 - t)))
            .enumerate()
        {
            toffoli[flipped][j] = c(1.0, 0.0);
        }
        assert_matches_unitary(&Gate::ccx(c0, c1, t), &toffoli, n, "CCX");
    }
}

#[test]
fn random_circuits_match_their_matrices_on_every_pauli_string() {
    // Circuits are not symmetric under reversal, so this also checks the order of the gates.
    let n = 3;
    let mut rng = StdRng::seed_from_u64(2024);
    for trial in 0..25 {
        let circuit: Vec<Gate> = (0..30).map(|_| random_gate(&mut rng, n)).collect();
        assert_matches_matrices(&circuit, n, &format!("random circuit {trial}"));
    }
}

#[test]
fn unitary_propagation_preserves_the_norm_and_is_undone_by_the_inverse() {
    let n = 6;
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..10 {
        // Rotations only, so that the inverse is the reversed circuit with negated angles.
        let circuit: Vec<Gate> = (0..40)
            .map(|_| {
                let ops: Vec<(usize, PauliOp)> = (0..n)
                    .map(|qubit| (qubit, OPS[rng.gen_range(0..4)]))
                    .collect();
                Gate::rotation(&ops, rng.gen_range(-3.0..3.0))
            })
            .collect();
        let inverse: Vec<Gate> = circuit
            .iter()
            .rev()
            .map(|gate| match gate {
                Gate::Rotation { generator, theta } => Gate::rotation(generator, -theta),
                _ => unreachable!(),
            })
            .collect();

        let start = [(0, PauliOp::Z), (3, PauliOp::X)];
        let mut sum = PauliSum::<1>::new(n).unwrap();
        sum.add(&start, 1.0).unwrap();

        propagate(&circuit, &mut sum, &Truncation::none()).unwrap();
        assert!(sum.len() > 1);
        assert!((sum.norm_sqr() - 1.0).abs() < 1e-11);

        // (U U†)† P (U U†): the inverse acts on the observable first.
        propagate(&inverse, &mut sum, &Truncation::none()).unwrap();
        let start_string = PauliString::<1>::from_ops(&start);
        for (string, coeff) in sum.iter() {
            let expected = if *string == start_string { 1.0 } else { 0.0 };
            assert!((coeff - expected).abs() < 1e-11);
        }
    }
}

#[test]
fn qubits_across_word_boundaries_behave_like_any_others() {
    // The same circuit on qubits 0..4 and on qubits spread over three 64-bit words.
    let n = 4;
    let placement = [62, 64, 127, 130];
    let mut rng = StdRng::seed_from_u64(5);
    let relabel = |gate: &Gate| -> Gate {
        let m = |q: &usize| placement[*q];
        match gate {
            Gate::H(q) => Gate::H(m(q)),
            Gate::X(q) => Gate::X(m(q)),
            Gate::Y(q) => Gate::Y(m(q)),
            Gate::Z(q) => Gate::Z(m(q)),
            Gate::S(q) => Gate::S(m(q)),
            Gate::Sdg(q) => Gate::Sdg(m(q)),
            Gate::Sx(q) => Gate::Sx(m(q)),
            Gate::Sxdg(q) => Gate::Sxdg(m(q)),
            Gate::Cx(a, b) => Gate::Cx(m(a), m(b)),
            Gate::Cy(a, b) => Gate::Cy(m(a), m(b)),
            Gate::Cz(a, b) => Gate::Cz(m(a), m(b)),
            Gate::Swap(a, b) => Gate::Swap(m(a), m(b)),
            Gate::Rotation { generator, theta } => Gate::Rotation {
                generator: generator.iter().map(|(q, op)| (m(q), *op)).collect(),
                theta: *theta,
            },
            Gate::PauliChannel { .. }
            | Gate::AmplitudeDamping { .. }
            | Gate::TransferMap { .. } => {
                unreachable!("random_gate makes Clifford gates and rotations only")
            }
        }
    };

    for _ in 0..20 {
        let circuit: Vec<Gate> = (0..40).map(|_| random_gate(&mut rng, n)).collect();
        let wide_circuit: Vec<Gate> = circuit.iter().map(relabel).collect();

        let mut narrow = PauliSum::<1>::new(n).unwrap();
        narrow
            .add(&[(1, PauliOp::Z), (3, PauliOp::Y)], 1.0)
            .unwrap();
        propagate(&circuit, &mut narrow, &Truncation::none()).unwrap();

        let mut wide = PauliSum::<3>::new(131).unwrap();
        wide.add(
            &[(placement[1], PauliOp::Z), (placement[3], PauliOp::Y)],
            1.0,
        )
        .unwrap();
        propagate(&wide_circuit, &mut wide, &Truncation::none()).unwrap();

        assert_eq!(narrow.len(), wide.len());
        for (string, coeff) in narrow.iter() {
            let ops: Vec<(usize, PauliOp)> = (0..n)
                .map(|q| (placement[q], string.get(q)))
                .filter(|&(_, op)| op != PauliOp::I)
                .collect();
            let wide_coeff = wide.coeff(&PauliString::<3>::from_ops(&ops));
            assert_eq!(coeff, wide_coeff);
        }
    }
}

#[test]
fn truncation_drops_heavy_and_small_terms_after_every_gate() {
    let n = 4;
    let circuit = [
        Gate::rzz(0.3, 0, 1),
        Gate::rx(0.4, 1),
        Gate::rzz(0.3, 1, 2),
        Gate::rx(0.4, 2),
        Gate::rzz(0.3, 2, 3),
    ];
    let mut exact = PauliSum::<1>::new(n).unwrap();
    exact.add(&[(3, PauliOp::X)], 1.0).unwrap();
    propagate(&circuit, &mut exact, &Truncation::none()).unwrap();
    assert!(exact.iter().any(|(p, _)| p.weight() > 2));

    let mut by_weight = PauliSum::<1>::new(n).unwrap();
    by_weight.add(&[(3, PauliOp::X)], 1.0).unwrap();
    let truncation = Truncation {
        max_weight: Some(2),
        ..Truncation::none()
    };
    propagate(&circuit, &mut by_weight, &truncation).unwrap();
    assert!(by_weight.iter().all(|(p, _)| p.weight() <= 2));
    assert!(by_weight.len() < exact.len());

    let mut by_coeff = PauliSum::<1>::new(n).unwrap();
    by_coeff.add(&[(3, PauliOp::X)], 1.0).unwrap();
    let truncation = Truncation {
        min_abs_coeff: 0.05,
        ..Truncation::none()
    };
    propagate(&circuit, &mut by_coeff, &truncation).unwrap();
    assert!(by_coeff.iter().all(|(_, c)| c.abs() >= 0.05));
    assert!(by_coeff.len() < exact.len());
}

#[test]
fn invalid_gates_are_rejected() {
    let mut sum = PauliSum::<1>::new(3).unwrap();
    sum.add(&[(0, PauliOp::Z)], 1.0).unwrap();
    let none = Truncation::none();
    assert!(propagate(&[Gate::H(3)], &mut sum, &none).is_err());
    assert!(propagate(&[Gate::Cx(1, 1)], &mut sum, &none).is_err());
    assert!(propagate(&[Gate::rxx(0.1, 2, 2)], &mut sum, &none).is_err());
    assert!(propagate(&[Gate::rx(f64::NAN, 0)], &mut sum, &none).is_err());
    assert!(PauliSum::<1>::new(65).is_err());
    assert!(sum.add(&[(3, PauliOp::X)], 1.0).is_err());
}
