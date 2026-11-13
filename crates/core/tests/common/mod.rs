//! Dense-matrix helpers shared by the reference tests.
//!
//! Nothing here uses the engine: gates are built from their textbook matrices and operators
//! are expanded in the Pauli basis by traces. Qubit 0 is the most significant bit of a
//! matrix index.
//!
//! Every test that uses these helpers runs under the memory cap of [`memory`].

#![allow(dead_code)]

mod memory;

use num_complex::Complex64 as C;
use pauliprop::{
    propagate, propagate_with, Gate, PauliOp, PauliString, PauliSum, Picture, Truncation,
};
use rand::{rngs::StdRng, Rng};
use std::f64::consts::FRAC_1_SQRT_2;

pub type Matrix = Vec<Vec<C>>;

pub const TOL: f64 = 1e-12;
pub const OPS: [PauliOp; 4] = [PauliOp::I, PauliOp::X, PauliOp::Y, PauliOp::Z];

pub fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

pub fn zeros(dim: usize) -> Matrix {
    vec![vec![c(0.0, 0.0); dim]; dim]
}

pub fn identity(dim: usize) -> Matrix {
    let mut m = zeros(dim);
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = c(1.0, 0.0);
    }
    m
}

pub fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    let dim = a.len();
    let mut out = zeros(dim);
    for i in 0..dim {
        for k in 0..dim {
            if a[i][k] == c(0.0, 0.0) {
                continue;
            }
            for j in 0..dim {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

pub fn dagger(a: &Matrix) -> Matrix {
    let dim = a.len();
    let mut out = zeros(dim);
    for i in 0..dim {
        for j in 0..dim {
            out[j][i] = a[i][j].conj();
        }
    }
    out
}

pub fn trace_of_product(a: &Matrix, b: &Matrix) -> C {
    let dim = a.len();
    let mut t = c(0.0, 0.0);
    for i in 0..dim {
        for j in 0..dim {
            t += a[i][j] * b[j][i];
        }
    }
    t
}

pub fn pauli_matrix(op: PauliOp) -> Matrix {
    let (o, l, i) = (c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0));
    match op {
        PauliOp::I => vec![vec![l, o], vec![o, l]],
        PauliOp::X => vec![vec![o, l], vec![l, o]],
        PauliOp::Y => vec![vec![o, -i], vec![i, o]],
        PauliOp::Z => vec![vec![l, o], vec![o, -l]],
    }
}

/// The bit of `qubit` in the basis state `index` of `n` qubits.
pub fn bit(index: usize, qubit: usize, n: usize) -> usize {
    (index >> (n - 1 - qubit)) & 1
}

/// Embeds the matrix `small`, which acts on `qubits` in that order, into `n` qubits.
pub fn embed(small: &Matrix, qubits: &[usize], n: usize) -> Matrix {
    let dim = 1 << n;
    let k = qubits.len();
    let sub = |index: usize| {
        qubits.iter().enumerate().fold(0, |acc, (pos, &q)| {
            acc | (bit(index, q, n) << (k - 1 - pos))
        })
    };
    let rest = |index: usize| {
        let mut r = index;
        for &q in qubits {
            r &= !(1 << (n - 1 - q));
        }
        r
    };
    let mut out = zeros(dim);
    for i in 0..dim {
        for j in 0..dim {
            if rest(i) == rest(j) {
                out[i][j] = small[sub(i)][sub(j)];
            }
        }
    }
    out
}

/// The matrix of the Pauli string with operator `ops[q]` on qubit `q`.
pub fn string_matrix(ops: &[PauliOp]) -> Matrix {
    let n = ops.len();
    let mut m = identity(1 << n);
    for (q, &op) in ops.iter().enumerate() {
        m = mul(&m, &embed(&pauli_matrix(op), &[q], n));
    }
    m
}

/// The operators of the `index`-th Pauli string on `n` qubits, in base 4.
pub fn ops_of_index(index: usize, n: usize) -> Vec<PauliOp> {
    (0..n).map(|q| OPS[(index >> (2 * q)) & 3]).collect()
}

pub fn sparse(ops: &[PauliOp]) -> Vec<(usize, PauliOp)> {
    ops.iter()
        .copied()
        .enumerate()
        .filter(|&(_, op)| op != PauliOp::I)
        .collect()
}

pub fn controlled(target: &Matrix) -> Matrix {
    let mut m = identity(4);
    for i in 0..2 {
        for j in 0..2 {
            m[2 + i][2 + j] = target[i][j];
        }
    }
    m
}

/// The textbook matrix of a gate on `n` qubits.
pub fn gate_matrix(gate: &Gate, n: usize) -> Matrix {
    let (o, l, i) = (c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0));
    let h = c(FRAC_1_SQRT_2, 0.0);
    let half = c(0.5, 0.0);
    match gate {
        Gate::H(q) => embed(&vec![vec![h, h], vec![h, -h]], &[*q], n),
        Gate::X(q) => embed(&pauli_matrix(PauliOp::X), &[*q], n),
        Gate::Y(q) => embed(&pauli_matrix(PauliOp::Y), &[*q], n),
        Gate::Z(q) => embed(&pauli_matrix(PauliOp::Z), &[*q], n),
        Gate::S(q) => embed(&vec![vec![l, o], vec![o, i]], &[*q], n),
        Gate::Sdg(q) => embed(&vec![vec![l, o], vec![o, -i]], &[*q], n),
        Gate::Sx(q) => embed(
            &vec![
                vec![half * (l + i), half * (l - i)],
                vec![half * (l - i), half * (l + i)],
            ],
            &[*q],
            n,
        ),
        Gate::Sxdg(q) => embed(
            &vec![
                vec![half * (l - i), half * (l + i)],
                vec![half * (l + i), half * (l - i)],
            ],
            &[*q],
            n,
        ),
        Gate::Cx(a, b) => embed(&controlled(&pauli_matrix(PauliOp::X)), &[*a, *b], n),
        Gate::Cy(a, b) => embed(&controlled(&pauli_matrix(PauliOp::Y)), &[*a, *b], n),
        Gate::Cz(a, b) => embed(&controlled(&pauli_matrix(PauliOp::Z)), &[*a, *b], n),
        Gate::Swap(a, b) => embed(
            &vec![
                vec![l, o, o, o],
                vec![o, o, l, o],
                vec![o, l, o, o],
                vec![o, o, o, l],
            ],
            &[*a, *b],
            n,
        ),
        Gate::PauliChannel { .. } | Gate::AmplitudeDamping { .. } | Gate::TransferMap { .. } => {
            panic!("the gate does not carry a unitary matrix")
        }
        Gate::Rotation { generator, theta } => {
            // exp(-i theta / 2 P) = cos(theta / 2) I - i sin(theta / 2) P
            let mut ops = vec![PauliOp::I; n];
            for &(q, op) in generator {
                ops[q] = op;
            }
            let p = string_matrix(&ops);
            let (sin, cos) = (theta / 2.0).sin_cos();
            let mut m = zeros(1 << n);
            for a in 0..1 << n {
                for b in 0..1 << n {
                    let id = if a == b { l } else { o };
                    m[a][b] = c(cos, 0.0) * id - i * c(sin, 0.0) * p[a][b];
                }
            }
            m
        }
    }
}

/// The unitary of a circuit whose gates are listed in the order they act on a state.
pub fn circuit_matrix(circuit: &[Gate], n: usize) -> Matrix {
    let mut u = identity(1 << n);
    for gate in circuit {
        u = mul(&gate_matrix(gate, n), &u);
    }
    u
}

/// Asserts that the engine maps every Pauli string `P` on `n` qubits to `U† P U`.
pub fn assert_matches_unitary(circuit: &[Gate], unitary: &Matrix, n: usize, what: &str) {
    let dim = (1usize << n) as f64;
    let basis: Vec<Matrix> = (0..1 << (2 * n))
        .map(|index| string_matrix(&ops_of_index(index, n)))
        .collect();
    let u_dagger = dagger(unitary);

    for input in 0..1 << (2 * n) {
        let input_ops = ops_of_index(input, n);

        let mut sum = PauliSum::<1>::new(n).unwrap();
        sum.add(&sparse(&input_ops), 1.0).unwrap();
        propagate(circuit, &mut sum, &Truncation::none()).unwrap();

        let evolved = mul(&mul(&u_dagger, &basis[input]), unitary);
        for (output, q_matrix) in basis.iter().enumerate() {
            let expected = trace_of_product(q_matrix, &evolved) / dim;
            assert!(
                expected.im.abs() < TOL,
                "{what}: reference coefficient is not real"
            );
            let string = PauliString::<1>::from_ops(&sparse(&ops_of_index(output, n)));
            let actual = sum.coeff(&string);
            assert!(
                (actual - expected.re).abs() < TOL,
                "{what}: {:?} -> {:?}: engine {actual}, reference {}",
                input_ops,
                ops_of_index(output, n),
                expected.re
            );
        }
    }
}

pub fn assert_matches_matrices(circuit: &[Gate], n: usize, what: &str) {
    assert_matches_unitary(circuit, &circuit_matrix(circuit, n), n, what);
}

pub fn random_gate(rng: &mut StdRng, n: usize) -> Gate {
    let q = rng.gen_range(0..n);
    let mut other = rng.gen_range(0..n - 1);
    if other >= q {
        other += 1;
    }
    let theta = rng.gen_range(-3.0..3.0);
    match rng.gen_range(0..20) {
        0 => Gate::H(q),
        1 => Gate::X(q),
        2 => Gate::Y(q),
        3 => Gate::Z(q),
        4 => Gate::S(q),
        5 => Gate::Sdg(q),
        6 => Gate::Sx(q),
        7 => Gate::Sxdg(q),
        8 => Gate::Cx(q, other),
        9 => Gate::Cy(q, other),
        10 => Gate::Cz(q, other),
        11 => Gate::Swap(q, other),
        12 => Gate::rx(theta, q),
        13 => Gate::ry(theta, q),
        14 => Gate::rz(theta, q),
        15 => Gate::rxx(theta, q, other),
        16 => Gate::ryy(theta, q, other),
        17 => Gate::rzz(theta, q, other),
        18 => Gate::t(q),
        _ => {
            let ops: Vec<(usize, PauliOp)> = (0..n)
                .map(|qubit| (qubit, OPS[rng.gen_range(0..4)]))
                .collect();
            Gate::rotation(&ops, theta)
        }
    }
}

/// One step of a circuit: the gate for the engine and its Kraus operators for the reference.
pub struct Step {
    pub gate: Gate,
    pub kraus: Vec<Matrix>,
}

pub fn scaled(matrix: &Matrix, factor: f64) -> Matrix {
    matrix
        .iter()
        .map(|row| row.iter().map(|&v| v * factor).collect())
        .collect()
}

pub fn unitary_step(gate: Gate, n: usize) -> Step {
    let kraus = vec![gate_matrix(&gate, n)];
    Step { gate, kraus }
}

/// A random Pauli error: the string `errors[k].0` on `qubits` with probability `errors[k].1`.
pub fn pauli_channel_step(qubits: &[usize], errors: &[(Vec<PauliOp>, f64)], n: usize) -> Step {
    let total: f64 = errors.iter().map(|(_, p)| p).sum();
    let mut kraus = vec![scaled(&identity(1 << n), (1.0 - total).sqrt())];
    for (ops, probability) in errors {
        let mut full = vec![PauliOp::I; n];
        for (&q, &op) in qubits.iter().zip(ops) {
            full[q] = op;
        }
        kraus.push(scaled(&string_matrix(&full), probability.sqrt()));
    }
    Step {
        gate: Gate::pauli_channel(qubits, errors).unwrap(),
        kraus,
    }
}

pub fn amplitude_damping_step(gamma: f64, q: usize, n: usize) -> Step {
    let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
    let k0 = vec![vec![l, o], vec![o, c((1.0 - gamma).sqrt(), 0.0)]];
    let k1 = vec![vec![o, c(gamma.sqrt(), 0.0)], vec![o, o]];
    Step {
        gate: Gate::amplitude_damping(gamma, q).unwrap(),
        kraus: vec![embed(&k0, &[q], n), embed(&k1, &[q], n)],
    }
}

/// Asserts that the engine maps every Pauli string on `n` qubits as the Kraus operators do
/// on an observable.
pub fn assert_matches_kraus(steps: &[Step], n: usize, what: &str) {
    assert_matches_kraus_in(Picture::Heisenberg, steps, n, what);
}

/// Asserts that the engine maps every Pauli string on `n` qubits as the Kraus operators do:
/// as an observable, `O -> sum_k K_k† O K_k` with the last step first, or as a state,
/// `rho -> sum_k K_k rho K_k†` with the first step first.
pub fn assert_matches_kraus_in(picture: Picture, steps: &[Step], n: usize, what: &str) {
    let dim = (1usize << n) as f64;
    let basis: Vec<Matrix> = (0..1 << (2 * n))
        .map(|index| string_matrix(&ops_of_index(index, n)))
        .collect();
    let circuit: Vec<Gate> = steps.iter().map(|step| step.gate.clone()).collect();

    for input in 0..1 << (2 * n) {
        let input_ops = ops_of_index(input, n);

        let mut sum = PauliSum::<1>::new(n).unwrap();
        sum.add(&sparse(&input_ops), 1.0).unwrap();
        propagate_with(&circuit, &mut sum, &Truncation::none(), picture, None).unwrap();

        let ordered: Vec<&Step> = match picture {
            Picture::Heisenberg => steps.iter().rev().collect(),
            Picture::Schrodinger => steps.iter().collect(),
        };
        let mut evolved = basis[input].clone();
        for step in ordered {
            let mut next = zeros(1 << n);
            for k in &step.kraus {
                let term = match picture {
                    Picture::Heisenberg => mul(&mul(&dagger(k), &evolved), k),
                    Picture::Schrodinger => mul(&mul(k, &evolved), &dagger(k)),
                };
                for (row, term_row) in next.iter_mut().zip(&term) {
                    for (value, term_value) in row.iter_mut().zip(term_row) {
                        *value += term_value;
                    }
                }
            }
            evolved = next;
        }

        for (output, q_matrix) in basis.iter().enumerate() {
            let expected = trace_of_product(q_matrix, &evolved) / dim;
            assert!(
                expected.im.abs() < TOL,
                "{what}: reference coefficient is not real"
            );
            let string = PauliString::<1>::from_ops(&sparse(&ops_of_index(output, n)));
            let actual = sum.coeff(&string);
            assert!(
                (actual - expected.re).abs() < TOL,
                "{what} ({picture:?}): {:?} -> {:?}: engine {actual}, reference {}",
                input_ops,
                ops_of_index(output, n),
                expected.re
            );
        }
    }
}

/// Random probabilities for `count` errors that sum to less than one.
pub fn random_probabilities(rng: &mut StdRng, count: usize) -> Vec<f64> {
    let weights: Vec<f64> = (0..count).map(|_| rng.gen_range(0.0..1.0)).collect();
    let scale = rng.gen_range(0.1..0.9) / weights.iter().sum::<f64>();
    weights.iter().map(|w| w * scale).collect()
}

/// A random Pauli channel on `qubits` with an error probability for every Pauli string.
pub fn random_pauli_channel_step(rng: &mut StdRng, qubits: &[usize], n: usize) -> Step {
    let k = qubits.len();
    let strings: Vec<Vec<PauliOp>> = (1..1 << (2 * k)).map(|i| ops_of_index(i, k)).collect();
    let probabilities = random_probabilities(rng, strings.len());
    let errors: Vec<_> = strings.into_iter().zip(probabilities).collect();
    pauli_channel_step(qubits, &errors, n)
}

/// A random circuit of Clifford gates, rotations, Pauli channels and amplitude damping.
pub fn random_noisy_steps(rng: &mut StdRng, n: usize, count: usize) -> Vec<Step> {
    (0..count)
        .map(|_| {
            let q = rng.gen_range(0..n);
            let mut other = rng.gen_range(0..n - 1);
            if other >= q {
                other += 1;
            }
            match rng.gen_range(0..6) {
                0 => random_pauli_channel_step(rng, &[q], n),
                1 => random_pauli_channel_step(rng, &[q, other], n),
                2 => amplitude_damping_step(rng.gen_range(0.0..1.0), q, n),
                _ => unitary_step(random_gate(rng, n), n),
            }
        })
        .collect()
}
