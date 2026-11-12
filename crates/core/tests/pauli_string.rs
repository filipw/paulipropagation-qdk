//! Reading and writing the operators of a Pauli string, checked against a plain list of
//! operators.

use pauliprop::{PauliOp, PauliString};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const OPS: [PauliOp; 4] = [PauliOp::I, PauliOp::X, PauliOp::Y, PauliOp::Z];

/// What a string should hold, one operator per qubit.
fn check<const W: usize>(string: &PauliString<W>, expected: &[PauliOp]) {
    for (q, &op) in expected.iter().enumerate() {
        assert_eq!(string.get(q), op, "qubit {q}");
        assert_eq!((string.x_bit(q), string.z_bit(q)), op.bits(), "qubit {q}");
    }
    let weight = expected.iter().filter(|&&op| op != PauliOp::I).count();
    assert_eq!(string.weight() as usize, weight);
    assert_eq!(
        string.max_qubit(),
        expected.iter().rposition(|&op| op != PauliOp::I)
    );
    let label: String = expected.iter().map(|op| op.to_char()).collect();
    assert_eq!(string.label(expected.len()), label);
    assert_eq!(
        string.has_x_or_y(),
        expected
            .iter()
            .any(|&op| op == PauliOp::X || op == PauliOp::Y)
    );
    assert_eq!(
        string.has_y_or_z(),
        expected
            .iter()
            .any(|&op| op == PauliOp::Y || op == PauliOp::Z)
    );
}

fn random_writes<const W: usize>(seed: u64) {
    let n = 64 * W;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut string = PauliString::<W>::identity();
    let mut expected = vec![PauliOp::I; n];
    check(&string, &expected);

    for step in 0..2000 {
        // The ends of every word are written more often than the rest.
        let q = if rng.gen_bool(0.3) {
            let word = rng.gen_range(0..W);
            64 * word + [0, 1, 62, 63][rng.gen_range(0..4)]
        } else {
            rng.gen_range(0..n)
        };
        let op = OPS[rng.gen_range(0..4)];
        string.set(q, op);
        expected[q] = op;
        assert_eq!(string.get(q), op);
        if step % 100 == 0 {
            check(&string, &expected);
        }
    }
    check(&string, &expected);

    let pairs: Vec<(usize, PauliOp)> = expected.iter().copied().enumerate().collect();
    assert_eq!(PauliString::<W>::from_ops(&pairs), string);
    // The identity may be left out, and the order of the pairs does not matter.
    let mut sparse: Vec<(usize, PauliOp)> = pairs
        .iter()
        .copied()
        .filter(|&(_, op)| op != PauliOp::I)
        .collect();
    sparse.reverse();
    assert_eq!(PauliString::<W>::from_ops(&sparse), string);
}

#[test]
fn operators_are_written_and_read_by_qubit_on_every_width() {
    random_writes::<1>(1);
    random_writes::<2>(2);
    random_writes::<4>(4);
    random_writes::<8>(8);
}

#[test]
fn writing_one_qubit_leaves_the_others() {
    // X on qubit 1 of four, then Z on qubit 0.
    let mut string = PauliString::<1>::from_ops(&[(1, PauliOp::X)]);
    assert_eq!(string.get(0), PauliOp::I);
    assert_eq!(string.get(1), PauliOp::X);
    string.set(0, PauliOp::Z);
    assert_eq!(
        string,
        PauliString::from_ops(&[(0, PauliOp::Z), (1, PauliOp::X)])
    );
    assert_eq!(string.label(4), "ZXII");

    // Writing the identity removes an operator.
    string.set(1, PauliOp::I);
    assert_eq!(string, PauliString::from_ops(&[(0, PauliOp::Z)]));
    string.set(0, PauliOp::I);
    assert_eq!(string, PauliString::identity());
}

#[test]
fn a_qubit_given_twice_keeps_the_last_operator() {
    let string = PauliString::<2>::from_ops(&[(70, PauliOp::X), (70, PauliOp::Z)]);
    assert_eq!(string, PauliString::from_ops(&[(70, PauliOp::Z)]));
}

#[test]
fn operators_convert_to_bits_and_characters_and_back() {
    for op in OPS {
        let (x, z) = op.bits();
        assert_eq!(PauliOp::from_bits(x, z), op);
        assert_eq!(PauliOp::from_char(op.to_char()), Some(op));
        assert_eq!(
            PauliOp::from_char(op.to_char().to_ascii_lowercase()),
            Some(op)
        );
    }
    assert_eq!(PauliOp::from_char('A'), None);
    assert_eq!(OPS.map(PauliOp::to_char).iter().collect::<String>(), "IXYZ");
}
