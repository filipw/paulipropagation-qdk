//! Checks the engine against the hash-map propagation of `pauliprop::reference`.
//!
//! The two share the rules for dropping terms and nothing else. They have to agree on every
//! term and on every bit of every coefficient, for every kind of gate, every truncation
//! rule and both pictures, and the engine has to give the same whatever the zones and the
//! number of threads.

mod common;

use common::*;
use pauliprop::{
    propagate_in_zones, reference, Coefficient, Complex64, CustomTruncation, Gate, PauliOp,
    PauliString, PauliSum, Picture, Truncation, WeightDamping, Zones,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

/// Layouts from a single zone to more zones than terms.
const LAYOUTS: [Zones; 5] = [
    Zones {
        terms: usize::MAX,
        max: 1,
    },
    Zones { terms: 0, max: 2 },
    Zones { terms: 3, max: 16 },
    Zones { terms: 50, max: 8 },
    Zones { terms: 0, max: 64 },
];

/// Numbers of threads for the default layout, which depends on them.
const THREADS: [usize; 3] = [1, 2, 5];

fn distinct_qubits(rng: &mut StdRng, n: usize, count: usize) -> Vec<usize> {
    let mut qubits: Vec<usize> = (0..n).collect();
    qubits.shuffle(rng);
    qubits.truncate(count);
    qubits
}

fn random_ops(rng: &mut StdRng, n: usize, weight: usize) -> Vec<(usize, PauliOp)> {
    distinct_qubits(rng, n, weight)
        .into_iter()
        .map(|q| (q, OPS[rng.gen_range(1..4)]))
        .collect()
}

/// A gate of any kind on `n` qubits.
fn any_gate(rng: &mut StdRng, n: usize) -> Gate {
    let q = rng.gen_range(0..n);
    let strength = rng.gen_range(0.0..0.3);
    match rng.gen_range(0..24) {
        0..=7 => random_gate(rng, n),
        8..=11 => {
            let weight = rng.gen_range(1..=n.min(4));
            Gate::rotation(&random_ops(rng, n, weight), rng.gen_range(-3.0..3.0))
        }
        // A generator on many qubits flips many separate runs of bits.
        12 => {
            let weight = rng.gen_range(1..=n.min(12));
            Gate::rotation(&random_ops(rng, n, weight), rng.gen_range(-3.0..3.0))
        }
        13 => Gate::depolarizing(strength, q).unwrap(),
        14 => Gate::pauli_x_noise(strength, q).unwrap(),
        15 => Gate::pauli_y_noise(strength, q).unwrap(),
        16 => Gate::pauli_z_noise(strength, q).unwrap(),
        17 | 18 => Gate::amplitude_damping(rng.gen_range(0.0..0.6), q).unwrap(),
        19 => {
            let qubits = distinct_qubits(rng, n, 2);
            let errors: Vec<(Vec<PauliOp>, f64)> = (0..4)
                .map(|_| {
                    let ops = vec![OPS[rng.gen_range(0..4)], OPS[rng.gen_range(1..4)]];
                    (ops, rng.gen_range(0.0..0.05))
                })
                .collect();
            Gate::pauli_channel(&qubits, &errors).unwrap()
        }
        20 | 21 => {
            // A map that sends a Pauli string to itself and a few others.
            let count = rng.gen_range(1..=2);
            let qubits = distinct_qubits(rng, n, count);
            let dim = 1 << (2 * qubits.len());
            let matrix: Vec<f64> = (0..dim * dim)
                .map(|entry| {
                    if entry % (dim + 1) == 0 {
                        rng.gen_range(0.5..1.0)
                    } else if rng.gen_bool(1.0 / dim as f64) {
                        rng.gen_range(-1.0..1.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            Gate::transfer_map(&qubits, &matrix).unwrap()
        }
        _ => {
            let qubits = distinct_qubits(rng, n, 2);
            let inner = [
                random_gate(rng, 2),
                random_gate(rng, 2),
                random_gate(rng, 2),
            ];
            let matrix: Vec<Complex64> = circuit_matrix(&inner, 2).into_iter().flatten().collect();
            Gate::unitary(&qubits, &matrix).unwrap()
        }
    }
}

fn any_circuit(rng: &mut StdRng, n: usize, gates: usize) -> Vec<Gate> {
    (0..gates).map(|_| any_gate(rng, n)).collect()
}

/// The gate with qubit `q` replaced by `place[q]`.
fn moved(gate: &Gate, place: &[usize]) -> Gate {
    let all = |qubits: &[usize]| -> Vec<usize> { qubits.iter().map(|&q| place[q]).collect() };
    match gate {
        Gate::H(q) => Gate::H(place[*q]),
        Gate::X(q) => Gate::X(place[*q]),
        Gate::Y(q) => Gate::Y(place[*q]),
        Gate::Z(q) => Gate::Z(place[*q]),
        Gate::S(q) => Gate::S(place[*q]),
        Gate::Sdg(q) => Gate::Sdg(place[*q]),
        Gate::Sx(q) => Gate::Sx(place[*q]),
        Gate::Sxdg(q) => Gate::Sxdg(place[*q]),
        Gate::Cx(a, b) => Gate::Cx(place[*a], place[*b]),
        Gate::Cy(a, b) => Gate::Cy(place[*a], place[*b]),
        Gate::Cz(a, b) => Gate::Cz(place[*a], place[*b]),
        Gate::Swap(a, b) => Gate::Swap(place[*a], place[*b]),
        Gate::Rotation { generator, theta } => Gate::Rotation {
            generator: generator.iter().map(|&(q, op)| (place[q], op)).collect(),
            theta: *theta,
        },
        Gate::PauliChannel { qubits, factors } => Gate::PauliChannel {
            qubits: all(qubits),
            factors: factors.clone(),
        },
        Gate::AmplitudeDamping { qubit, gamma } => Gate::AmplitudeDamping {
            qubit: place[*qubit],
            gamma: *gamma,
        },
        Gate::TransferMap { qubits, columns } => Gate::TransferMap {
            qubits: all(qubits),
            columns: columns.clone(),
        },
    }
}

/// A sum of a few light Pauli strings, added in no particular order and some of them twice.
fn any_sum<const W: usize, C: Coefficient>(
    rng: &mut StdRng,
    n: usize,
    coeff: impl Fn(&mut StdRng) -> C,
) -> PauliSum<W, C> {
    let mut sum = PauliSum::new(n).unwrap();
    for _ in 0..rng.gen_range(1..12) {
        let weight = rng.gen_range(0..=n.min(3));
        let ops = random_ops(rng, n, weight);
        sum.add(&ops, coeff(rng)).unwrap();
        if rng.gen_bool(0.2) {
            sum.add(&ops, coeff(rng)).unwrap();
        }
    }
    sum
}

fn real(rng: &mut StdRng) -> f64 {
    // Coefficients of very different sizes, for the relative threshold to tell apart.
    rng.gen_range(-1.0..1.0) * 10f64.powi(-rng.gen_range(0..4))
}

fn complex(rng: &mut StdRng) -> Complex64 {
    Complex64::new(real(rng), real(rng))
}

fn truncations() -> Vec<(&'static str, Truncation)> {
    let none = Truncation::none();
    vec![
        ("none", none),
        ("default", Truncation::default()),
        (
            "max_weight",
            Truncation {
                max_weight: Some(3),
                ..none
            },
        ),
        (
            "min_abs_coeff",
            Truncation {
                min_abs_coeff: 1e-3,
                ..none
            },
        ),
        (
            "min_rel_coeff",
            Truncation {
                min_rel_coeff: Some(1e-2),
                ..none
            },
        ),
        (
            "min_rel_coeff and max_weight",
            Truncation {
                min_rel_coeff: Some(1e-3),
                max_weight: Some(3),
                min_abs_coeff: 1e-6,
                ..none
            },
        ),
        (
            "max_freq",
            Truncation {
                max_freq: Some(5),
                ..none
            },
        ),
        (
            "max_sins",
            Truncation {
                max_sins: Some(2),
                ..none
            },
        ),
        (
            "weight_damping",
            Truncation {
                weight_damping: Some(WeightDamping {
                    gamma: 0.4,
                    min_abs_coeff: 1e-3,
                }),
                ..none
            },
        ),
        (
            "all",
            Truncation {
                max_weight: Some(4),
                min_abs_coeff: 1e-5,
                min_rel_coeff: Some(1e-4),
                max_freq: Some(9),
                max_sins: Some(4),
                weight_damping: Some(WeightDamping {
                    gamma: 0.1,
                    min_abs_coeff: 1e-5,
                }),
            },
        ),
    ]
}

/// Propagates `start` with the reference and with the engine in every layout, and returns
/// the terms, which all of them have to agree on.
fn agreed_terms<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    start: &PauliSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    custom: Option<&CustomTruncation<'_, W, C>>,
    what: &str,
) -> Vec<(PauliString<W>, C)> {
    let mut expected = start.clone();
    reference::propagate_with(circuit, &mut expected, truncation, picture, custom).unwrap();
    let expected = expected.sorted_terms();

    let check = |zones: &Zones, layout: &str| {
        let mut sum = start.clone();
        propagate_in_zones(circuit, &mut sum, truncation, picture, custom, zones).unwrap();
        let terms = sum.sorted_terms();
        assert_eq!(
            terms.len(),
            expected.len(),
            "{what}, {layout}: number of terms"
        );
        for (ours, theirs) in terms.iter().zip(&expected) {
            assert_eq!(ours, theirs, "{what}, {layout}");
        }
    };
    for zones in &LAYOUTS {
        check(zones, &format!("{zones:?}"));
    }
    for threads in THREADS {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| check(&Zones::default(), &format!("{threads} threads")));
    }
    expected
}

/// Runs every truncation in both pictures on random circuits and returns, per truncation,
/// how many terms came out in total.
fn sweep<const W: usize, C: Coefficient>(
    seed: u64,
    n: usize,
    gates: usize,
    circuits: usize,
    always: &Truncation,
    coeff: impl Fn(&mut StdRng) -> C + Copy,
) -> Vec<(&'static str, usize)> {
    let place: Vec<usize> = (0..n).collect();
    sweep_on::<W, C>(seed, &place, gates, circuits, always, coeff)
}

/// [`sweep`] on the qubits `place` of a register that ends with the last of them.
fn sweep_on<const W: usize, C: Coefficient>(
    seed: u64,
    place: &[usize],
    gates: usize,
    circuits: usize,
    always: &Truncation,
    coeff: impl Fn(&mut StdRng) -> C + Copy,
) -> Vec<(&'static str, usize)> {
    let n = place.len();
    let register = place.iter().max().unwrap() + 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let cases: Vec<(Vec<Gate>, PauliSum<W, C>)> = (0..circuits)
        .map(|_| {
            let circuit = any_circuit(&mut rng, n, gates)
                .iter()
                .map(|gate| moved(gate, place))
                .collect();
            let start = any_sum::<W, C>(&mut rng, n, coeff);
            (circuit, placed(&start, place, register))
        })
        .collect();
    sweep_over(&cases, always)
}

/// The sum with qubit `q` of every string at `place[q]` of a register of `register` qubits.
fn placed<const W: usize, C: Coefficient>(
    sum: &PauliSum<W, C>,
    place: &[usize],
    register: usize,
) -> PauliSum<W, C> {
    let mut placed = PauliSum::new(register).unwrap();
    for (string, coeff) in sum.sorted_terms() {
        let ops: Vec<(usize, PauliOp)> = place
            .iter()
            .enumerate()
            .map(|(q, &at)| (at, string.get(q)))
            .collect();
        placed.add(&ops, coeff).unwrap();
    }
    placed
}

/// Runs every truncation in both pictures on the circuits and sums of `cases` and returns,
/// per truncation, how many terms came out in total.
fn sweep_over<const W: usize, C: Coefficient>(
    cases: &[(Vec<Gate>, PauliSum<W, C>)],
    always: &Truncation,
) -> Vec<(&'static str, usize)> {
    let mut totals: Vec<(&'static str, usize)> =
        truncations().iter().map(|(name, _)| (*name, 0)).collect();
    for (index, (circuit, start)) in cases.iter().enumerate() {
        for (total, (name, truncation)) in totals.iter_mut().zip(truncations()) {
            // The rules of `always` bound the sum where the register is too wide to go
            // without.
            let truncation = Truncation {
                max_weight: truncation.max_weight.or(always.max_weight),
                min_abs_coeff: truncation.min_abs_coeff.max(always.min_abs_coeff),
                ..truncation
            };
            for picture in [Picture::Heisenberg, Picture::Schrodinger] {
                let what = format!("circuit {index}, {name}, {picture:?}");
                let terms = agreed_terms(circuit, start, &truncation, picture, None, &what);
                total.1 += terms.len();
            }
        }
    }
    totals
}

/// Every rule has to drop terms somewhere, or the sweep says nothing about it.
fn assert_every_rule_acts(totals: &[(&'static str, usize)], at_least: usize) {
    let untruncated = totals[0].1;
    assert!(untruncated >= at_least, "only {untruncated} terms in all");
    for (name, total) in &totals[2..] {
        assert!(
            *total < untruncated,
            "{name} dropped nothing: {total} of {untruncated} terms"
        );
        assert!(*total > 0, "{name} dropped everything");
    }
}

#[test]
fn real_sums_on_a_few_qubits() {
    let totals = sweep::<1, f64>(1, 5, 40, 12, &Truncation::none(), real);
    assert_every_rule_acts(&totals, 5_000);
}

#[test]
fn real_sums_on_more_qubits() {
    let bound = Truncation {
        max_weight: Some(6),
        ..Truncation::none()
    };
    let totals = sweep::<1, f64>(2, 9, 60, 4, &bound, real);
    assert_every_rule_acts(&totals, 20_000);
}

#[test]
fn complex_sums() {
    let totals = sweep::<1, Complex64>(3, 5, 40, 6, &Truncation::none(), complex);
    assert_every_rule_acts(&totals, 2_000);
}

#[test]
fn sums_on_registers_of_several_words() {
    // Qubits on both sides of the ends of the words that hold 64 of them each.
    let totals = sweep_on::<2, f64>(
        4,
        &[62, 63, 64, 65, 0, 100],
        40,
        6,
        &Truncation::none(),
        real,
    );
    assert_every_rule_acts(&totals, 5_000);
    let place = [63, 64, 127, 128, 191, 192, 5, 250];
    let bound = Truncation {
        max_weight: Some(6),
        ..Truncation::none()
    };
    let totals = sweep_on::<4, Complex64>(5, &place, 50, 3, &bound, complex);
    assert_every_rule_acts(&totals, 5_000);
}

#[test]
fn gates_on_a_register_that_the_sum_does_not_fill() {
    // Most gates act where no term has anything but the identity, at first, and the strings
    // spread over the register a few qubits at a time.
    let register = 40;
    let mut rng = StdRng::seed_from_u64(8);
    let cases: Vec<(Vec<Gate>, PauliSum<1, f64>)> = (0..4)
        .map(|_| {
            let circuit: Vec<Gate> = (0..120)
                .map(|_| {
                    let place = distinct_qubits(&mut rng, register, 3);
                    moved(&any_gate(&mut rng, 3), &place)
                })
                .collect();
            let start = any_sum::<1, f64>(&mut rng, 3, real);
            let place = distinct_qubits(&mut rng, register, 3);
            let missed = circuit.iter().filter(|gate| misses(gate, &place)).count();
            assert!(missed > 80, "only {missed} gates away from the sum");
            (circuit, placed(&start, &place, register))
        })
        .collect();
    let bound = Truncation {
        max_weight: Some(5),
        min_abs_coeff: 1e-5,
        ..Truncation::none()
    };
    let totals = sweep_over(&cases, &bound);
    assert_every_rule_acts(&totals, 10_000);
}

#[test]
fn clifford_gates_on_a_sum_that_no_gate_has_truncated() {
    // Until a gate truncates every term, a Clifford gate is all that applies the rules that
    // read the Pauli string, to the terms it moves and to those it leaves where they are.
    let mut rng = StdRng::seed_from_u64(9);
    let n = 6;
    let mut totals: Vec<(&'static str, bool, usize)> = truncations()
        .iter()
        .map(|(name, rules)| {
            let reads = rules.max_weight.is_some() || rules.weight_damping.is_some();
            (*name, reads, 0)
        })
        .collect();
    let (mut with_custom, mut moved_or_not) = (0, [0, 0]);
    for index in 0..30 {
        let length = rng.gen_range(1..10);
        let mut circuit: Vec<Gate> = Vec::new();
        while circuit.len() < length {
            let gate = random_gate(&mut rng, n);
            if !matches!(gate, Gate::Rotation { .. }) {
                moved_or_not[usize::from(matches!(gate, Gate::X(_) | Gate::Y(_) | Gate::Z(_)))] +=
                    1;
                circuit.push(gate);
            }
        }
        // Strings of every weight with coefficients of very different sizes.
        let mut start: PauliSum<1, f64> = PauliSum::new(n).unwrap();
        for _ in 0..40 {
            let weight = rng.gen_range(0..=n);
            let ops = random_ops(&mut rng, n, weight);
            start.add(&ops, real(&mut rng)).unwrap();
        }
        let rule = |string: &PauliString<1>, coeff: f64| string.weight() > 2 && coeff.abs() < 0.05;
        for picture in [Picture::Heisenberg, Picture::Schrodinger] {
            for (total, (name, truncation)) in totals.iter_mut().zip(truncations()) {
                let what = format!("Clifford circuit {index}, {name}, {picture:?}");
                total.2 += agreed_terms(&circuit, &start, &truncation, picture, None, &what).len();
            }
            let what = format!("Clifford circuit {index}, custom rule, {picture:?}");
            let none = Truncation::none();
            with_custom += agreed_terms(&circuit, &start, &none, picture, Some(&rule), &what).len();
        }
    }
    assert!(
        moved_or_not[0] > 20 && moved_or_not[1] > 20,
        "{moved_or_not:?}"
    );
    let all = totals[0].2;
    for (name, reads, total) in totals {
        if reads {
            assert!(
                total > 0 && total < all,
                "{name} kept {total} of {all} terms"
            );
        } else {
            assert_eq!(total, all, "{name} dropped terms after Clifford gates");
        }
    }
    assert!(with_custom > 0 && with_custom < all);
}

/// Whether the gate acts on none of `qubits`.
fn misses(gate: &Gate, qubits: &[usize]) -> bool {
    let away = |on: &[usize]| on.iter().all(|q| !qubits.contains(q));
    match gate {
        Gate::H(q)
        | Gate::X(q)
        | Gate::Y(q)
        | Gate::Z(q)
        | Gate::S(q)
        | Gate::Sdg(q)
        | Gate::Sx(q)
        | Gate::Sxdg(q) => away(&[*q]),
        Gate::Cx(a, b) | Gate::Cy(a, b) | Gate::Cz(a, b) | Gate::Swap(a, b) => away(&[*a, *b]),
        Gate::Rotation { generator, .. } => generator.iter().all(|(q, _)| !qubits.contains(q)),
        Gate::PauliChannel { qubits: on, .. } | Gate::TransferMap { qubits: on, .. } => away(on),
        Gate::AmplitudeDamping { qubit, .. } => away(&[*qubit]),
    }
}

#[test]
fn a_custom_rule() {
    let mut rng = StdRng::seed_from_u64(6);
    let (mut with, mut without) = (0, 0);
    for index in 0..8 {
        let circuit = any_circuit(&mut rng, 6, 40);
        let start: PauliSum<1, f64> = any_sum(&mut rng, 6, real);
        let rule = |string: &PauliString<1>, coeff: f64| {
            (string.get(0) == PauliOp::Y && coeff.abs() < 0.1)
                || (string.weight() > 3 && coeff.abs() < 1e-3)
        };
        for picture in [Picture::Heisenberg, Picture::Schrodinger] {
            let what = format!("circuit {index}, {picture:?}");
            let truncation = Truncation::default();
            with += agreed_terms(&circuit, &start, &truncation, picture, Some(&rule), &what).len();
            without += agreed_terms(&circuit, &start, &truncation, picture, None, &what).len();
        }
    }
    assert!(with > 5_000 && with < without, "{with} of {without} terms");
}

#[test]
fn circuits_of_rotations_that_grow_a_large_sum() {
    // Most of the time goes into rotations on sums of many terms, in many zones.
    let mut rng = StdRng::seed_from_u64(7);
    let n = 12;
    let mut circuit = Vec::new();
    for _ in 0..10 {
        for q in 0..n {
            circuit.push(Gate::rx(rng.gen_range(0.2..1.2), q));
            circuit.push(Gate::rzz(rng.gen_range(0.2..1.2), q, (q + 1) % n));
        }
    }
    let mut start: PauliSum<1, f64> = PauliSum::new(n).unwrap();
    start.add(&[(5, PauliOp::Z)], 1.0).unwrap();
    let truncation = Truncation {
        max_weight: Some(7),
        min_abs_coeff: 1e-7,
        ..Truncation::none()
    };
    for picture in [Picture::Heisenberg, Picture::Schrodinger] {
        let terms = agreed_terms(&circuit, &start, &truncation, picture, None, "rotations");
        assert!(terms.len() > 20_000, "only {} terms", terms.len());
    }
}
