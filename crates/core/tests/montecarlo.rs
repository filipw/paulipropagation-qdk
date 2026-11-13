//! Checks the random methods by their averages over many seeds, which must agree with the
//! propagation without sampling, and by what they must keep exactly.

mod common;

use common::*;
use num_complex::Complex64;
use pauliprop::{
    estimate_mse, propagate_resampling, propagate_with, resample, sample_paths, sample_sum,
    Coefficient, Gate, PathLimits, PauliOp, PauliString, PauliSum, Picture, Resampling, Rng, State,
    Truncation,
};
use rand::{rngs::StdRng, Rng as _, SeedableRng};
use std::collections::BTreeMap;

type Sum = PauliSum<1>;
type Terms = BTreeMap<PauliString<1>, f64>;

/// The mean of many values and its standard error.
#[derive(Default, Clone, Copy)]
struct Mean {
    count: f64,
    total: f64,
    squares: f64,
}

impl Mean {
    fn add(&mut self, value: f64) {
        self.count += 1.0;
        self.total += value;
        self.squares += value * value;
    }

    fn value(&self) -> f64 {
        self.total / self.count
    }

    fn error(&self) -> f64 {
        let variance = (self.squares / self.count - self.value().powi(2)).max(0.0);
        (variance / self.count).sqrt()
    }

    /// Whether the mean is within six standard errors of `expected`.
    fn agrees_with(&self, expected: f64) -> bool {
        (self.value() - expected).abs() <= 6.0 * self.error() + 1e-12
    }
}

fn random_sum(rng: &mut StdRng, n: usize, terms: usize) -> Sum {
    let mut sum = Sum::new(n).unwrap();
    while sum.len() < terms {
        let ops: Vec<PauliOp> = (0..n).map(|_| OPS[rng.gen_range(0..4)]).collect();
        // Sizes over several orders of magnitude, so that some terms are heavy.
        let size = 10f64.powf(rng.gen_range(-3.0..0.0));
        let coeff = if rng.gen_bool(0.5) { size } else { -size };
        sum.add(&sparse(&ops), coeff).unwrap();
    }
    sum
}

const METHODS: [Resampling; 4] = [
    Resampling::Multinomial,
    Resampling::Systematic {
        calibrate: false,
        rtol: 0.0,
        atol: 0.0,
    },
    Resampling::Systematic {
        calibrate: true,
        rtol: 0.01,
        atol: 1.0,
    },
    Resampling::SemiDeterministic,
];

/// Seeds, the first six numbers and the 1001st number of xoshiro256** whose state is filled
/// with four numbers of splitmix64 started at the seed. They were printed by the reference
/// implementations of the two generators by Blackman and Vigna, `xoshiro256starstar.c` and
/// `splitmix.c` of <https://prng.di.unimi.it>.
const GENERATOR_VECTORS: [(u64, [u64; 6], u64); 5] = [
    (
        0,
        [
            11_091_344_671_253_066_420,
            13_793_997_310_169_335_082,
            1_900_383_378_846_508_768,
            7_684_712_102_626_143_532,
            13_521_403_990_117_723_737,
            18_442_103_541_295_991_498,
        ],
        3_215_403_766_075_632_002,
    ),
    (
        1,
        [
            12_966_619_160_104_079_557,
            9_600_361_134_598_540_522,
            10_590_380_919_521_690_900,
            7_218_738_570_589_545_383,
            12_860_671_823_995_680_371,
            2_648_436_617_965_840_162,
        ],
        14_841_950_361_884_779_394,
    ),
    (
        42,
        [
            1_546_998_764_402_558_742,
            6_990_951_692_964_543_102,
            12_544_586_762_248_559_009,
            17_057_574_109_182_124_193,
            18_295_552_978_065_317_476,
            14_199_186_830_065_750_584,
        ],
        1_103_426_724_799_410_610,
    ),
    (
        1_234_567,
        [
            3_504_822_795_582_309_479,
            1_819_558_768_956_484_042,
            1_250_851_346_055_027_673,
            16_940_231_675_099_994_102,
            11_585_879_347_611_423_030,
            8_134_400_763_355_999_650,
        ],
        8_702_727_793_300_186_723,
    ),
    (
        u64::MAX,
        [
            10_328_197_420_357_168_392,
            14_156_678_507_024_973_869,
            9_357_971_779_955_476_126,
            13_791_585_006_304_312_367,
            10_463_432_026_814_718_762,
            13_498_236_496_097_551_653,
        ],
        1_855_756_785_506_932_879,
    ),
];

#[test]
fn the_generator_gives_the_numbers_of_its_reference_implementation() {
    for (seed, first, later) in GENERATOR_VECTORS {
        let mut rng = Rng::new(seed);
        let numbers: Vec<u64> = (0..1001).map(|_| rng.next_u64()).collect();
        assert_eq!(numbers[..6], first, "seed {seed}");
        assert_eq!(numbers[1000], later, "seed {seed}");

        // A number in [0, 1) is the upper 53 bits of the next number.
        let uniform = Rng::new(seed).uniform();
        assert_eq!(uniform, (first[0] >> 11) as f64 / 9_007_199_254_740_992.0);
    }
}

#[test]
fn the_generator_gives_the_same_numbers_for_the_same_seed() {
    let (mut a, mut b, mut other) = (Rng::new(7), Rng::new(7), Rng::new(8));
    let numbers: Vec<f64> = (0..10_000).map(|_| a.uniform()).collect();
    assert!(numbers.iter().all(|&x| (0.0..1.0).contains(&x)));
    assert!(numbers.iter().all(|&x| x == b.uniform()));
    assert!(numbers.iter().any(|&x| x != other.uniform()));
    let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
    let below_tenth = numbers.iter().filter(|&&x| x < 0.1).count();
    assert!((mean - 0.5).abs() < 0.01 && (900..1100).contains(&below_tenth));
}

#[test]
fn resampling_is_unbiased() {
    let mut rng = StdRng::seed_from_u64(41);
    let sum = random_sum(&mut rng, 4, 60);
    let original = sum.sorted_terms();
    for method in METHODS {
        let mut means = vec![Mean::default(); original.len()];
        let mut sizes = Mean::default();
        for seed in 0..4000 {
            let mut resampled = sum.clone();
            resample(&mut resampled, 20, Some(method), false, &mut Rng::new(seed)).unwrap();
            sizes.add(resampled.len() as f64);
            for (mean, (string, _)) in means.iter_mut().zip(&original) {
                mean.add(resampled.coeff(string));
            }
        }
        for (mean, (string, coeff)) in means.iter().zip(&original) {
            assert!(
                mean.agrees_with(*coeff),
                "{method:?}: {} has {coeff}, mean {} +- {}",
                string.label(4),
                mean.value(),
                mean.error()
            );
        }
        assert!(
            sizes.value() > 8.0 && sizes.value() <= 21.0,
            "{method:?}: {}",
            sizes.value()
        );
    }
}

#[test]
fn resampling_keeps_what_it_must() {
    let mut rng = StdRng::seed_from_u64(42);
    for round in 0..200 {
        let sum = random_sum(&mut rng, 5, 150);
        let original = sum.sorted_terms();
        let total: f64 = original.iter().map(|t| t.1.abs()).sum();
        let target = 1 + round % 100;
        for method in METHODS {
            let mut resampled = sum.clone();
            resample(
                &mut resampled,
                target,
                Some(method),
                false,
                &mut Rng::new(round as u64),
            )
            .unwrap();
            let terms = resampled.sorted_terms();
            let weight: f64 = terms.iter().map(|t| t.1.abs()).sum();

            // Every survivor is a term of the sum with its sign, and none is zero.
            for (string, coeff) in &terms {
                let before = sum.coeff(string);
                assert!(before != 0.0 && coeff.signum() == before.signum() && *coeff != 0.0);
            }
            match method {
                Resampling::Multinomial => {
                    assert!(terms.len() <= target);
                    assert!((weight - total).abs() < 1e-9 * total);
                }
                Resampling::Systematic {
                    calibrate: false, ..
                } => {
                    // The comb has as many teeth as the target, give or take the one at the end.
                    assert!(terms.len() <= target);
                    assert!((weight - total).abs() <= total / target as f64 * (1.0 + 1e-9));
                }
                // A narrowed comb keeps the target on average, which the test of the
                // averages checks. A single run can keep a term more for every stretch of
                // light terms between heavy ones.
                Resampling::Systematic { .. } => assert!(terms.len() <= 2 * target),
                Resampling::SemiDeterministic => {
                    assert!(terms.len() <= target + 1);
                    for (string, coeff) in &original {
                        if coeff.abs() > total / target as f64 {
                            assert_eq!(resampled.coeff(string), *coeff);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn a_comb_gives_every_term_the_teeth_on_its_interval() {
    let mut rng = StdRng::seed_from_u64(43);
    for seed in 0..50 {
        let sum = random_sum(&mut rng, 5, 80);
        let original = sum.sorted_terms();
        let target = 25;
        let total: f64 = original.iter().map(|t| t.1.abs()).sum();
        let spacing = total / target as f64;
        // The comb takes its offset as the first number of the generator. Its teeth are at
        // (k + offset) * spacing for k = 0, 1, ...
        let offset = Rng::new(seed).uniform();
        let teeth: Vec<f64> = (0..=target)
            .map(|k| (k as f64 + offset) * spacing)
            .collect();

        let mut resampled = sum.clone();
        let method = Resampling::Systematic {
            calibrate: false,
            rtol: 0.0,
            atol: 0.0,
        };
        resample(
            &mut resampled,
            target,
            Some(method),
            false,
            &mut Rng::new(seed),
        )
        .unwrap();

        let mut start = 0.0;
        for (string, coeff) in &original {
            let end = start + coeff.abs();
            // A tooth that is on the boundary of an interval to rounding may go either way.
            let near = |tooth: &f64| (tooth - start).abs() < 1e-9 || (tooth - end).abs() < 1e-9;
            if !teeth.iter().any(near) {
                let hits = teeth
                    .iter()
                    .filter(|&&tooth| start < tooth && tooth <= end)
                    .count();
                let expected = (hits as f64 * spacing).copysign(*coeff);
                let got = resampled.coeff(string);
                assert!((got - if hits == 0 { 0.0 } else { expected }).abs() < 1e-12);
            }
            start = end;
        }
    }
}

#[test]
fn resampling_by_squared_coefficients_keeps_the_sum_of_squares() {
    let mut rng = StdRng::seed_from_u64(44);
    let sum = random_sum(&mut rng, 5, 150);
    let squares: f64 = sum.sorted_terms().iter().map(|t| t.1 * t.1).sum();
    let mut shares = Mean::default();
    let heaviest = sum
        .sorted_terms()
        .into_iter()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap();
    for seed in 0..2000 {
        // Without a method, squared sampling is multinomial.
        let mut resampled = sum.clone();
        resample(&mut resampled, 30, None, true, &mut Rng::new(seed)).unwrap();
        let terms = resampled.sorted_terms();
        let kept: f64 = terms.iter().map(|t| t.1 * t.1).sum();
        assert!((kept - squares).abs() < 1e-9 * squares);
        assert!(terms.len() <= 30 && terms.iter().all(|t| t.1 > 0.0));
        shares.add(resampled.coeff(&heaviest.0).powi(2));
    }
    // The square of a coefficient is kept on average.
    assert!(shares.agrees_with(heaviest.1 * heaviest.1));
}

#[test]
fn complex_coefficients_keep_their_phases() {
    let mut rng = StdRng::seed_from_u64(45);
    let mut sum = PauliSum::<1, Complex64>::new(5).unwrap();
    while sum.len() < 100 {
        let ops: Vec<PauliOp> = (0..5).map(|_| OPS[rng.gen_range(0..4)]).collect();
        let coeff = Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        sum.add(&sparse(&ops), coeff).unwrap();
    }
    for method in METHODS {
        let mut resampled = sum.clone();
        resample(&mut resampled, 30, Some(method), false, &mut Rng::new(1)).unwrap();
        assert!(!resampled.is_empty() && resampled.len() <= 31);
        for (string, coeff) in resampled.sorted_terms() {
            let before = sum.coeff(&string);
            assert!((coeff / coeff.abs() - before / before.abs()).norm() < 1e-12);
        }
    }
    let mut resampled = sum.clone();
    resample(&mut resampled, 30, None, true, &mut Rng::new(1)).unwrap();
    for (string, coeff) in resampled.sorted_terms() {
        let before = sum.coeff(&string);
        assert!((coeff / coeff.abs() - (before / before.abs()).powi(2)).norm() < 1e-12);
    }
}

#[test]
fn requests_that_cannot_be_met_are_rejected() {
    let mut rng = StdRng::seed_from_u64(46);
    let sum = random_sum(&mut rng, 3, 10);
    let mut generator = Rng::new(0);
    for method in METHODS {
        assert!(resample(&mut sum.clone(), 0, Some(method), false, &mut generator).is_err());
        let more = resample(&mut sum.clone(), 11, Some(method), false, &mut generator);
        assert_eq!(
            more.is_ok(),
            method == Resampling::Multinomial,
            "{method:?}"
        );
        assert!(resample(&mut sum.clone(), 10, Some(method), false, &mut generator).is_ok());
    }
    let semi = Some(Resampling::SemiDeterministic);
    assert!(resample(&mut sum.clone(), 5, semi, true, &mut generator).is_err());

    let noisy = [Gate::rx(0.3, 0), Gate::depolarizing(0.1, 0).unwrap()];
    let mut terms = sum.sorted_terms();
    let heisenberg = Picture::Heisenberg;
    assert!(sample_paths(&noisy, 3, &mut terms, heisenberg, false, &mut generator).is_err());
    let limits = PathLimits::default();
    let z = PauliString::<1>::from_ops(&[(0, PauliOp::Z)]);
    let mse = |circuit: &[Gate], samples, radii: &[f64]| {
        let state = State::Zero;
        estimate_mse(
            circuit,
            3,
            &z,
            1.0,
            samples,
            radii,
            &limits,
            None,
            &state,
            heisenberg,
            &mut Rng::new(0),
        )
    };
    assert!(mse(&noisy, 10, &[1.0]).is_err());
    // A transfer map is not a Clifford gate, also where it changes nothing.
    let mut unchanged = vec![0.0; 16];
    for index in 0..4 {
        unchanged[index * 4 + index] = 1.0;
    }
    let mapped = [
        Gate::rx(0.3, 0),
        Gate::transfer_map(&[0], &unchanged).unwrap(),
    ];
    assert!(sample_paths(&mapped, 3, &mut terms, heisenberg, false, &mut generator).is_err());
    assert!(sample_sum(&mapped, &sum, 10, heisenberg, false, &mut generator).is_err());
    assert!(sample_sum(
        &[Gate::rx(0.3, 0)],
        &sum,
        0,
        heisenberg,
        false,
        &mut generator
    )
    .is_err());
    assert!(mse(&mapped, 10, &[1.0]).is_err());
    assert!(mse(&[Gate::rx(0.3, 0)], 10, &[]).is_err());
    assert!(mse(&[Gate::rx(0.3, 0)], 0, &[1.0]).is_err());
    assert!(mse(&[Gate::rx(0.3, 0)], 10, &[1.0]).is_ok());

    let counted = Truncation {
        max_freq: Some(2),
        ..Truncation::none()
    };
    let run = |truncation: &Truncation, max_size, resampling_size| {
        let mut sum = sum.clone();
        let circuit = [Gate::rx(0.3, 0)];
        propagate_resampling(
            &circuit,
            &mut sum,
            truncation,
            heisenberg,
            None,
            max_size,
            resampling_size,
            None,
            false,
            &mut Rng::new(0),
        )
    };
    assert!(run(&counted, 10, 5).is_err());
    assert!(run(&Truncation::none(), 10, 0).is_err());
    assert!(run(&Truncation::none(), 10, 11).is_err());
    assert!(run(&Truncation::none(), 10, 5).is_ok());
}

/// A circuit of rotations and Clifford gates in which rotations split most terms.
fn mixed_circuit(rng: &mut StdRng, n: usize, gates: usize) -> Vec<Gate> {
    (0..gates)
        .map(|_| loop {
            let gate = random_gate(rng, n);
            if !matches!(gate, Gate::X(_) | Gate::Y(_) | Gate::Z(_)) {
                break gate;
            }
        })
        .collect()
}

#[test]
fn sampled_paths_average_to_the_propagated_sum() {
    let mut rng = StdRng::seed_from_u64(47);
    for round in 0..6 {
        let n = 3;
        let circuit = mixed_circuit(&mut rng, n, 10);
        let start = random_sum(&mut rng, n, 3);
        let picture = if round % 2 == 0 {
            Picture::Heisenberg
        } else {
            Picture::Schrodinger
        };
        let mut exact = start.clone();
        propagate_with(&circuit, &mut exact, &Truncation::none(), picture, None).unwrap();

        let samples = 40_000;
        let mut generator = Rng::new(round);
        let mut means: BTreeMap<PauliString<1>, Mean> = BTreeMap::new();
        let mut spread = 0usize;
        for _ in 0..samples {
            let mut terms = start.sorted_terms();
            sample_paths(&circuit, n, &mut terms, picture, false, &mut generator).unwrap();
            let mut sampled = Terms::new();
            for (string, coeff) in terms {
                *sampled.entry(string).or_insert(0.0) += coeff;
            }
            for string in sampled.keys() {
                means.entry(*string).or_default();
            }
            for (string, mean) in &mut means {
                mean.add(sampled.get(string).copied().unwrap_or(0.0));
            }
            spread = spread.max(sampled.len());
        }
        assert!(means.len() > 6, "round {round}: the paths have to branch");
        for (string, mean) in &means {
            // A string that was first reached late missed the zeros before it.
            let seen = mean.count / samples as f64;
            let expected = exact.coeff(string);
            let value = mean.value() * seen;
            assert!(
                (value - expected).abs() <= 6.0 * mean.error() + 0.02 * (1.0 - seen) + 1e-12,
                "round {round}: {} is {expected}, sampled {value} +- {}",
                string.label(n),
                mean.error()
            );
        }
        // No Pauli string that the propagation does not reach is sampled.
        assert!(means
            .keys()
            .all(|string| exact.coeff(string) != 0.0 || means[string].value().abs() < 0.05));
    }
}

/// Every path of `string` through rotations given in the order they act, with the
/// probability that sampling by squared factors gives it.
fn squared_paths(
    rotations: &[(PauliString<1>, f64)],
    string: PauliString<1>,
) -> Vec<(PauliString<1>, f64)> {
    let mut paths = vec![(string, 1.0)];
    for (generator, theta) in rotations {
        let mut next = Vec::new();
        for (path, probability) in paths {
            if path.commutes(generator) {
                next.push((path, probability));
            } else {
                next.push((path, probability * theta.cos().powi(2)));
                next.push((generator.xor(&path), probability * theta.sin().powi(2)));
            }
        }
        paths = next;
    }
    paths
}

#[test]
fn paths_sampled_by_squares_keep_the_coefficient_and_follow_the_squared_factors() {
    let mut rng = StdRng::seed_from_u64(48);
    let n = 3;
    let circuit: Vec<Gate> = (0..7)
        .map(|_| {
            let ops: Vec<(usize, PauliOp)> =
                (0..n).map(|q| (q, OPS[rng.gen_range(0..4)])).collect();
            Gate::rotation(&ops, rng.gen_range(-3.0..3.0))
        })
        .collect();
    let in_order: Vec<(PauliString<1>, f64)> = circuit
        .iter()
        .rev()
        .map(|gate| match gate {
            Gate::Rotation { generator, theta } => (PauliString::from_ops(generator), *theta),
            _ => unreachable!(),
        })
        .collect();
    let start = PauliString::<1>::from_ops(&[(0, PauliOp::Z), (2, PauliOp::X)]);
    let mut expected = Terms::new();
    for (string, probability) in squared_paths(&in_order, start) {
        *expected.entry(string).or_insert(0.0) += probability;
    }
    assert!(expected.len() > 4);

    let samples = 100_000;
    let mut terms = vec![(start, -0.7); samples];
    sample_paths(
        &circuit,
        n,
        &mut terms,
        Picture::Heisenberg,
        true,
        &mut Rng::new(3),
    )
    .unwrap();
    let mut counts = Terms::new();
    for (string, coeff) in terms {
        assert!(
            (coeff + 0.7).abs() < 1e-12,
            "the coefficient is kept, with its sign"
        );
        *counts.entry(string).or_insert(0.0) += 1.0;
    }
    assert!(counts.keys().all(|string| expected.contains_key(string)));
    for (string, probability) in expected {
        let share = counts.get(&string).copied().unwrap_or(0.0) / samples as f64;
        let error = (probability * (1.0 - probability) / samples as f64).sqrt();
        assert!(
            (share - probability).abs() <= 6.0 * error + 1e-12,
            "{}",
            string.label(n)
        );
    }
}

#[test]
fn clifford_gates_change_no_sign_of_paths_sampled_by_squares() {
    let mut rng = StdRng::seed_from_u64(54);
    let n = 4;
    let circuit: Vec<Gate> = (0..40)
        .map(|_| loop {
            let gate = random_gate(&mut rng, n);
            if !matches!(gate, Gate::Rotation { .. }) {
                break gate;
            }
        })
        .collect();
    let sum = random_sum(&mut rng, n, 30);
    for picture in [Picture::Heisenberg, Picture::Schrodinger] {
        let mut exact = sum.clone();
        propagate_with(&circuit, &mut exact, &Truncation::none(), picture, None).unwrap();
        let start = sum.sorted_terms();

        // Without squares a path through Clifford gates is the propagation of its term.
        let mut plain = start.clone();
        sample_paths(&circuit, n, &mut plain, picture, false, &mut Rng::new(1)).unwrap();
        let mut flipped = 0;
        for ((string, coeff), (_, before)) in plain.iter().zip(&start) {
            assert_eq!(*coeff, exact.coeff(string));
            assert_eq!(coeff.abs(), before.abs());
            flipped += usize::from(coeff != before);
        }
        assert!(flipped > 5, "{flipped} signs changed");

        // With squares it ends on the same string with the coefficient it started with.
        let mut squared = start.clone();
        sample_paths(&circuit, n, &mut squared, picture, true, &mut Rng::new(1)).unwrap();
        for (((string, coeff), (_, before)), (expected, _)) in
            squared.iter().zip(&start).zip(&plain)
        {
            assert_eq!(string, expected);
            assert_eq!(coeff, before);
        }
    }
}

#[test]
fn the_mean_of_paths_is_taken_over_all_samples() {
    // More samples than are held at a time: through Clifford gates every path is the
    // propagated term, and so is their mean.
    let n = 2;
    let circuit = [Gate::H(0), Gate::Cx(0, 1), Gate::S(1), Gate::X(0)];
    let mut sum = Sum::new(n).unwrap();
    sum.add(&[(0, PauliOp::Z), (1, PauliOp::Y)], 0.8).unwrap();
    sum.add(&[(1, PauliOp::X)], -0.3).unwrap();
    let mut exact = sum.clone();
    let heisenberg = Picture::Heisenberg;
    propagate_with(&circuit, &mut exact, &Truncation::none(), heisenberg, None).unwrap();
    for samples in [1, 1000, 65_536, 65_537, 150_000] {
        let mean =
            sample_sum(&circuit, &sum, samples, heisenberg, false, &mut Rng::new(2)).unwrap();
        assert_eq!(mean.len(), exact.len());
        for (string, coeff) in exact.sorted_terms() {
            assert!(
                (mean.coeff(&string) - coeff).abs() < 1e-9,
                "{samples} samples: {} for {coeff}",
                mean.coeff(&string)
            );
        }
    }

    // And through a rotation, the mean of that many paths is close to the propagated sum.
    let circuit = [Gate::rx(0.9, 0), Gate::rzz(0.5, 0, 1)];
    let mut exact = sum.clone();
    propagate_with(&circuit, &mut exact, &Truncation::none(), heisenberg, None).unwrap();
    let samples = 150_000;
    let mean = sample_sum(&circuit, &sum, samples, heisenberg, false, &mut Rng::new(3)).unwrap();
    for (string, coeff) in exact.sorted_terms() {
        // A path carries at most (|cos| + |sin|)^2 < 2 times the size of its term.
        assert!(
            (mean.coeff(&string) - coeff).abs() < 6.0 * 2.0 / (samples as f64).sqrt(),
            "{} for {coeff}",
            mean.coeff(&string)
        );
    }
}

fn trotter_circuit(rng: &mut StdRng, n: usize, layers: usize) -> Vec<Gate> {
    let mut circuit = Vec::new();
    for _ in 0..layers {
        circuit.extend((0..n - 1).map(|q| Gate::rzz(rng.gen_range(0.2..0.9), q, q + 1)));
        circuit.extend((0..n).map(|q| Gate::rx(rng.gen_range(0.2..0.9), q)));
    }
    circuit
}

#[test]
fn a_resampling_propagation_that_never_resamples_is_the_propagation() {
    let mut rng = StdRng::seed_from_u64(49);
    let n = 6;
    let circuit = trotter_circuit(&mut rng, n, 3);
    let truncation = Truncation {
        max_weight: Some(4),
        min_abs_coeff: 1e-6,
        ..Truncation::none()
    };
    for picture in [Picture::Heisenberg, Picture::Schrodinger] {
        let start = random_sum(&mut rng, n, 3);
        let mut expected = start.clone();
        propagate_with(&circuit, &mut expected, &truncation, picture, None).unwrap();
        let mut sampled = start.clone();
        let mut generator = Rng::new(0);
        propagate_resampling(
            &circuit,
            &mut sampled,
            &truncation,
            picture,
            None,
            usize::MAX,
            1,
            None,
            false,
            &mut generator,
        )
        .unwrap();
        assert_eq!(sampled.sorted_terms(), expected.sorted_terms());
    }
}

#[test]
fn a_resampling_propagation_is_its_steps_written_out() {
    // Gate by gate: the propagation through the gate, and a resampling of the sum once it
    // is larger than the largest size, with random numbers from one generator.
    let mut rng = StdRng::seed_from_u64(55);
    let n = 6;
    let circuit = trotter_circuit(&mut rng, n, 5);
    let truncation = Truncation {
        max_weight: Some(4),
        min_abs_coeff: 1e-6,
        ..Truncation::none()
    };
    let cases = [
        (None, false),
        (None, true),
        (Some(Resampling::Multinomial), false),
        (Some(Resampling::Multinomial), true),
        (Some(Resampling::systematic()), false),
        (Some(Resampling::systematic()), true),
        (Some(Resampling::SemiDeterministic), false),
    ];
    for picture in [Picture::Heisenberg, Picture::Schrodinger] {
        let start = random_sum(&mut rng, n, 3);
        for (method, squared) in cases {
            let (max_size, resampling_size) = (30, 20);
            let mut expected = start.clone();
            let mut generator = Rng::new(11);
            let mut resampled = 0;
            let in_order: Vec<&Gate> = match picture {
                Picture::Heisenberg => circuit.iter().rev().collect(),
                Picture::Schrodinger => circuit.iter().collect(),
            };
            for gate in in_order {
                let gate = std::slice::from_ref(gate);
                propagate_with(gate, &mut expected, &truncation, picture, None).unwrap();
                if expected.len() > max_size {
                    let size = resampling_size;
                    resample(&mut expected, size, method, squared, &mut generator).unwrap();
                    resampled += 1;
                }
            }
            assert!(resampled >= 2, "resampled {resampled} times");

            let mut sum = start.clone();
            propagate_resampling(
                &circuit,
                &mut sum,
                &truncation,
                picture,
                None,
                max_size,
                resampling_size,
                method,
                squared,
                &mut Rng::new(11),
            )
            .unwrap();
            assert_eq!(
                sum.sorted_terms(),
                expected.sorted_terms(),
                "{method:?}, squared {squared}"
            );
        }
    }
}

#[test]
fn a_resampling_propagation_is_unbiased_and_stays_small() {
    let mut rng = StdRng::seed_from_u64(50);
    let n = 8;
    let circuit = trotter_circuit(&mut rng, n, 4);
    let mut start = Sum::new(n).unwrap();
    start.add(&[(2, PauliOp::Z)], 1.0).unwrap();
    start.add(&[(3, PauliOp::Z), (4, PauliOp::Z)], 0.5).unwrap();
    let mut exact = start.clone();
    propagate_with(
        &circuit,
        &mut exact,
        &Truncation::none(),
        Picture::Heisenberg,
        None,
    )
    .unwrap();
    assert!(exact.len() > 1000, "{} terms", exact.len());

    for (method, max_size) in [
        (None, 100),
        (Some(Resampling::systematic()), 60),
        (Some(Resampling::Multinomial), 200),
    ] {
        let mut value = Mean::default();
        for seed in 0..1500 {
            let mut sum = start.clone();
            let mut generator = Rng::new(seed);
            propagate_resampling(
                &circuit,
                &mut sum,
                &Truncation::none(),
                Picture::Heisenberg,
                None,
                max_size,
                max_size / 2,
                method,
                false,
                &mut generator,
            )
            .unwrap();
            // A gate at most doubles the sum, which is resampled once it is above the size.
            assert!(sum.len() <= 2 * max_size);
            value.add(sum.overlap_with_zero());
        }
        assert!(value.error() > 1e-4, "the sum was resampled");
        assert!(
            value.agrees_with(exact.overlap_with_zero()),
            "{method:?}: {} +- {}, exact {}",
            value.value(),
            value.error(),
            exact.overlap_with_zero()
        );
    }
}

/// Every path of `string` through the gates of a circuit of rotations, in the order they
/// act: the Pauli string it ends on, its probability when every rotation that splits it
/// takes the sine branch with `probability`, and whether `dropped` holds anywhere on it.
fn paths_with_limits(
    rotations: &[PauliString<1>],
    string: PauliString<1>,
    probability: f64,
    dropped: &dyn Fn(&PauliString<1>, u32, u32) -> bool,
) -> Vec<(PauliString<1>, f64, bool)> {
    let mut paths = vec![(string, 1.0, 0u32, 0u32, false)];
    for generator in rotations {
        let mut next = Vec::new();
        for (path, weight, freq, sins, dead) in paths {
            if path.commutes(generator) {
                next.push((path, weight, freq, sins, dead || dropped(&path, freq, sins)));
            } else {
                let sine = generator.xor(&path);
                next.push((
                    path,
                    weight * (1.0 - probability),
                    freq + 1,
                    sins,
                    dead || dropped(&path, freq + 1, sins),
                ));
                next.push((
                    sine,
                    weight * probability,
                    freq + 1,
                    sins + 1,
                    dead || dropped(&sine, freq + 1, sins + 1),
                ));
            }
        }
        paths = next;
    }
    paths
        .into_iter()
        .map(|(path, weight, _, _, dead)| (path, weight, dead))
        .collect()
}

#[test]
fn the_estimated_error_is_the_weight_of_the_dropped_paths() {
    let mut rng = StdRng::seed_from_u64(51);
    let n = 4;
    let circuit = trotter_circuit(&mut rng, n, 2);
    let generators: Vec<PauliString<1>> = circuit
        .iter()
        .rev()
        .map(|gate| match gate {
            Gate::Rotation { generator, .. } => PauliString::from_ops(generator),
            _ => unreachable!(),
        })
        .collect();
    let start = PauliString::<1>::from_ops(&[(1, PauliOp::Z)]);
    let radius = 1.1f64;
    let probability = 0.5 * (1.0 - (2.0 * radius).sin() / (2.0 * radius));
    let samples = 200_000;

    type Dropped<'a> = &'a dyn Fn(&PauliString<1>, u32, u32) -> bool;
    let cases: [(PathLimits, Dropped); 4] = [
        (
            PathLimits {
                max_weight: Some(2),
                ..PathLimits::default()
            },
            &|p, _, _| p.weight() > 2,
        ),
        (
            PathLimits {
                max_freq: Some(5),
                ..PathLimits::default()
            },
            &|_, freq, _| freq > 5,
        ),
        (
            PathLimits {
                max_sins: Some(1),
                ..PathLimits::default()
            },
            &|_, _, sins| sins > 1,
        ),
        (
            PathLimits {
                max_weight: Some(3),
                max_sins: Some(2),
                max_freq: Some(7),
            },
            &|p, freq, sins| p.weight() > 3 || freq > 7 || sins > 2,
        ),
    ];
    for (limits, dropped) in cases {
        let expected: f64 = paths_with_limits(&generators, start, probability, dropped)
            .iter()
            .filter(|(path, _, dead)| *dead && !path.has_x_or_y())
            .map(|(_, weight, _)| 0.25 * weight)
            .sum();
        assert!(expected > 1e-3, "{limits:?} has to drop paths that count");
        let radii = vec![radius; generators.len()];
        let estimate = estimate_mse(
            &circuit,
            n,
            &start,
            0.5,
            samples,
            &radii,
            &limits,
            None,
            &State::Zero,
            Picture::Heisenberg,
            &mut Rng::new(9),
        )
        .unwrap();
        let error = 0.25 * (4.0 * expected * (1.0 - 4.0 * expected) / samples as f64).sqrt();
        assert!(
            (estimate - expected).abs() <= 6.0 * error,
            "{limits:?}: {estimate}, expected {expected}"
        );
    }

    // In a state in which Pauli strings have the value -1, the squares of the values are
    // added up: the estimate is that of the state of zeros, path for path.
    let limits = PathLimits {
        max_sins: Some(1),
        ..PathLimits::default()
    };
    let ones = [1usize, 2];
    let dropped = paths_with_limits(&generators, start, probability, &|_, _, sins| sins > 1);
    let value = |path: &PauliString<1>| State::Computational(&ones).expectation_of(path, n);
    let signed: f64 = dropped
        .iter()
        .filter(|(_, _, dead)| *dead)
        .map(|(path, weight, _)| weight * value(path))
        .sum();
    let squares: f64 = dropped
        .iter()
        .filter(|(_, _, dead)| *dead)
        .map(|(path, weight, _)| weight * value(path).powi(2))
        .sum();
    assert!((signed - squares).abs() > 0.02, "{signed} and {squares}");
    let radii = vec![radius; generators.len()];
    let in_state = |state: &State<'_, 1>| {
        estimate_mse(
            &circuit,
            n,
            &start,
            1.0,
            samples,
            &radii,
            &limits,
            None,
            state,
            Picture::Heisenberg,
            &mut Rng::new(12),
        )
        .unwrap()
    };
    let estimate = in_state(&State::Computational(&ones));
    assert_eq!(estimate, in_state(&State::Zero));
    assert!((estimate - squares).abs() < 6.0 * (squares * (1.0 - squares) / samples as f64).sqrt());

    // Limits that drop nothing leave no error, and a custom rule that drops everything
    // leaves the squared weight of all the paths that count.
    let none = PathLimits {
        max_weight: Some(4),
        max_freq: Some(100),
        max_sins: Some(100),
    };
    let radii = vec![radius; generators.len()];
    let run = |limits: &PathLimits, custom: Option<&(dyn Fn(&PauliString<1>) -> bool + Sync)>| {
        estimate_mse(
            &circuit,
            n,
            &start,
            1.0,
            samples,
            &radii,
            limits,
            custom,
            &State::Zero,
            Picture::Heisenberg,
            &mut Rng::new(10),
        )
        .unwrap()
    };
    assert_eq!(run(&none, None), 0.0);
    let all: f64 = paths_with_limits(&generators, start, probability, &|_, _, _| true)
        .iter()
        .filter(|(path, _, _)| !path.has_x_or_y())
        .map(|(_, weight, _)| weight)
        .sum();
    let everything = run(&PathLimits::default(), Some(&|_| true));
    assert!((everything - all).abs() < 6.0 * (all * (1.0 - all) / samples as f64).sqrt());
}

#[test]
fn the_estimated_error_is_the_mean_squared_error_of_the_truncated_propagation() {
    // With angles drawn from the whole circle, the error of a weight truncation, averaged
    // over the angles, is what the estimate is of.
    let mut rng = StdRng::seed_from_u64(52);
    let n = 5;
    let mut observable = Sum::new(n).unwrap();
    observable.add(&[(2, PauliOp::Z)], 1.0).unwrap();
    let string = observable.sorted_terms()[0].0;
    let truncation = Truncation {
        max_weight: Some(2),
        ..Truncation::none()
    };
    let shape = trotter_circuit(&mut rng, n, 3);
    let rotations = shape.len();

    let mut error = Mean::default();
    for _ in 0..3000 {
        let circuit: Vec<Gate> = shape
            .iter()
            .map(|gate| match gate {
                Gate::Rotation { generator, .. } => Gate::rotation(
                    generator,
                    rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
                ),
                _ => unreachable!(),
            })
            .collect();
        let (mut exact, mut truncated) = (observable.clone(), observable.clone());
        propagate_with(
            &circuit,
            &mut exact,
            &Truncation::none(),
            Picture::Heisenberg,
            None,
        )
        .unwrap();
        propagate_with(
            &circuit,
            &mut truncated,
            &truncation,
            Picture::Heisenberg,
            None,
        )
        .unwrap();
        error.add((exact.overlap_with_zero() - truncated.overlap_with_zero()).powi(2));
    }

    let limits = PathLimits {
        max_weight: Some(2),
        ..PathLimits::default()
    };
    let radii = vec![std::f64::consts::PI; rotations];
    let samples = 400_000;
    let estimate = estimate_mse(
        &shape,
        n,
        &string,
        1.0,
        samples,
        &radii,
        &limits,
        None,
        &State::Zero,
        Picture::Heisenberg,
        &mut Rng::new(11),
    )
    .unwrap();
    let sampling = (estimate * (1.0 - estimate) / samples as f64).sqrt();
    assert!(estimate > 1e-3);
    assert!(
        (estimate - error.value()).abs() <= 6.0 * (error.error() + sampling),
        "estimate {estimate}, measured {} +- {}",
        error.value(),
        error.error()
    );
}

#[test]
fn coefficients_of_any_kind_can_be_sampled() {
    // The complex sum (1 + 2i) O is sampled like O.
    let mut rng = StdRng::seed_from_u64(53);
    let n = 3;
    let circuit = mixed_circuit(&mut rng, n, 8);
    let string = PauliString::<1>::from_ops(&[(1, PauliOp::Y)]);
    let factor = Complex64::new(1.0, 2.0);
    let mut real = vec![(string, 1.0); 500];
    let mut complex = vec![(string, factor); 500];
    sample_paths(
        &circuit,
        n,
        &mut real,
        Picture::Heisenberg,
        false,
        &mut Rng::new(5),
    )
    .unwrap();
    sample_paths(
        &circuit,
        n,
        &mut complex,
        Picture::Heisenberg,
        false,
        &mut Rng::new(5),
    )
    .unwrap();
    for ((a, x), (b, y)) in real.iter().zip(&complex) {
        assert_eq!(a, b);
        assert!((factor.scale(*x) - y).norm() < 1e-12);
    }
}
