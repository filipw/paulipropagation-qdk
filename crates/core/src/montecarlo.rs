//! Propagation by random sampling: resampling a sum down to fewer terms, sampling one path
//! per term through a circuit, and an estimate of the error of a truncation.
//!
//! The methods and their conventions are those of PauliPropagation.jl.

use crate::gradient::State;
use crate::{
    propagate_in_zones, Coefficient, CustomTruncation, Error, Gate, PauliString, PauliSum, Picture,
    Truncation, Zones,
};

/// A generator of random numbers, xoshiro256** seeded through splitmix64. The same seed
/// gives the same numbers on every machine.
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut mixer = seed;
        let mut next = || {
            mixer = mixer.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = mixer;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }

    /// A generator with a seed that differs from run to run.
    #[must_use]
    pub fn from_entropy() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let hasher = std::collections::hash_map::RandomState::new().build_hasher();
        Self::new(hasher.finish())
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let shifted = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= shifted;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A number in `[0, 1)`, from the upper 53 bits of [`Rng::next_u64`].
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// How [`resample`] chooses the terms that survive.
///
/// Every method lines the terms up, each on an interval as long as its weight, which is the
/// absolute value of its coefficient. It lays teeth on that line and gives every term the
/// weight of the teeth on its interval. Terms without a tooth are dropped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resampling {
    /// As many independent random teeth as the target size. A term can be hit several times,
    /// so fewer terms than the target survive.
    Multinomial,
    /// A comb of evenly spaced teeth at a random offset. With `calibrate`, the spacing is
    /// narrowed until the expected number of surviving terms is at least
    /// `(1 - rtol) * target - atol`.
    Systematic {
        calibrate: bool,
        rtol: f64,
        atol: f64,
    },
    /// The terms whose weight exceeds the total weight divided by the target size are kept
    /// as they are, and a comb over the others fills the rest of the target size.
    SemiDeterministic,
}

impl Resampling {
    /// The comb with the calibration of PauliPropagation.jl.
    #[must_use]
    pub fn systematic() -> Self {
        Resampling::Systematic {
            calibrate: true,
            rtol: 0.01,
            atol: 1.0,
        }
    }
}

/// A function that counts the teeth of a comb below a position.
fn comb(spacing: f64, rng: &mut Rng) -> impl Fn(f64) -> f64 {
    let teeth_per_weight = if spacing == 0.0 { 0.0 } else { 1.0 / spacing };
    let offset = rng.uniform();
    move |position| (position * teeth_per_weight - offset).floor()
}

/// The spacing of a comb that keeps close to `target` terms: a comb keeps a term of weight
/// `w` with probability `min(1, w / spacing)`.
fn calibrated_spacing(weights: &[f64], total: f64, target: f64, rtol: f64, atol: f64) -> f64 {
    let enough = (1.0 - rtol) * target - atol;
    let mut spacing = total / target;
    for _ in 0..10 {
        let heavy = weights.iter().filter(|&&w| w >= spacing).count() as f64;
        let light: f64 = weights.iter().filter(|&&w| w < spacing).sum();
        if heavy + light / spacing >= enough || light == 0.0 {
            return spacing;
        }
        spacing = light / (target - heavy);
    }
    spacing
}

/// Reduces `sum` to about `target_size` terms by random sampling.
///
/// Unless `squared` is set, the result is an unbiased estimate of the sum: the expected
/// value of every coefficient is the coefficient it had. With `squared`, the weight of a
/// term is the squared absolute value of its coefficient, the terms that survive get the
/// square root of the weight of their teeth, and the result is not unbiased.
///
/// `method` defaults to [`Resampling::SemiDeterministic`], and to
/// [`Resampling::Multinomial`] with `squared`. Only the multinomial method can be asked for
/// more terms than the sum has, and the semi-deterministic method does not take `squared`.
///
/// The random numbers are taken from `rng` in a fixed order: one for the offset of a comb,
/// and as many as the target size for the multinomial method.
pub fn resample<const W: usize, C: Coefficient>(
    sum: &mut PauliSum<W, C>,
    target_size: usize,
    method: Option<Resampling>,
    squared: bool,
    rng: &mut Rng,
) -> Result<(), Error> {
    let method = method.unwrap_or(if squared {
        Resampling::Multinomial
    } else {
        Resampling::SemiDeterministic
    });
    if target_size == 0 {
        return Err(Error::Invalid("the target size must be positive"));
    }
    if method != Resampling::Multinomial && target_size > sum.len() {
        return Err(Error::Invalid(
            "only multinomial resampling can be asked for more terms than the sum has",
        ));
    }
    if method == Resampling::SemiDeterministic && squared {
        return Err(Error::Invalid(
            "semi-deterministic resampling does not sample by squared coefficients",
        ));
    }

    let (strings, coeffs) = sum.take_ordered();
    let weight = |c: C| if squared { c.norm_sqr() } else { c.abs() };
    // The coefficient of a term with `teeth` teeth that weigh `per_tooth` each.
    let reweighed = |c: C, teeth: f64, per_tooth: f64| {
        let mass = teeth * per_tooth;
        if squared {
            c.times(c).with_abs(mass.sqrt())
        } else {
            c.with_abs(mass)
        }
    };
    let target = target_size as f64;

    let mut weights: Vec<f64> = coeffs.iter().map(|&c| weight(c)).collect();
    let mut kept_as_it_is = vec![false; coeffs.len()];
    let per_tooth;
    let teeth_below: Box<dyn Fn(f64) -> f64> = match method {
        Resampling::Multinomial => {
            let total: f64 = weights.iter().sum();
            per_tooth = total / target;
            let mut draws: Vec<f64> = (0..target_size).map(|_| rng.uniform() * total).collect();
            draws.sort_unstable_by(f64::total_cmp);
            Box::new(move |position| draws.partition_point(|&draw| draw < position) as f64)
        }
        Resampling::Systematic {
            calibrate,
            rtol,
            atol,
        } => {
            let total: f64 = weights.iter().sum();
            per_tooth = if calibrate {
                calibrated_spacing(&weights, total, target, rtol, atol)
            } else {
                total / target
            };
            Box::new(comb(per_tooth, rng))
        }
        Resampling::SemiDeterministic => {
            let total: f64 = weights.iter().sum();
            let keep_above = total / target;
            let mut kept = 0usize;
            for (weight, as_it_is) in weights.iter_mut().zip(&mut kept_as_it_is) {
                if *weight > keep_above {
                    *as_it_is = true;
                    *weight = 0.0;
                    kept += 1;
                }
            }
            // The comb runs over the other terms only.
            let teeth = target_size - kept;
            per_tooth = if teeth > 0 {
                weights.iter().sum::<f64>() / teeth as f64
            } else {
                0.0
            };
            Box::new(comb(per_tooth, rng))
        }
    };

    let mut result = (Vec::new(), Vec::new());
    let mut end = 0.0;
    let mut below_end = teeth_below(end);
    for (index, (string, coeff)) in strings.into_iter().zip(coeffs).enumerate() {
        let below_start = below_end;
        end += weights[index];
        below_end = teeth_below(end);
        let teeth = below_end - below_start;
        let coeff = if kept_as_it_is[index] {
            coeff
        } else if teeth > 0.0 {
            reweighed(coeff, teeth, per_tooth)
        } else {
            continue;
        };
        if coeff != C::ZERO {
            result.0.push(string);
            result.1.push(coeff);
        }
    }
    sum.set_ordered(result.0, result.1);
    Ok(())
}

/// Propagates like [`crate::propagate_with`], and whenever a gate leaves the sum with more than
/// `max_size` terms, resamples it down to about `resampling_size` terms with [`resample`].
///
/// The sum then never holds more than twice `max_size` terms, whatever the circuit, and
/// without `squared` the result is an unbiased estimate of the propagation with the same
/// truncation. `max_freq` and `max_sins` are not supported. The gates are applied one at a
/// time on the thread of the caller.
#[allow(clippy::too_many_arguments)]
pub fn propagate_resampling<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &mut PauliSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    custom: Option<&CustomTruncation<'_, W, C>>,
    max_size: usize,
    resampling_size: usize,
    method: Option<Resampling>,
    squared: bool,
    rng: &mut Rng,
) -> Result<(), Error> {
    if truncation.max_freq.is_some() || truncation.max_sins.is_some() {
        return Err(Error::Invalid(
            "a resampling propagation cannot be truncated by max_freq or max_sins",
        ));
    }
    if resampling_size == 0 || resampling_size > max_size {
        return Err(Error::Invalid(
            "the resampling size must be positive and at most the largest size",
        ));
    }
    for gate in circuit {
        gate.validate(sum.num_qubits())?;
    }
    let mut step = |gate: &Gate, sum: &mut PauliSum<W, C>| {
        let gate = std::slice::from_ref(gate);
        propagate_in_zones(gate, sum, truncation, picture, custom, &Zones::ONE)?;
        if sum.len() > max_size {
            resample(sum, resampling_size, method, squared, rng)?;
        }
        Ok(())
    };
    match picture {
        Picture::Heisenberg => circuit.iter().rev().try_for_each(|gate| step(gate, sum)),
        Picture::Schrodinger => circuit.iter().try_for_each(|gate| step(gate, sum)),
    }
}

/// The sign of a number as PauliPropagation.jl takes it: zero for zero.
fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The gates in the order they act on the sum, each as it acts on an observable.
fn in_order_of_action(circuit: &[Gate], picture: Picture) -> Vec<Gate> {
    match picture {
        Picture::Heisenberg => circuit.iter().rev().cloned().collect(),
        Picture::Schrodinger => circuit.iter().map(Gate::adjoint).collect(),
    }
}

fn only_rotations_and_cliffords(circuit: &[Gate], num_qubits: usize) -> Result<(), Error> {
    for gate in circuit {
        gate.validate(num_qubits)?;
        if gate.is_noise() || matches!(gate, Gate::TransferMap { .. }) {
            return Err(Error::Invalid(
                "paths can be sampled only through Clifford gates and Pauli rotations",
            ));
        }
    }
    Ok(())
}

/// Takes every term through the circuit on one random path: where a rotation would split a
/// term in two, one of the two is kept at random.
///
/// The cosine and the sine branch are kept with probabilities proportional to `|cos|` and
/// `|sin|`, and the coefficient is multiplied by `|cos| + |sin|` with the sign of the
/// branch, so that the expected value of the terms is the propagated sum. With `squared`
/// the probabilities are `cos²` and `sin²` and the coefficient keeps its absolute value and
/// loses its signs, so that a Pauli string is reached with the probability that its squared
/// coefficient has in a sum of squares.
///
/// `terms` may repeat a Pauli string: many copies of a term are many samples of its paths.
/// One random number is taken from `rng` for every term at every rotation that splits it,
/// term by term in the order of `terms`. The circuit must consist of Clifford gates and
/// rotations.
pub fn sample_paths<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    num_qubits: usize,
    terms: &mut [(PauliString<W>, C)],
    picture: Picture,
    squared: bool,
    rng: &mut Rng,
) -> Result<(), Error> {
    only_rotations_and_cliffords(circuit, num_qubits)?;
    let gates = in_order_of_action(circuit, picture);
    let power = if squared { 2 } else { 1 };
    // For every rotation: its generator, the probability of the sine branch, and the
    // factors of the cosine and the sine branch.
    let branches: Vec<Option<(PauliString<W>, f64, f64, f64)>> = gates
        .iter()
        .map(|gate| match gate {
            Gate::Rotation { generator, theta } => {
                let (sin, cos) = theta.sin_cos();
                let normalization = sin.abs().powi(power) + cos.abs().powi(power);
                Some((
                    PauliString::from_ops(generator),
                    sin.abs().powi(power) / normalization,
                    normalization * sign(cos).powi(power),
                    normalization * sign(sin).powi(power),
                ))
            }
            _ => None,
        })
        .collect();

    for (string, coeff) in terms.iter_mut() {
        for (gate, branch) in gates.iter().zip(&branches) {
            match branch {
                None => {
                    if gate.conjugate_clifford(string) && !squared {
                        *coeff = coeff.scale(-1.0);
                    }
                }
                Some((generator, sine_probability, cos_factor, sin_factor)) => {
                    if string.commutes(generator) {
                        continue;
                    }
                    if rng.uniform() < *sine_probability {
                        // i G P is -P' or +P', as in the propagation.
                        let (product, k) = generator.mul(string);
                        let product_sign = if k == 3 || squared { 1.0 } else { -1.0 };
                        *string = product;
                        *coeff = coeff.scale(sin_factor * product_sign);
                    } else {
                        *coeff = coeff.scale(*cos_factor);
                    }
                }
            }
        }
    }
    Ok(())
}

/// The mean of `samples` random paths of every term of `sum`, see [`sample_paths`]: an
/// estimate of the propagated sum, or with `squared`, a sum in which the coefficient of a
/// term is spread over the Pauli strings in proportion to their squared factors.
pub fn sample_sum<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &PauliSum<W, C>,
    samples: usize,
    picture: Picture,
    squared: bool,
    rng: &mut Rng,
) -> Result<PauliSum<W, C>, Error> {
    /// The paths that are held at a time.
    const AT_A_TIME: usize = 1 << 16;
    if samples == 0 {
        return Err(Error::Invalid("the number of samples must be positive"));
    }
    let num_qubits = sum.num_qubits();
    let share = 1.0 / samples as f64;
    let mut result = PauliSum::new(num_qubits)?;
    for term in sum.sorted_terms() {
        let mut left = samples;
        while left > 0 {
            let mut paths = vec![term; left.min(AT_A_TIME)];
            left -= paths.len();
            sample_paths(circuit, num_qubits, &mut paths, picture, squared, rng)?;
            for (string, coeff) in paths {
                result.add_string(string, coeff.scale(share));
            }
        }
    }
    Ok(result)
}

/// The rules that [`estimate_mse`] estimates the error of. Each drops a path as the
/// truncation of the same name drops a term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PathLimits {
    pub max_weight: Option<u32>,
    pub max_freq: Option<u32>,
    pub max_sins: Option<u32>,
}

/// An estimate of the mean squared error that a truncation makes in the expectation value
/// of `coeff * string` after `circuit`, averaged over the angles of the rotations.
///
/// The angle of the `k`-th rotation of the circuit is taken as uniformly distributed in
/// `[-radii[k], radii[k]]`, whatever angle the circuit gives it, and the angles as
/// independent. Then the paths of the propagation do not interfere on average, and the mean
/// squared error is the total squared weight of the dropped paths that end on a Pauli
/// string with an expectation value in `state`. That is estimated from `samples` random
/// paths, each of which takes the sine branch of a rotation that splits it with the
/// probability that the squared sine has on average.
///
/// `custom` drops a path where it returns true for its Pauli string. The circuit must
/// consist of Clifford gates and rotations.
#[allow(clippy::too_many_arguments)]
pub fn estimate_mse<const W: usize>(
    circuit: &[Gate],
    num_qubits: usize,
    string: &PauliString<W>,
    coeff: f64,
    samples: usize,
    radii: &[f64],
    limits: &PathLimits,
    custom: Option<&(dyn Fn(&PauliString<W>) -> bool + Sync)>,
    state: &State<'_, W>,
    picture: Picture,
    rng: &mut Rng,
) -> Result<f64, Error> {
    only_rotations_and_cliffords(circuit, num_qubits)?;
    let rotations = circuit
        .iter()
        .filter(|gate| matches!(gate, Gate::Rotation { .. }))
        .count();
    if radii.len() != rotations {
        return Err(Error::Invalid(
            "there must be one range of angles for every rotation of the circuit",
        ));
    }
    if samples == 0 {
        return Err(Error::Invalid("the number of samples must be positive"));
    }

    // The mean of sin² over angles in [-r, r].
    let split_probability = |radius: f64| {
        if radius == 0.0 {
            0.0
        } else {
            0.5 * (1.0 - (2.0 * radius).sin() / (2.0 * radius))
        }
    };
    let mut probabilities: Vec<f64> = radii.iter().map(|&r| split_probability(r)).collect();
    if picture == Picture::Heisenberg {
        probabilities.reverse();
    }
    let gates = in_order_of_action(circuit, picture);
    let mut probabilities = probabilities.into_iter();
    let steps: Vec<(&Gate, Option<(PauliString<W>, f64)>)> = gates
        .iter()
        .map(|gate| match gate {
            Gate::Rotation { generator, .. } => {
                let probability = probabilities.next().expect("one for every rotation");
                (gate, Some((PauliString::from_ops(generator), probability)))
            }
            _ => (gate, None),
        })
        .collect();

    let dropped = |string: &PauliString<W>, freq: u32, sins: u32| {
        limits.max_weight.is_some_and(|max| string.weight() > max)
            || limits.max_freq.is_some_and(|max| freq > max)
            || limits.max_sins.is_some_and(|max| sins > max)
            || custom.is_some_and(|custom| custom(string))
    };

    let mut total = 0.0;
    for _ in 0..samples {
        let mut path = *string;
        let (mut freq, mut sins) = (0u32, 0u32);
        let mut alive = true;
        for (gate, rotation) in &steps {
            match rotation {
                None => {
                    gate.conjugate_clifford(&mut path);
                }
                Some((generator, probability)) => {
                    if !path.commutes(generator) {
                        freq += 1;
                        if rng.uniform() < *probability {
                            sins += 1;
                            path = generator.xor(&path);
                        }
                    }
                }
            }
            alive = alive && !dropped(&path, freq, sins);
        }
        if !alive {
            let value = state.expectation_of(&path, num_qubits);
            total += value * value;
        }
    }
    Ok(coeff * coeff * total / samples as f64)
}
