"""Checks the random methods: with a seed, against the same methods written out in Python
on the same random numbers, and by their averages, which are the results without sampling."""

import bisect
import functools
import math
import operator
import random

import pytest
from reference_helpers import anticommute, multiply, pauli_sum

import paulipropagation_qdk as pp

MASK = (1 << 64) - 1


class Xoshiro:
    """xoshiro256** seeded through splitmix64, the generator behind `seed`."""

    def __init__(self, seed: int):
        mixer = seed

        def mix() -> int:
            nonlocal mixer
            mixer = (mixer + 0x9E3779B97F4A7C15) & MASK
            z = mixer
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
            return z ^ (z >> 31)

        self.state = [mix() for _ in range(4)]

    def uniform(self) -> float:
        s = self.state
        rotl = lambda x, k: ((x << k) | (x >> (64 - k))) & MASK  # noqa: E731
        result = (rotl((s[1] * 5) & MASK, 7) * 9) & MASK
        shifted = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= shifted
        s[3] = rotl(s[3], 45)
        return (result >> 11) * 2.0**-53


def total(values) -> float:
    """The sum from left to right, without the compensation of the built-in sum."""
    return functools.reduce(operator.add, values, 0.0)


def comb(spacing: float, rng: Xoshiro):
    teeth_per_weight = 0.0 if spacing == 0.0 else 1.0 / spacing
    offset = rng.uniform()
    return lambda position: math.floor(position * teeth_per_weight - offset)


def resampled_by_hand(terms: list, target: int, method: str, seed: int) -> dict:
    """Resampling as PauliPropagation.jl describes it, on the terms in their order."""
    rng = Xoshiro(seed)
    weights = [abs(coeff) for _, coeff in terms]
    kept = [False] * len(terms)
    if method == "multinomial":
        per_tooth = total(weights) / target
        draws = sorted(rng.uniform() * total(weights) for _ in range(target))
        teeth_below = lambda position: bisect.bisect_left(draws, position)  # noqa: E731
    elif method == "systematic":
        # The spacing is narrowed until about `target` terms survive on average.
        per_tooth = total(weights) / target
        for _ in range(10):
            heavy = sum(w >= per_tooth for w in weights)
            light = total(w for w in weights if w < per_tooth)
            if heavy + light / per_tooth >= 0.99 * target - 1 or light == 0.0:
                break
            per_tooth = light / (target - heavy)
        teeth_below = comb(per_tooth, rng)
    else:
        keep_above = total(weights) / target
        kept = [w > keep_above for w in weights]
        weights = [0.0 if keep else w for keep, w in zip(kept, weights)]
        teeth = target - sum(kept)
        per_tooth = total(weights) / teeth if teeth > 0 else 0.0
        teeth_below = comb(per_tooth, rng)

    result = {}
    end = 0.0
    below_end = teeth_below(end)
    for (text, coeff), weight, keep in zip(terms, weights, kept):
        below_start = below_end
        end += weight
        below_end = teeth_below(end)
        teeth = below_end - below_start
        if keep:
            result[text] = coeff
        elif teeth > 0:
            result[text] = math.copysign(teeth * per_tooth, coeff)
    return result


def random_sum(rng: random.Random, n: int, count: int) -> pp.PauliSum:
    terms = {}
    while len(terms) < count:
        text = "".join(rng.choice("IXYZ") for _ in range(n))
        terms[text] = rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 0)
    return pauli_sum(n, terms)


@pytest.mark.parametrize("method", ["multinomial", "systematic", "semideterministic"])
def test_resampling_with_a_seed_is_the_method_written_out(method):
    rng = random.Random(method)
    for seed in range(40):
        start = random_sum(rng, 6, 200)
        target = rng.randint(1, 150)
        resampled = pp.resample(start, target, method=method, seed=seed)
        assert dict(resampled.terms()) == resampled_by_hand(start.terms(), target, method, seed)
        assert 0 < len(resampled) < len(start)
    default = pp.resample(start, target, seed=5)
    assert default.terms() == pp.resample(start, target, method="semideterministic", seed=5).terms()


def test_resampling_is_unbiased_and_keeps_the_input():
    rng = random.Random(1)
    start = random_sum(rng, 5, 80)
    before = start.terms()
    means = {text: 0.0 for text, _ in before}
    runs = 3000
    for seed in range(runs):
        for text, coeff in pp.resample(start, 25, seed=seed).terms():
            means[text] += coeff / runs
    assert start.terms() == before
    spacing = sum(abs(c) for _, c in before) / 25
    for text, coeff in before:
        # A comb changes a coefficient by less than its spacing.
        assert means[text] == pytest.approx(coeff, abs=6 * spacing / math.sqrt(runs))
    assert pp.resample(start, 25).terms() != pp.resample(start, 25).terms()


def test_resampling_by_squares_keeps_the_sum_of_squares():
    rng = random.Random(2)
    start = random_sum(rng, 5, 80)
    squares = sum(c * c for _, c in start.terms())
    resampled = pp.resample(start, 20, squared=True, seed=3)
    assert sum(c * c for _, c in resampled.terms()) == pytest.approx(squares)
    assert all(c > 0 for _, c in resampled.terms()) and len(resampled) <= 20
    assert len(pp.resample(start, 500, method="multinomial", seed=3)) <= 80


def test_resampling_rejects_what_it_cannot_do():
    start = random_sum(random.Random(3), 4, 10)
    with pytest.raises(ValueError, match="positive"):
        pp.resample(start, 0)
    with pytest.raises(ValueError, match="more terms"):
        pp.resample(start, 11)
    with pytest.raises(ValueError, match="squared"):
        pp.resample(start, 5, method="semideterministic", squared=True)
    with pytest.raises(ValueError, match="method"):
        pp.resample(start, 5, method="comb")


def rotations_circuit(rng: random.Random, n: int, gates: int) -> tuple[pp.Circuit, list]:
    circuit, layout = pp.Circuit(n), []
    for _ in range(gates):
        qubits = rng.sample(range(n), rng.randint(1, min(3, n)))
        paulis = "".join(rng.choice("XYZ") for _ in qubits)
        theta = rng.uniform(-3, 3)
        circuit.rotation(paulis, qubits, theta)
        generator = ["I"] * n
        for pauli, q in zip(paulis, qubits):
            generator[q] = pauli
        layout.append(("".join(generator), theta))
    return circuit, layout


def paths_by_hand(layout: list, terms: list, samples: int, seed: int, squared: bool) -> dict:
    """One random path per sample of every term, last gate first, as the engine draws them."""
    rng = Xoshiro(seed)
    power = 2 if squared else 1
    sign = lambda x: (x > 0) - (x < 0)  # noqa: E731
    result = {}
    for text, coeff in terms:
        for _ in range(samples):
            path, value = text, coeff
            for generator, theta in reversed(layout):
                if not anticommute(generator, path):
                    continue
                sin, cos = math.sin(theta), math.cos(theta)
                normalization = abs(sin) ** power + abs(cos) ** power
                if rng.uniform() < abs(sin) ** power / normalization:
                    # i G P is -P' if G P = i P' and +P' if G P = -i P'.
                    path, phase = multiply(generator, path)
                    product_sign = 1.0 if phase == 3 or squared else -1.0
                    value = value * (normalization * sign(sin) ** power * product_sign)
                else:
                    value = value * (normalization * sign(cos) ** power)
            result[path] = result.get(path, 0.0) + value * (1.0 / samples)
    return result


@pytest.mark.parametrize("squared", [False, True])
def test_sampled_paths_with_a_seed_are_the_paths_written_out(squared):
    rng = random.Random(4)
    for seed in range(10):
        n = rng.randint(2, 5)
        circuit, layout = rotations_circuit(rng, n, 12)
        start = random_sum(rng, n, 3)
        sampled = pp.mcsample(circuit, start, 50, squared=squared, seed=seed)
        expected = paths_by_hand(layout, start.terms(), 50, seed, squared)
        assert dict(sampled.terms()) == pytest.approx(expected, abs=1e-15, rel=1e-13)
        if squared:
            # Every path keeps the coefficient it started with.
            assert sum(c for _, c in sampled.terms()) == pytest.approx(sum(c for _, c in start.terms()))


def mixed_circuit(rng: random.Random, n: int, gates: int) -> pp.Circuit:
    circuit = pp.Circuit(n)
    for _ in range(gates):
        kind = rng.randrange(6)
        a, b = rng.sample(range(n), 2)
        if kind == 0:
            circuit.h(a)
        elif kind == 1:
            circuit.s(a)
        elif kind == 2:
            circuit.cx(a, b)
        elif kind == 3:
            circuit.rzz(rng.uniform(-1, 1), a, b)
        else:
            circuit.rotation(rng.choice("XY"), [a], rng.uniform(-1, 1))
    return circuit


@pytest.mark.parametrize("heisenberg", [True, False])
def test_sampled_paths_average_to_the_propagated_sum(heisenberg):
    rng = random.Random(10)
    n = 4
    circuit = mixed_circuit(rng, n, 14)
    start = pauli_sum(n, {"ZIIX": 1.0, "IYZI": -0.6})
    exact = dict(pp.propagate(circuit, start, min_abs_coeff=0.0, heisenberg=heisenberg).terms())
    assert len(exact) > 20

    # The spread of a coefficient is measured over runs with seeds of their own.
    runs = 60
    sums = {text: [0.0, 0.0] for text in exact}
    for seed in range(runs):
        sampled = dict(pp.mcsample(circuit, start, 5000, heisenberg=heisenberg, seed=seed).terms())
        assert sampled.keys() <= exact.keys()
        for text, moments in sums.items():
            value = sampled.get(text, 0.0)
            moments[0] += value
            moments[1] += value * value
    for text, coeff in exact.items():
        mean = sums[text][0] / runs
        error = math.sqrt(max(sums[text][1] / runs - mean * mean, 0.0) / runs)
        assert error < 0.01
        assert mean == pytest.approx(coeff, abs=6 * error + 1e-12)


def trotter(n: int, layers: int) -> pp.Circuit:
    circuit = pp.Circuit(n)
    for _ in range(layers):
        for q in range(n - 1):
            circuit.rzz(0.35, q, q + 1)
        for q in range(n):
            circuit.rx(0.45, q)
    return circuit


def test_a_resampling_propagation_stays_small_and_is_unbiased():
    n = 12
    circuit = trotter(n, 5)
    observable = pp.PauliSum(n)
    observable.add("Z", [5])
    exact = pp.propagate(circuit, observable, min_abs_coeff=1e-12)
    assert len(exact) > 10_000

    runs = 300
    values = []
    for seed in range(runs):
        sampled = pp.mcpropagate(circuit, observable, max_size=400, min_abs_coeff=1e-12, seed=seed)
        assert len(sampled) <= 800
        values.append(sampled.overlap_with_zero())
    mean = sum(values) / runs
    error = math.sqrt(sum((v - mean) ** 2 for v in values) / runs / runs)
    assert error > 1e-5
    assert mean == pytest.approx(exact.overlap_with_zero(), abs=6 * error)


def test_a_resampling_propagation_with_a_seed_is_reproducible():
    n = 14
    circuit = trotter(n, 6)
    observable = pp.PauliSum(n)
    observable.add("Z", [6])
    options = dict(max_size=3000, resampling_size=1000, max_weight=6, min_abs_coeff=1e-8, seed=11)
    first = pp.mcpropagate(circuit, observable, **options)
    assert 0 < len(first) <= 6000
    assert pp.mcpropagate(circuit, observable, **options).terms() == first.terms()
    assert pp.mcpropagate(circuit, observable, **{**options, "seed": 12}).terms() != first.terms()
    for method in ("multinomial", "systematic"):
        assert len(pp.mcpropagate(circuit, observable, method=method, **options)) <= 6000
    assert len(pp.mcpropagate(circuit, observable, squared=True, **options)) <= 6000

    # A size that is never reached leaves the propagation as it is.
    plain = pp.propagate(circuit, observable, max_weight=3, min_abs_coeff=1e-8)
    unsampled = pp.mcpropagate(circuit, observable, max_size=10**9, max_weight=3, min_abs_coeff=1e-8)
    assert unsampled.terms() == plain.terms()
    state = pp.propagate(circuit, observable, max_weight=3, heisenberg=False)
    assert pp.mcpropagate(circuit, observable, max_size=10**9, max_weight=3, heisenberg=False).terms() == state.terms()
    kept = lambda text, coeff: "Y" in text  # noqa: E731
    ruled = pp.propagate(circuit, observable, custom_truncation=kept, max_weight=3)
    assert pp.mcpropagate(circuit, observable, max_size=10**9, custom_truncation=kept, max_weight=3).terms() == ruled.terms()


def test_the_estimated_error_is_the_error_of_the_truncated_propagation():
    # The error of a weight truncation, averaged over angles from the whole circle.
    rng = random.Random(7)
    n = 6
    shape = trotter(n, 3)
    observable = pp.PauliSum(n)
    observable.add("Z", [2])
    rotations = len(shape.angles)

    errors = []
    for _ in range(1500):
        circuit = shape.with_angles([rng.uniform(-math.pi, math.pi) for _ in range(rotations)])
        exact = pp.expectation(circuit, observable, min_abs_coeff=0.0)
        truncated = pp.expectation(circuit, observable, max_weight=2, min_abs_coeff=0.0)
        errors.append((exact - truncated) ** 2)
    measured = sum(errors) / len(errors)
    spread = math.sqrt(sum((e - measured) ** 2 for e in errors)) / len(errors)

    samples = 300_000
    estimate = pp.estimate_mse(shape, observable, samples, max_weight=2, seed=8)
    sampling = math.sqrt(estimate * (1 - estimate) / samples)
    assert estimate > 1e-3
    assert estimate == pytest.approx(measured, abs=6 * (spread + sampling))

    # The estimate does not use the angles of the circuit, and grows with the truncation.
    assert pp.estimate_mse(shape.with_angles([0.0] * rotations), observable, samples, max_weight=2, seed=8) == estimate
    assert pp.estimate_mse(shape, observable, samples, max_weight=n, seed=8) == 0.0
    assert pp.estimate_mse(shape, observable, samples, max_weight=1, seed=8) > estimate
    # Without a spread of the angles nothing splits, and the observable is kept or dropped whole.
    assert pp.estimate_mse(shape, observable, samples, thetas=0.0, max_weight=1, seed=8) == 0.0
    assert pp.estimate_mse(shape, observable, samples, thetas=0.0, max_weight=0, seed=8) == 1.0
    by_rule = pp.estimate_mse(shape, observable, samples, custom_truncation=lambda text: text.count("I") < n - 2, seed=8)
    assert by_rule == estimate
    scaled = pp.estimate_mse(shape, observable * 3.0, samples, max_weight=2, seed=8)
    assert scaled == pytest.approx(9 * estimate)


def test_an_error_estimate_rejects_what_it_cannot_do():
    circuit = trotter(4, 1)
    observable = pp.PauliSum(4)
    observable.add("Z", [1])
    two_terms = observable + pauli_sum(4, {"XIII": 1.0})
    with pytest.raises(ValueError, match="single Pauli string"):
        pp.estimate_mse(circuit, two_terms, 10)
    with pytest.raises(ValueError, match="real coefficients"):
        pp.estimate_mse(circuit, observable.astype(complex), 10)
    with pytest.raises(ValueError, match="one range of angles"):
        pp.estimate_mse(circuit, observable, 10, thetas=[1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        pp.estimate_mse(circuit, observable, 0)
    with pytest.raises(TypeError):
        pp.estimate_mse(circuit, observable, 10, custom_truncation=3)

    def fails(text):
        raise KeyError(text)

    with pytest.raises(KeyError):
        pp.estimate_mse(circuit, observable, 10, custom_truncation=fails)
    noisy = trotter(4, 1)
    noisy.depolarizing(0.1, 0)
    for run in (
        lambda: pp.estimate_mse(noisy, observable, 10),
        lambda: pp.mcsample(noisy, observable, 10),
        lambda: pp.mcsample(circuit, observable, 0),
        lambda: pp.mcpropagate(circuit, observable, max_size=10, resampling_size=11),
        lambda: pp.mcpropagate(circuit, observable, max_size=10, resampling_size=0),
    ):
        with pytest.raises(ValueError):
            run()
    with pytest.raises(TypeError):
        pp.mcpropagate(circuit, observable, max_size=10, max_freq=3)
    # Noise channels are propagated, and resampled like any other gate.
    assert len(pp.mcpropagate(noisy, observable, max_size=10, seed=1)) <= 20
