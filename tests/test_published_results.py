"""Results stated in the papers on Pauli propagation, which a correct implementation has to
reproduce: exact identities, values of a published experiment, and bounds on the error of a
truncation.

- Kim et al., "Evidence for the utility of quantum computing before fault tolerance", Nature
  618, 500 (2023), and Rudolph et al., "Classical surrogate simulation of quantum systems
  with LOWESA", arXiv:2308.09109: the kicked Ising circuit on 127 qubits.
- Fontana et al., "Classical simulations of noisy variational quantum circuits",
  arXiv:2306.05400.
- "Classically estimating observables of noiseless quantum circuits", arXiv:2409.01706.
- "Efficient quantum-enhanced classical simulation for patches of quantum landscapes",
  arXiv:2411.19896.
- "Simulating quantum circuits with arbitrary local noise using Pauli Propagation",
  arXiv:2501.13101.
- "Leveraging Symmetry Merging in Pauli Propagation", arXiv:2512.12094.
"""

import itertools
import math
import random

import pytest
from reference_helpers import label, pauli_sum, single, weight

import paulipropagation_qdk as pp

# The kicked Ising circuit of Kim et al. on the heavy-hex lattice of 127 qubits.

# The rows of the lattice, and the qubits that join a row to the next at the given places.
ROWS = [range(0, 14), range(18, 33), range(37, 52), range(56, 71), range(75, 90), range(94, 109),
        range(113, 127)]
BRIDGES = [(14, 0, 18), (15, 4, 22), (16, 8, 26), (17, 12, 30),
           (33, 20, 39), (34, 24, 43), (35, 28, 47), (36, 32, 51),
           (52, 37, 56), (53, 41, 60), (54, 45, 64), (55, 49, 68),
           (71, 58, 77), (72, 62, 81), (73, 66, 85), (74, 70, 89),
           (90, 75, 94), (91, 79, 98), (92, 83, 102), (93, 87, 106),
           (109, 96, 114), (110, 100, 118), (111, 104, 122), (112, 108, 126)]  # fmt: skip


def heavy_hex_bonds() -> list[tuple[int, int]]:
    bonds = [(q, q + 1) for row in ROWS for q in row[:-1]]
    bonds += [bond for bridge, above, below in BRIDGES for bond in ((above, bridge), (bridge, below))]
    assert len(bonds) == 144 and len({q for bond in bonds for q in bond}) == 127
    return bonds


def kicked_ising(steps: int, theta_h: float, last_kick: bool = False) -> pp.Circuit:
    """`steps` times a rotation about X by `theta_h` on every qubit and then a rotation about
    ZZ by -pi/2 on every bond; with `last_kick`, the rotations about X once more."""
    circuit = pp.Circuit(127)
    for step in range(steps + last_kick):
        for q in range(127):
            circuit.rx(theta_h, q)
        if step < steps:
            for a, b in heavy_hex_bonds():
                # exp(i pi/4 ZZ), the rotation about ZZ by -pi/2, as Clifford gates.
                circuit.s_adj(a)
                circuit.s_adj(b)
                circuit.cz(a, b)
    return circuit


# The observables of the experiment that can be verified exactly: the Trotter steps, whether
# the kicks follow once more, the observable, and the number of sines in its one path. At
# theta_h = pi/2 the circuit is a Clifford circuit that takes each of them to a product of Z
# operators with a sign, so that its value is 1 or -1. Rudolph et al. find the whole
# expectation value of the last two close to the one path that takes the sine at every kick
# it meets, sin(theta_h)^25 and sin(theta_h)^34.
KICKED_ISING_OBSERVABLES = [
    (5, False, {"X": [13, 29, 31], "Y": [9, 30], "Z": [8, 12, 17, 28, 32]}, 15),
    (5, False, {"X": [37, 41, 52, 56, 57, 58, 62, 79], "Y": [75], "Z": [38, 40, 42, 63, 72, 80, 90, 91]}, 25),
    (5, True, {"X": [37, 41, 52, 56, 57, 58, 62, 79], "Y": [38, 40, 42, 63, 72, 80, 90, 91], "Z": [75]}, 34),
]


@pytest.mark.parametrize(("steps", "last_kick", "qubits_of", "sines"), KICKED_ISING_OBSERVABLES)
def test_the_observables_of_the_kicked_ising_experiment_at_the_clifford_point(
    steps, last_kick, qubits_of, sines
):
    paulis = [(q, pauli) for pauli, qubits in qubits_of.items() for q in qubits]
    observable = single(127, "".join(p for _, p in paulis), [q for q, _ in paulis])
    circuit = kicked_ising(steps, math.pi / 2, last_kick)

    value = pp.expectation(circuit, observable)
    assert abs(value) == pytest.approx(1.0, abs=1e-12, rel=0)
    # A kick the other way changes the sign of every X and Y.
    flips = len(qubits_of["X"]) + len(qubits_of["Y"])
    backwards = pp.expectation(kicked_ising(steps, -math.pi / 2, last_kick), observable)
    assert backwards == pytest.approx((-1) ** flips * value, abs=1e-12, rel=0)
    # Without the kicks nothing takes the state out of the computational basis.
    assert pp.expectation(kicked_ising(steps, 0.0, last_kick), observable) == 0.0

    # All of the value is one path, with the published number of sines.
    assert pp.expectation(circuit, observable, max_sins=sines) == value
    assert pp.expectation(circuit, observable, max_sins=sines - 1) == 0.0
    # Where every cosine is zero, the frequency of that path is its number of sines.
    assert pp.expectation(circuit, observable, max_freq=sines) == value
    assert pp.expectation(circuit, observable, max_freq=sines - 1) == 0.0


def test_the_magnetization_of_the_kicked_ising_experiment_at_its_exact_points():
    observable = single(127, "Z", [62])
    assert pp.expectation(kicked_ising(20, 0.0), observable) == 1.0
    # At the Clifford point the magnetization of the experiment is zero.
    value = pp.expectation(kicked_ising(20, math.pi / 2), observable)
    assert value == pytest.approx(0.0, abs=1e-12, rel=0)


# Fontana et al.: the circuit H Rz(theta)^d H on the state 0 has the expectation value
# cos(d theta) of Z, and all of it is at the frequency d.


@pytest.mark.parametrize("depth", [1, 2, 7, 40])
def test_repeated_rotations_give_the_highest_frequency(depth):
    theta = 0.37
    circuit = pp.Circuit(1)
    circuit.h(0)
    for _ in range(depth):
        circuit.rz(theta, 0)
    circuit.h(0)
    observable = single(1, "Z", [0])

    exact = math.cos(depth * theta)
    assert pp.expectation(circuit, observable) == pytest.approx(exact, abs=1e-12, rel=0)
    kept = pp.expectation(circuit, observable, max_freq=depth)
    assert kept == pytest.approx(exact, abs=1e-12, rel=0)
    assert pp.expectation(circuit, observable, max_freq=depth - 1) == 0.0


# arXiv:2501.13101: amplitude damping of strength q on two qubits takes
# ZZ + IZ + ZI to (1 - q)^2 ZZ + (1 - q^2) (IZ + ZI) + (2 q + q^2) II.


@pytest.mark.parametrize("damping", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_amplitude_damping_of_two_qubits(damping):
    circuit = pp.Circuit(2)
    circuit.amplitude_damping(damping, 0)
    circuit.amplitude_damping(damping, 1)
    observable = pauli_sum(2, {"ZZ": 1.0, "IZ": 1.0, "ZI": 1.0})

    result = dict(pp.propagate(circuit, observable, min_abs_coeff=0.0).terms())
    expected = {
        "ZZ": (1 - damping) ** 2,
        "IZ": 1 - damping**2,
        "ZI": 1 - damping**2,
        "II": 2 * damping + damping**2,
    }
    for text, coeff in expected.items():
        assert result.get(text, 0.0) == pytest.approx(coeff, abs=1e-15, rel=0), text
    assert set(result) <= set(expected)


@pytest.mark.parametrize("strength", [0.0, 0.05, 0.3, 1.0])
def test_depolarizing_noise_damps_by_the_weight(strength):
    # The same paper: depolarizing noise on every qubit takes P to (1 - p)^|P| P.
    n = 9
    circuit = pp.Circuit(n)
    for q in range(n):
        circuit.depolarizing(strength, q)
    rng = random.Random(3)
    terms = {"".join(rng.choice("IXYZ") for _ in range(n)): rng.uniform(-1, 1) for _ in range(40)}
    result = dict(pp.propagate(circuit, pauli_sum(n, terms), min_abs_coeff=0.0).terms())
    for text, coeff in terms.items():
        expected = coeff * (1 - strength) ** weight(text)
        assert result.get(text, 0.0) == pytest.approx(expected, abs=1e-15, rel=0), text


# arXiv:2512.12094: the number of Pauli strings that are left when translations are merged.


def orbits(sizes: tuple[int, ...]) -> int:
    """The number of sets of Pauli strings on a ring or a torus that translations turn into
    each other, by Burnside's lemma: the mean number of strings that a translation leaves."""
    total = 0
    for shift in itertools.product(*(range(size) for size in sizes)):
        # A translation moves the sites in cycles as long as its order.
        order = math.lcm(*(size // math.gcd(size, s) for size, s in zip(sizes, shift)))
        total += 4 ** (math.prod(sizes) // order)
    return total // math.prod(sizes)


def test_the_number_of_orbits_of_small_rings():
    # The paper gives (4^n + 4 (n - 1)) / n for a prime n, which is 24 for 3 qubits.
    assert orbits((3,)) == 24
    for n in (2, 3, 5, 7):
        assert orbits((n,)) == (4**n + 4 * (n - 1)) // n


@pytest.mark.parametrize("sizes", [(1,), (2,), (3,), (4,), (5,), (6,), (2, 2), (2, 3), (3, 2)])
def test_merging_translations_leaves_one_string_for_every_orbit(sizes):
    n = math.prod(sizes)
    every_string = {"".join(chars): 1.0 for chars in itertools.product("IXYZ", repeat=n)}
    total = pauli_sum(n, every_string)
    merged = total.translation_merge(*sizes) if len(sizes) == 2 else total.translation_merge()
    assert len(merged) == orbits(sizes)
    # Every string went to one of them.
    assert sum(coeff for _, coeff in merged.terms()) == 4**n


# arXiv:2409.01706: in circuits whose layers are scrambled by random
# single-qubit Clifford gates, the truncation by weight has a mean squared error of at most
# (2/3)^(k+1) times the squared norm of the observable, and the paths that are kept are
# uncorrelated with those that are dropped, so that the error is the difference of the mean
# squares of the exact and the truncated value.


def scrambled_circuit(rng: random.Random, n: int, layers: int) -> pp.Circuit:
    """Layers of a random single-qubit Clifford gate on every qubit and one rotation."""
    circuit = pp.Circuit(n)
    for _ in range(layers):
        for q in range(n):
            # One of the six ways to permute X, Y and Z, and one of the four signs.
            for gate in rng.choice(["", "h", "s", "hs", "sh", "hsh"]) + rng.choice(["", "x", "y", "z"]):
                getattr(circuit, gate)(q)
        a, b = rng.sample(range(n), 2)
        circuit.rotation(rng.choice("XYZ") + rng.choice("XYZ"), [a, b], rng.uniform(-math.pi, math.pi))
    return circuit


@pytest.mark.parametrize("max_weight", [1, 2, 3])
def test_the_error_of_weight_truncation_in_scrambled_circuits(max_weight):
    n, layers, circuits = 6, 24, 600
    rng = random.Random(max_weight)
    observable = single(n, "ZZ", [2, 3])
    errors, correlations, exact_squares, kept_squares = [], [], [], []
    for _ in range(circuits):
        circuit = scrambled_circuit(rng, n, layers)
        exact = pp.expectation(circuit, observable, min_abs_coeff=0.0)
        kept = pp.expectation(circuit, observable, min_abs_coeff=0.0, max_weight=max_weight)
        errors.append((exact - kept) ** 2)
        correlations.append(kept * (exact - kept))
        exact_squares.append(exact**2)
        kept_squares.append(kept**2)

    def mean(values):
        return sum(values) / len(values)

    def standard_error(values):
        return math.sqrt(mean([(v - mean(values)) ** 2 for v in values]) / len(values))

    assert mean(errors) > 1e-3, "the truncation has to drop something"
    assert mean(errors) + 5 * standard_error(errors) < (2 / 3) ** (max_weight + 1)
    assert abs(mean(correlations)) < 5 * standard_error(correlations)
    difference = [a - b - c for a, b, c in zip(exact_squares, kept_squares, errors)]
    assert abs(mean(difference)) < 5 * standard_error(difference)


# arXiv:2411.19896: an angle that is uniformly distributed in [-r, r] has the mean squared sine
# (1 - sin(2r) / (2r)) / 2, and dropping the paths with at least k sines out of m rotations
# with such angles has a mean squared error of at most (e m r^2 / (3 k))^k.


@pytest.mark.parametrize("radius", [0.2, 1.0, math.pi])
def test_the_mean_squared_sine_of_one_rotation(radius):
    # Z becomes -sin(theta) Z through a rotation about X and the square root of X, and
    # without the path of the sine nothing is left.
    circuit = pp.Circuit(1)
    circuit.sx(0)
    circuit.rx(0.3, 0)
    observable = single(1, "Z", [0])
    assert abs(pp.expectation(circuit, observable)) == pytest.approx(math.sin(0.3), abs=1e-15)

    samples = 400_000
    expected = (1 - math.sin(2 * radius) / (2 * radius)) / 2
    estimate = pp.estimate_mse(circuit, observable, samples, thetas=radius, max_sins=0, seed=5)
    assert abs(estimate - expected) < 5 * math.sqrt(expected * (1 - expected) / samples)


@pytest.mark.parametrize(("radius", "sines"), [(0.1, 1), (0.17, 2)])
def test_the_error_of_dropping_paths_with_many_sines(radius, sines):
    n, layers = 8, 3
    circuit = pp.Circuit(n)
    for _ in range(layers):
        for q in range(n - 1):
            circuit.rzz(0.1, q, q + 1)
        for q in range(n):
            circuit.rx(0.1, q)
    rotations = circuit.num_rotations
    bound = (math.e * rotations * radius**2 / (3 * sines)) ** sines
    assert bound < 0.6, "the bound has to say something"

    observable = single(n, "Z", [n // 2])
    estimate = pp.estimate_mse(circuit, observable, 200_000, thetas=radius, max_sins=sines - 1, seed=sines)
    assert 0 < estimate < bound

    # The error at angles within the radius is of the size that the estimate gives.
    rng = random.Random(sines)
    errors = []
    for _ in range(40):
        bound_circuit = circuit.with_angles([rng.uniform(-radius, radius) for _ in range(rotations)])
        exact = pp.expectation(bound_circuit, observable, min_abs_coeff=0.0)
        kept = pp.expectation(bound_circuit, observable, min_abs_coeff=0.0, max_sins=sines - 1)
        errors.append((exact - kept) ** 2)
    assert sum(errors) / len(errors) < bound
    assert label(n, "Z", [n // 2]) in dict(pp.propagate(circuit, observable).terms())
