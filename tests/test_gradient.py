"""Checks gradients against the parameter-shift rule: the derivative with respect to the
angle of a Pauli rotation is half the difference of the expectation values with the angle
shifted by pi/2 either way. The values come from numpy matrices on a few qubits and from
propagations without truncation on more."""

import math
import random

import numpy as np
import pytest
from reference_helpers import (
    H_MATRIX,
    CNOT_MATRIX,
    circuit_matrix,
    embed,
    label_matrix,
    pauli_sum,
    rotation_matrix,
)

import paulipropagation_qdk as pp


def random_layout(rng: random.Random, n: int, gates: int) -> list[tuple]:
    layout = []
    for _ in range(gates):
        kind = rng.choice(["rotation", "rotation", "h", "cx"])
        if kind == "rotation":
            qubits = rng.sample(range(n), rng.randint(1, min(n, 3)))
            layout.append(("".join(rng.choice("XYZ") for _ in qubits), qubits))
        elif kind == "h":
            layout.append(("h", [rng.randrange(n)]))
        else:
            layout.append(("cx", rng.sample(range(n), 2)))
    return layout


def circuit_of(n: int, layout: list[tuple], angles: list[float]) -> pp.Circuit:
    circuit = pp.Circuit(n)
    angles = iter(angles)
    for kind, qubits in layout:
        if kind == "h":
            circuit.h(*qubits)
        elif kind == "cx":
            circuit.cx(*qubits)
        else:
            circuit.rotation(kind, qubits, next(angles))
    return circuit


def dense_value(n: int, layout, angles, observable: np.ndarray, state: np.ndarray) -> float:
    angles = iter(angles)
    matrices = []
    for kind, qubits in layout:
        if kind == "h":
            matrices.append(embed(n, H_MATRIX, qubits))
        elif kind == "cx":
            matrices.append(embed(n, CNOT_MATRIX, qubits))
        else:
            matrices.append(rotation_matrix(n, kind, qubits, next(angles)))
    unitary = circuit_matrix(matrices)
    return float(np.real(np.trace(state @ unitary.conj().T @ observable @ unitary)))


def shifted(angles: list[float], index: int, by: float) -> list[float]:
    return [angle + by if i == index else angle for i, angle in enumerate(angles)]


@pytest.mark.parametrize("seed", range(8))
def test_gradients_match_the_parameter_shift_rule_on_matrices(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 4)
    dim = 2**n
    layout = random_layout(rng, n, 12)
    angles = [rng.uniform(-3, 3) for kind, _ in layout if kind not in ("h", "cx")]
    terms = {"".join(rng.choice("IXYZ") for _ in range(n)): rng.uniform(-1, 1) for _ in range(3)}
    observable = sum(c * label_matrix(text) for text, c in terms.items())
    ones = [q for q in range(n) if rng.random() < 0.5]
    basis = np.zeros((dim, dim))
    index = sum(1 << (n - 1 - q) for q in ones)
    basis[index, index] = 1.0
    mixture = {"".join(rng.choice("IXYZ") for _ in range(n)): rng.uniform(-1, 1) for _ in range(4)}

    zero = np.zeros((dim, dim))
    zero[0, 0] = 1.0
    states = [
        ("zero", zero),
        ("plus", np.full((dim, dim), 1 / dim)),
        ("mixed", np.eye(dim) / dim),
        (ones, basis),
        # Tr[rho P] is 2^n times the coefficient of P in rho.
        (pauli_sum(n, mixture), sum(c * label_matrix(t) for t, c in mixture.items())),
    ]
    for state, density in states:
        value, derivatives = pp.gradient(
            circuit_of(n, layout, angles), pauli_sum(n, terms), state=state, min_abs_coeff=0.0
        )
        assert value == pytest.approx(dense_value(n, layout, angles, observable, density), abs=1e-11)
        assert len(derivatives) == len(angles)
        for k, derivative in enumerate(derivatives):
            up = dense_value(n, layout, shifted(angles, k, math.pi / 2), observable, density)
            down = dense_value(n, layout, shifted(angles, k, -math.pi / 2), observable, density)
            assert derivative == pytest.approx((up - down) / 2, abs=1e-11)


def trotter(n: int, layers: int, angles: list[float]) -> pp.Circuit:
    circuit = pp.Circuit(n)
    angles = iter(angles)
    for _ in range(layers):
        for q in range(n - 1):
            circuit.rzz(next(angles), q, q + 1)
        for q in range(n):
            circuit.rx(next(angles), q)
        circuit.cz(0, n - 1)
    return circuit


def test_a_gradient_on_20_qubits_matches_shifted_propagations():
    rng = random.Random(1)
    n, layers = 20, 3
    angles = [rng.uniform(0.2, 0.8) for _ in range(layers * (2 * n - 1))]
    observable = pp.PauliSum(n)
    observable.add("Z", [9])
    observable.add("ZZ", [3, 4], 0.5)

    circuit = trotter(n, layers, angles)
    assert circuit.angles == angles
    value, derivatives = pp.gradient(circuit, observable, min_abs_coeff=0.0)
    assert value == pytest.approx(pp.expectation(circuit, observable, min_abs_coeff=0.0), abs=1e-12)
    assert sum(abs(d) > 1e-3 for d in derivatives) > 20
    for k in rng.sample(range(len(angles)), 25):
        up = pp.expectation(circuit.with_angles(shifted(angles, k, math.pi / 2)), observable, min_abs_coeff=0.0)
        down = pp.expectation(circuit.with_angles(shifted(angles, k, -math.pi / 2)), observable, min_abs_coeff=0.0)
        assert derivatives[k] == pytest.approx((up - down) / 2, abs=1e-11)


def test_a_truncated_gradient_is_close_and_does_not_depend_on_threads():
    rng = random.Random(2)
    n, layers = 24, 4
    angles = [rng.uniform(0.1, 0.5) for _ in range(layers * (2 * n - 1))]
    circuit = trotter(n, layers, angles)
    observable = pp.PauliSum(n)
    observable.add("Z", [11])

    exact = pp.gradient(circuit, observable, max_weight=8, min_abs_coeff=1e-9)
    truncated = pp.gradient(circuit, observable, max_weight=4, min_abs_coeff=1e-4)
    assert truncated[0] == pp.expectation(circuit, observable, max_weight=4, min_abs_coeff=1e-4)
    error = sum(abs(a - b) for a, b in zip(truncated[1], exact[1]))
    assert 0 < error < 0.05 * sum(map(abs, exact[1]))

    for threads in (1, 2, 5):
        assert pp.gradient(circuit, observable, max_weight=4, min_abs_coeff=1e-4, threads=threads) == truncated


@pytest.mark.parametrize(
    "option",
    [{"max_weight": 3}, {"min_abs_coeff": 1e-3}, {"min_rel_coeff": 1e-3}, {"weight_damping": (0.3, 1e-4)}],
    ids=lambda option: next(iter(option)),
)
def test_every_truncation_reaches_a_gradient(option):
    rng = random.Random(3)
    n, layers = 16, 3
    angles = [rng.uniform(0.1, 0.5) for _ in range(layers * (2 * n - 1))]
    circuit = trotter(n, layers, angles)
    observable = pp.PauliSum(n)
    observable.add("Z", [7])

    exact = pp.gradient(circuit, observable, min_abs_coeff=0.0)
    options = {"min_abs_coeff": 0.0, **option}
    value, derivatives = pp.gradient(circuit, observable, **options)
    # The value is that of the propagation with the same truncation, which drops terms.
    assert value == pp.expectation(circuit, observable, **options)
    assert value != exact[0]
    error = sum(abs(a - b) for a, b in zip(derivatives, exact[1]))
    assert 0 < error < 0.1 * sum(map(abs, exact[1]))


def test_t_gates_are_rotations_with_a_derivative():
    circuit = pp.Circuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    observable = pp.PauliSum(1)
    observable.add("Z", [0])
    value, derivatives = pp.gradient(circuit, observable)
    # H T H turns Z into cos(theta) Z - sin(theta) Y at theta = pi/4.
    assert circuit.angles == [math.pi / 4]
    assert value == pytest.approx(math.cos(math.pi / 4))
    assert derivatives == [pytest.approx(-math.sin(math.pi / 4))]


def test_what_a_gradient_cannot_do_is_rejected():
    circuit = pp.Circuit(2)
    circuit.rx(0.3, 0)
    observable = pp.PauliSum(2)
    observable.add("Z", [0])

    noisy = pp.Circuit(2)
    noisy.rx(0.3, 0)
    noisy.depolarizing(0.1, 0)
    with pytest.raises(ValueError, match="noise"):
        pp.gradient(noisy, observable)
    with pytest.raises(ValueError, match="real coefficients"):
        pp.gradient(circuit, observable.astype(complex))
    with pytest.raises(ValueError):
        pp.gradient(circuit, observable, state="one")
    with pytest.raises(ValueError):
        pp.gradient(circuit, observable, state=[2])
    with pytest.raises(ValueError):
        pp.gradient(circuit, observable, state=pp.PauliSum(3))
    with pytest.raises(TypeError):
        pp.gradient(circuit, observable, state=1.5)
    with pytest.raises(TypeError):
        pp.gradient(circuit, observable, max_freq=3)
    with pytest.raises(ValueError):
        circuit.with_angles([0.1, 0.2])
    with pytest.raises(ValueError):
        circuit.with_angles([math.nan])
