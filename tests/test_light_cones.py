"""Checks propagation on wide registers against a state vector of the light cone.

A gate outside the light cone of an observable cannot change its expectation value. A random
circuit of rotations and other gates on 300 qubits, shallow enough for the light cone of an
observable to stay on about ten qubits, is therefore simulated exactly by a state vector of
those qubits alone, while the propagation runs on the whole register without truncation.
"""

import cmath
import math
import random

import numpy as np
import pytest
from reference_helpers import single

import paulipropagation_qdk as pp

PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
# Matrices of the gates without an angle, with the first qubit as the most significant bit.
FIXED = {
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "s": np.diag([1, 1j]),
    "s_adj": np.diag([1, -1j]),
    "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    "t": np.diag([1, cmath.exp(1j * math.pi / 4)]),
    "y": PAULI["Y"],
    "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "cy": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
    "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
    "ccx": np.eye(8, dtype=complex)[[0, 1, 2, 3, 4, 5, 7, 6]],
}
ONE_QUBIT = ["h", "s", "s_adj", "sx", "t", "y"]
TWO_QUBIT = ["cx", "cy", "cz", "swap"]

# A gate is its name or the Pauli operators of a rotation, its qubits, and an angle or None.
Gates = list[tuple[str, tuple[int, ...], float | None]]


def random_shallow_circuit(rng: random.Random, n: int, layers: int) -> Gates:
    """Layers of a gate on every qubit and a gate on every other pair of neighbours."""
    gates: Gates = []
    for layer in range(layers):
        for q in range(n):
            if rng.random() < 0.6:
                gates.append((rng.choice("XYZ"), (q,), rng.uniform(-math.pi, math.pi)))
            else:
                gates.append((rng.choice(ONE_QUBIT), (q,), None))
        for q in range(layer % 2, n - 1, 2):
            pair = (q, q + 1) if rng.random() < 0.5 else (q + 1, q)
            if rng.random() < 0.6:
                paulis = rng.choice("XYZ") + rng.choice("XYZ")
                gates.append((paulis, pair, rng.uniform(-math.pi, math.pi)))
            else:
                gates.append((rng.choice(TWO_QUBIT), pair, None))
    return gates


def build(n: int, gates: Gates) -> pp.Circuit:
    circuit = pp.Circuit(n)
    for name, qubits, angle in gates:
        if angle is None:
            getattr(circuit, name)(*qubits)
        else:
            circuit.rotation(name, list(qubits), angle)
    return circuit


def light_cone(gates: Gates, qubits: tuple[int, ...]) -> tuple[list[int], Gates]:
    """The qubits that the observable on `qubits` can depend on, and the gates that it can
    depend on: going back through the circuit, every gate that touches the cone joins it."""
    cone, kept = set(qubits), []
    for gate in reversed(gates):
        if cone & set(gate[1]):
            cone |= set(gate[1])
            kept.append(gate)
    return sorted(cone), kept[::-1]


def apply(state: np.ndarray, matrix: np.ndarray, axes: list[int]) -> np.ndarray:
    """A matrix on the given axes of a state with one axis of length 2 for every qubit."""
    k = len(axes)
    tensor = matrix.reshape((2,) * (2 * k))
    moved = np.tensordot(tensor, state, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(moved, list(range(k)), axes)


def matrix_of(name: str, angle: float | None) -> np.ndarray:
    if angle is None:
        return FIXED[name]
    generator = np.eye(1, dtype=complex)
    for pauli in name:
        generator = np.kron(generator, PAULI[pauli])
    return math.cos(angle / 2) * np.eye(len(generator)) - 1j * math.sin(angle / 2) * generator


def expectation_on_the_cone(
    gates: Gates, paulis: str, qubits: tuple[int, ...], register: int | None = None
) -> tuple[float, int]:
    """The expectation value from a state vector of the light cone, and the size of the cone.
    With `register`, the state vector is that of all qubits and all gates instead."""
    cone, kept = light_cone(gates, qubits) if register is None else (list(range(register)), gates)
    place = {q: axis for axis, q in enumerate(cone)}
    state = np.zeros((2,) * len(cone), dtype=complex)
    state[(0,) * len(cone)] = 1
    for name, gate_qubits, angle in kept:
        state = apply(state, matrix_of(name, angle), [place[q] for q in gate_qubits])
    measured = state
    for pauli, q in zip(paulis, qubits):
        measured = apply(measured, PAULI[pauli], [place[q]])
    return float(np.vdot(state, measured).real), len(cone)


def test_the_state_vector_of_a_cone_is_that_of_the_register():
    rng = random.Random(1)
    n = 11
    gates = random_shallow_circuit(rng, n, 3)
    sizes = []
    for paulis, qubits in [("Z", (5,)), ("X", (0,)), ("Y", (3,)), ("YZ", (9, 10)), ("XY", (4, 5)), ("ZZ", (6, 7))]:
        on_cone, size = expectation_on_the_cone(gates, paulis, qubits)
        whole, _ = expectation_on_the_cone(gates, paulis, qubits, register=n)
        assert size < n
        sizes.append(abs(whole))
        assert on_cone == pytest.approx(whole, abs=1e-14, rel=0)
        value = pp.expectation(build(n, gates), single(n, paulis, list(qubits)), min_abs_coeff=0.0)
        assert value == pytest.approx(whole, abs=1e-13, rel=0)
    assert sum(size > 1e-3 for size in sizes) >= 3


@pytest.mark.parametrize("seed", range(4))
def test_observables_on_a_wide_register_match_their_light_cones(seed):
    n, layers = 300, 4
    rng = random.Random(100 + seed)
    gates = random_shallow_circuit(rng, n, layers)
    circuit = build(n, gates)
    assert circuit.num_rotations > 800

    # Observables at the ends of the register and on both sides of the 64-bit words.
    places = [0, 1, 31, 32, 62, 63, 64, 65, 127, 128, 150, 191, 192, 255, 256, 298, 299]
    largest = 0.0
    for q in places:
        for paulis, qubits in [(rng.choice("XYZ"), (q,)), (rng.choice("XYZ") + rng.choice("XYZ"), (q - 1, q))]:
            if min(qubits) < 0:
                continue
            expected, size = expectation_on_the_cone(gates, paulis, qubits)
            assert 4 <= size <= 2 * layers + 2
            value = pp.expectation(circuit, single(n, paulis, list(qubits)), min_abs_coeff=0.0)
            assert value == pytest.approx(expected, abs=1e-12, rel=0), (paulis, qubits)
            largest = max(largest, abs(expected))
    assert largest > 0.1


@pytest.mark.parametrize("n", [100, 200, 300])
def test_complex_observables_on_a_wide_register_match_their_light_cones(n):
    """A sum with complex coefficients, on registers of two, four and five 64-bit words."""
    rng = random.Random(n)
    gates = random_shallow_circuit(rng, n, 3)
    circuit = build(n, gates)

    observable = pp.PauliSum(n, dtype=complex)
    expected = 0.0
    for q in [0, 63, 64, 65, n // 2, n - 2]:
        paulis, qubits = rng.choice("XYZ") + rng.choice("XYZ"), (q, q + 1)
        coeff = complex(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        observable.add(paulis, list(qubits), coeff)
        expected += coeff * expectation_on_the_cone(gates, paulis, qubits)[0]

    evolved = pp.propagate(circuit, observable, min_abs_coeff=0.0)
    assert evolved.dtype is complex
    assert abs(expected.imag) > 1e-3
    assert evolved.overlap_with_zero() == pytest.approx(expected, abs=1e-12, rel=0)

    # The real and the imaginary part are propagated like sums of their own.
    real = pp.PauliSum(n)
    imaginary = pp.PauliSum(n)
    for text, coeff in observable.terms():
        real.add(text, list(range(n)), coeff.real)
        imaginary.add(text, list(range(n)), coeff.imag)
    parts = pp.propagate(circuit, real, min_abs_coeff=0.0) + 1j * pp.propagate(circuit, imaginary, min_abs_coeff=0.0)
    assert parts.dtype is complex
    difference = evolved - parts
    assert len(evolved) > 100
    assert difference.max_abs_coeff() < 1e-14
    # A complex sum without imaginary parts is the real sum again.
    assert dict(pp.propagate(circuit, real.astype(complex), min_abs_coeff=0.0).astype(float).terms()) == dict(
        pp.propagate(circuit, real, min_abs_coeff=0.0).terms()
    )


def test_toffoli_gates_match_a_state_vector():
    rng = random.Random(3)
    n = 5
    gates: Gates = []
    for _ in range(5):
        for q in range(n):
            gates.append((rng.choice("XYZ"), (q,), rng.uniform(-math.pi, math.pi)))
        gates.append(("ccx", tuple(rng.sample(range(n), 3)), None))
    circuit = build(n, gates)

    largest = 0.0
    for paulis, qubits in [("Z", (0,)), ("X", (3,)), ("YZ", (1, 4)), ("ZZZ", (0, 2, 3)), ("XYZXY", (0, 1, 2, 3, 4))]:
        expected, _ = expectation_on_the_cone(gates, paulis, qubits, register=n)
        value = pp.expectation(circuit, single(n, paulis, list(qubits)), min_abs_coeff=0.0)
        assert value == pytest.approx(expected, abs=1e-12, rel=0), (paulis, qubits)
        largest = max(largest, abs(expected))
    assert largest > 0.05

    for qubits in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        with pytest.raises(ValueError, match="distinct"):
            circuit.ccx(*qubits)
