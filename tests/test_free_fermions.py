"""Circuits of RXX on neighbouring qubits and RZ are free fermions: an observable that is
quadratic in Majorana operators stays quadratic, so a matrix of twice the number of qubits
holds it exactly at any size. That gives every coefficient of a propagated sum on a hundred
qubits without a Pauli sum, a state vector or a truncation.

The Majorana operators of a chain are `g[2j] = Z...Z X_j` and `g[2j+1] = Z...Z Y_j`, with Z on
the qubits before `j`. Then `Z_j = -i g[2j] g[2j+1]` and `X_j X_{j+1} = -i g[2j+1] g[2j+2]`,
and the rotation `exp(-i theta/2 (-i g[a] g[b]))` turns `g[a]` and `g[b]` into each other.
"""

import math
import random

import numpy as np
import pytest
from reference_helpers import label_matrix, multiply, rotation_matrix

import paulipropagation_qdk as pp


def majorana(num_qubits: int, index: int) -> str:
    site = index // 2
    return "Z" * site + "XY"[index % 2] + "I" * (num_qubits - site - 1)


def bilinear(num_qubits: int, a: int, b: int) -> tuple[str, int]:
    """`i g[a] g[b]` as (label, sign)."""
    text, power = multiply(majorana(num_qubits, a), majorana(num_qubits, b))
    assert power % 2 == 1, "the product of two different Majorana operators is anti-Hermitian"
    return text, 1 if (power + 1) % 4 == 0 else -1


def pair_of(gate: tuple) -> tuple[int, int]:
    """The Majorana operators `a < b` whose product `-i g[a] g[b]` generates the gate."""
    kind, _, qubit = gate
    return (2 * qubit, 2 * qubit + 1) if kind == "rz" else (2 * qubit + 1, 2 * qubit + 2)


def evolve(h: np.ndarray, gates: list[tuple]) -> np.ndarray:
    """The matrix of `U† O U` for `O = sum over a < b of h[a, b] i g[a] g[b]`, h antisymmetric.

    Under `exp(-i theta/2 (-i g[a] g[b]))` an observable sees `g[a] -> cos g[a] - sin g[b]`
    and `g[b] -> cos g[b] + sin g[a]`. The last gate of the circuit acts on the observable
    first.
    """
    h = h.copy()
    for gate in reversed(gates):
        a, b = pair_of(gate)
        cos, sin = math.cos(gate[1]), math.sin(gate[1])
        turn = np.array([[cos, -sin], [sin, cos]])
        h[[a, b], :] = turn.T @ h[[a, b], :]
        h[:, [a, b]] = h[:, [a, b]] @ turn
    return h


def terms_of(h: np.ndarray, num_qubits: int) -> dict[str, float]:
    terms = {}
    for a in range(2 * num_qubits):
        for b in range(a + 1, 2 * num_qubits):
            if h[a, b] != 0.0:
                text, sign = bilinear(num_qubits, a, b)
                terms[text] = sign * h[a, b]
    return terms


def random_gates(rng: random.Random, num_qubits: int, layers: int) -> list[tuple]:
    gates = []
    for _ in range(layers):
        for start in (0, 1):
            for q in range(start, num_qubits - 1, 2):
                gates.append(("rxx", rng.uniform(-math.pi, math.pi), q))
        for q in range(num_qubits):
            gates.append(("rz", rng.uniform(-math.pi, math.pi), q))
    return gates


def circuit_of(num_qubits: int, gates: list[tuple]) -> pp.Circuit:
    circuit = pp.Circuit(num_qubits)
    for kind, theta, q in gates:
        if kind == "rz":
            circuit.rz(theta, q)
        else:
            circuit.rxx(theta, q, q + 1)
    return circuit


def random_quadratic(rng: random.Random, num_qubits: int, entries: int) -> np.ndarray:
    h = np.zeros((2 * num_qubits, 2 * num_qubits))
    for _ in range(entries):
        a, b = rng.sample(range(2 * num_qubits), 2)
        h[a, b] = rng.uniform(-1, 1)
        h[b, a] = -h[a, b]
    return h


def sum_of(terms: dict[str, float], num_qubits: int) -> pp.PauliSum:
    result = pp.PauliSum(num_qubits)
    for text, coeff in terms.items():
        result.add(text, list(range(num_qubits)), coeff)
    return result


def zero_state_value(h: np.ndarray, num_qubits: int) -> float:
    """`<0...0| O |0...0>`: only `i g[2j] g[2j+1] = -Z_j` has a value there, which is -1."""
    return -sum(h[2 * j, 2 * j + 1] for j in range(num_qubits))


@pytest.mark.parametrize("seed", range(4))
def test_the_majorana_matrix_agrees_with_dense_matrices(seed):
    # The reference itself, checked where a matrix of the whole register still fits.
    rng = random.Random(seed)
    n = 4
    gates = random_gates(rng, n, layers=3)
    h = random_quadratic(rng, n, entries=6)

    unitary = np.eye(2**n, dtype=complex)
    for kind, theta, q in gates:
        paulis, qubits = ("Z", [q]) if kind == "rz" else ("XX", [q, q + 1])
        unitary = rotation_matrix(n, paulis, qubits, theta) @ unitary
    observable = sum(c * label_matrix(text) for text, c in terms_of(h, n).items())
    evolved = unitary.conj().T @ observable @ unitary

    expected = sum(c * label_matrix(text) for text, c in terms_of(evolve(h, gates), n).items())
    assert np.allclose(evolved, expected, atol=1e-12)
    assert evolved[0, 0].real == pytest.approx(zero_state_value(evolve(h, gates), n), abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_every_coefficient_on_100_qubits(seed):
    rng = random.Random(100 + seed)
    n = 100
    gates = random_gates(rng, n, layers=12)
    h = random_quadratic(rng, n, entries=5)
    expected_h = evolve(h, gates)
    expected = terms_of(expected_h, n)

    evolved = pp.propagate(circuit_of(n, gates), sum_of(terms_of(h, n), n), min_abs_coeff=0.0)

    got = dict(evolved.terms())
    assert len(expected) > 3000, "the observable has to spread"
    for text in got.keys() | expected.keys():
        assert got.get(text, 0.0) == pytest.approx(expected.get(text, 0.0), abs=1e-12), text
    assert evolved.overlap_with_zero() == pytest.approx(zero_state_value(expected_h, n), abs=1e-12)


def test_a_magnetization_on_100_qubits_with_the_default_truncation():
    # Z in the middle of the chain, through a Trotter circuit of the transverse-field Ising
    # model. The default truncation drops only what is below 1e-10.
    n = 100
    gates = []
    for _ in range(20):
        gates += [("rxx", 0.3, q) for q in range(n - 1)]
        gates += [("rz", 0.4, q) for q in range(n)]
    h = np.zeros((2 * n, 2 * n))
    h[2 * 50, 2 * 50 + 1], h[2 * 50 + 1, 2 * 50] = -1.0, 1.0  # Z_50 = -i g[100] g[101]

    observable = pp.PauliSum(n)
    observable.add("Z", [50])
    assert dict(observable.terms()) == terms_of(h, n)

    value = pp.expectation(circuit_of(n, gates), observable)
    assert value == pytest.approx(zero_state_value(evolve(h, gates), n), abs=1e-7)
