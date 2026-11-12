"""Checks gates given as matrices against a simulation with numpy.

A circuit of random unitaries, or of random channels given by Kraus operators, acts on a
state vector or a density matrix as plain matrix products. The engine gets the same gates as
unitary matrices or as transfer matrices and has to give the same expectation values, whether
it propagates the observable backwards or the state forwards.
"""

import itertools
import random

import numpy as np
import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    PAULIS,
    computational_state,
    embed,
    label,
    label_matrix,
    pauli_sum,
    rotation_matrix,
    transfer_matrix,
)


def random_unitary(rng: np.random.Generator, num_qubits: int) -> np.ndarray:
    size = 2**num_qubits
    q, r = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_kraus(rng: np.random.Generator, num_qubits: int) -> list[np.ndarray]:
    """Four operators with `sum K† K = 1`: the blocks of the columns of a larger unitary."""
    size = 2**num_qubits
    isometry = random_unitary(rng, num_qubits + 2)[:, :size]
    return [isometry[i * size : (i + 1) * size, :] for i in range(4)]


def all_labels(num_qubits: int) -> list[str]:
    return ["".join(chars) for chars in itertools.product(PAULIS, repeat=num_qubits)]


@pytest.mark.parametrize("seed", range(10))
def test_unitary_gates_match_a_state_vector(seed):
    rng = np.random.default_rng(seed)
    choose = random.Random(seed)
    num_qubits = 5
    circuit = pp.Circuit(num_qubits)
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0

    for _ in range(8):
        qubits = choose.sample(range(num_qubits), choose.randint(1, 3))
        unitary = random_unitary(rng, len(qubits))
        circuit.unitary(qubits, unitary.tolist())
        state = embed(num_qubits, unitary, qubits) @ state

        paulis = "".join(choose.choice("XYZ") for _ in range(2))
        pair, theta = choose.sample(range(num_qubits), 2), choose.uniform(-3.0, 3.0)
        circuit.rotation(paulis, pair, theta)
        state = rotation_matrix(num_qubits, paulis, pair, theta) @ state

    evolved_state = pp.propagate(
        circuit, computational_state(num_qubits), heisenberg=False, min_abs_coeff=0.0
    )
    labels = choose.sample(all_labels(num_qubits), 40)
    for text in labels:
        expected = np.vdot(state, label_matrix(text) @ state).real
        value = pp.expectation(circuit, pauli_sum(num_qubits, {text: 1.0}), min_abs_coeff=0.0)
        assert value == pytest.approx(expected, abs=1e-10, rel=0), text
        coeff = evolved_state.coeff(text, list(range(num_qubits)))
        assert coeff * 2**num_qubits == pytest.approx(expected, abs=1e-10, rel=0), text


@pytest.mark.parametrize("seed", range(10))
def test_channels_given_as_transfer_matrices_match_a_density_matrix(seed):
    rng = np.random.default_rng(1000 + seed)
    choose = random.Random(1000 + seed)
    num_qubits = 4
    circuit = pp.Circuit(num_qubits)
    rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    rho[0, 0] = 1.0

    for _ in range(6):
        paulis = "".join(choose.choice("XYZ") for _ in range(2))
        pair, theta = choose.sample(range(num_qubits), 2), choose.uniform(-3.0, 3.0)
        circuit.rotation(paulis, pair, theta)
        rotation = rotation_matrix(num_qubits, paulis, pair, theta)
        rho = rotation @ rho @ rotation.conj().T

        qubits = choose.sample(range(num_qubits), choose.randint(1, 2))
        kraus = random_kraus(rng, len(qubits))
        assert np.allclose(sum(k.conj().T @ k for k in kraus), np.eye(2 ** len(qubits)))
        circuit.transfer_map(qubits, transfer_matrix(*kraus))
        embedded = [embed(num_qubits, k, qubits) for k in kraus]
        rho = sum(k @ rho @ k.conj().T for k in embedded)

    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    assert np.trace(rho @ rho).real < 0.9

    evolved_state = pp.propagate(
        circuit, computational_state(num_qubits), heisenberg=False, min_abs_coeff=0.0
    )
    largest = 0.0
    for text in all_labels(num_qubits):
        expected = np.trace(rho @ label_matrix(text)).real
        value = pp.expectation(circuit, pauli_sum(num_qubits, {text: 1.0}), min_abs_coeff=0.0)
        assert value == pytest.approx(expected, abs=1e-10, rel=0), text
        coeff = evolved_state.coeff(text, list(range(num_qubits)))
        assert coeff * 2**num_qubits == pytest.approx(expected, abs=1e-10, rel=0), text
        if text.strip("I"):
            largest = max(largest, abs(expected))
    assert largest > 1e-3


def test_any_real_matrix_acts_column_by_column():
    """Column `j` of a transfer matrix is the image of the Pauli string numbered `j`."""
    choose = random.Random(5)
    num_qubits = 4
    for qubits in ([2], [3, 0], [1, 2]):
        size = 4 ** len(qubits)
        matrix = [[choose.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
        # A column of zeros maps its Pauli string to nothing.
        for row in matrix:
            row[size - 1] = 0.0
        circuit = pp.Circuit(num_qubits)
        circuit.transfer_map(qubits, matrix)

        terms = {text: choose.uniform(-1.0, 1.0) for text in choose.sample(all_labels(num_qubits), 60)}
        expected: dict[str, float] = {}
        for text, coeff in terms.items():
            column = sum(PAULIS.index(text[q]) << (2 * i) for i, q in enumerate(qubits))
            for row in range(size):
                chars = list(text)
                for i, q in enumerate(qubits):
                    chars[q] = PAULIS[(row >> (2 * i)) & 3]
                image = "".join(chars)
                expected[image] = expected.get(image, 0.0) + coeff * matrix[row][column]

        evolved = dict(pp.propagate(circuit, pauli_sum(num_qubits, terms), min_abs_coeff=0.0).terms())
        expected = {text: coeff for text, coeff in expected.items() if text in evolved or coeff != 0.0}
        assert evolved.keys() == expected.keys()
        assert evolved == pytest.approx(expected, abs=1e-13, rel=0)
        dropped = label(num_qubits, "Z" * len(qubits), qubits)
        assert pp.propagate(circuit, pauli_sum(num_qubits, {dropped: 1.0})).terms() == []
