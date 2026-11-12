"""The cases of ``test/test_paulitransfermaps.jl`` of PauliPropagation.jl.

A gate is given by a unitary matrix or by the matrix of its action on Pauli strings; both
have to act as the circuit the matrix was computed from with numpy.
"""

import math
import random

import numpy as np
import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    CNOT_MATRIX,
    H_MATRIX,
    MATRICES,
    PAULIS,
    T_MATRIX,
    bricklayer_topology,
    circuit_matrix,
    embed,
    pauli_sum,
    rotation_matrix,
    single,
    transfer_matrix,
)


def action(circuit: pp.Circuit, heisenberg: bool = True) -> np.ndarray:
    """The matrix whose column `j` holds the image of the Pauli operator `j` on one qubit."""
    matrix = np.zeros((4, 4))
    for j, pauli in enumerate(PAULIS):
        evolved = pp.propagate(
            circuit, single(1, pauli, [0]), heisenberg=heisenberg, min_abs_coeff=0.0
        )
        for text, coeff in evolved.terms():
            matrix[PAULIS.index(text), j] = coeff
    return matrix


def assert_close(ours: pp.PauliSum, theirs: pp.PauliSum, tolerance: float) -> None:
    ours, theirs = dict(ours.terms()), dict(theirs.terms())
    assert len(theirs) > 1
    for text in ours.keys() | theirs.keys():
        assert ours.get(text, 0.0) == pytest.approx(theirs.get(text, 0.0), abs=tolerance, rel=0)


def test_transfer_matrix_of_a_y_rotation():
    theta = random.Random(0).random()
    cos, sin = math.cos(theta), math.sin(theta)
    expected = np.array([[1, 0, 0, 0], [0, cos, 0, -sin], [0, 0, 1, 0], [0, sin, 0, cos]])

    circuit = pp.Circuit(1)
    circuit.unitary([0], rotation_matrix(1, "Y", [0], theta).tolist())
    assert np.linalg.norm(action(circuit) - expected) < 1e-12
    assert np.linalg.norm(action(circuit, heisenberg=False) - expected.T) < 1e-12
    assert np.linalg.norm(np.array(transfer_matrix(rotation_matrix(1, "Y", [0], theta))) - expected) < 1e-12


def test_transfer_matrix_of_the_t_gate():
    r = 1 / math.sqrt(2)
    expected = np.array([[1, 0, 0, 0], [0, r, r, 0], [0, -r, r, 0], [0, 0, 0, 1]])

    as_matrix, as_gate = pp.Circuit(1), pp.Circuit(1)
    as_matrix.unitary([0], T_MATRIX.tolist())
    as_gate.t(0)
    for circuit in (as_matrix, as_gate):
        assert np.linalg.norm(action(circuit) - expected) < 1e-12
        assert np.linalg.norm(action(circuit, heisenberg=False) - expected.T) < 1e-12


def test_a_matrix_gate_acts_as_the_rotation_it_was_computed_from():
    theta = math.pi / 4
    unitary = rotation_matrix(1, "X", [0], theta)
    rotation, as_unitary, as_map = pp.Circuit(1), pp.Circuit(1), pp.Circuit(1)
    rotation.rx(theta, 0)
    as_unitary.unitary([0], unitary.tolist())
    as_map.transfer_map([0], transfer_matrix(unitary))

    assert np.linalg.norm(action(as_unitary) - action(rotation)) < 1e-15
    assert np.linalg.norm(action(as_map) - action(rotation)) < 1e-15
    # The identity is mapped to itself and to nothing else.
    assert action(as_unitary)[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_matrices_of_the_wrong_shape_are_rejected():
    circuit = pp.Circuit(2)
    with pytest.raises(ValueError):
        circuit.transfer_map([0], [[1.0] * 3] * 3)
    with pytest.raises(ValueError):
        circuit.unitary([0], [[1.0] * 3] * 3)
    with pytest.raises(ValueError):
        circuit.transfer_map([0], [[1.0] * 4] * 3)
    # A matrix for one qubit on two qubits.
    with pytest.raises(ValueError):
        circuit.transfer_map([0, 1], transfer_matrix(np.eye(2)))
    with pytest.raises(ValueError):
        circuit.unitary([0, 1], np.eye(2).tolist())
    with pytest.raises(ValueError, match="unitary"):
        circuit.unitary([0], [[1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        circuit.unitary([0, 0], np.eye(4).tolist())
    with pytest.raises(ValueError, match="out of range"):
        circuit.unitary([2], np.eye(2).tolist())
    assert len(circuit) == 0


def test_a_map_for_one_qubit_acts_on_the_qubit_it_is_placed_on():
    theta = 0.3
    unitary = rotation_matrix(1, "X", [0], theta)
    rotation, as_unitary, as_map = pp.Circuit(2), pp.Circuit(2), pp.Circuit(2)
    rotation.rx(theta, 1)
    as_unitary.unitary([1], unitary.tolist())
    as_map.transfer_map([1], transfer_matrix(unitary))

    for pauli in PAULIS:
        start = single(2, "Z" + pauli, [0, 1])
        expected = dict(pp.propagate(rotation, start).terms())
        for circuit in (as_unitary, as_map):
            assert dict(pp.propagate(circuit, start).terms()) == pytest.approx(
                expected, abs=1e-15, rel=0
            )


def test_a_map_between_other_gates():
    theta = 0.3
    unitary = rotation_matrix(1, "X", [0], theta)
    with_map, with_rotation = pp.Circuit(2), pp.Circuit(2)
    for circuit, add in [
        (with_map, lambda c: c.transfer_map([1], transfer_matrix(unitary))),
        (with_rotation, lambda c: c.rx(theta, 1)),
    ]:
        circuit.ry(0.4, 0)
        add(circuit)
        circuit.cx(0, 1)
        add(circuit)

    start = single(2, "Z", [1])
    assert_close(pp.propagate(with_map, start), pp.propagate(with_rotation, start), 1e-15)


def test_maps_on_qubits_that_are_not_neighbours():
    identity = pp.Circuit(4)
    identity.unitary([0, 2], np.eye(4).tolist())
    start = single(4, "ZY", [0, 2])
    assert pp.propagate(identity, start).terms() == start.terms()

    as_gate, as_unitary, as_map = pp.Circuit(4), pp.Circuit(4), pp.Circuit(4)
    as_gate.cx(0, 2)
    as_unitary.unitary([0, 2], CNOT_MATRIX.tolist())
    as_map.transfer_map([0, 2], transfer_matrix(CNOT_MATRIX))
    start = single(4, "XYZ", [0, 2, 3])
    expected = pp.propagate(as_gate, start).terms()
    assert expected != start.terms()
    assert pp.propagate(as_unitary, start).terms() == expected
    assert pp.propagate(as_map, start).terms() == expected

    # The order of the qubits is the order of the matrix.
    reversed_gate, reversed_unitary = pp.Circuit(4), pp.Circuit(4)
    reversed_gate.cx(2, 0)
    reversed_unitary.unitary([2, 0], CNOT_MATRIX.tolist())
    expected = pp.propagate(reversed_gate, start).terms()
    assert expected != pp.propagate(as_gate, start).terms()
    assert pp.propagate(reversed_unitary, start).terms() == expected


def test_a_map_of_a_circuit_with_clifford_and_t_gates():
    unitary = circuit_matrix(
        [
            CNOT_MATRIX,
            embed(2, MATRICES["X"], [0]),
            embed(2, H_MATRIX, [1]),
            embed(2, T_MATRIX, [0]),
            embed(2, T_MATRIX, [1]),
        ]
    )
    as_unitary, as_circuit = pp.Circuit(2), pp.Circuit(2)
    as_unitary.unitary([0, 1], unitary.tolist())
    as_circuit.cx(0, 1)
    as_circuit.x(0)
    as_circuit.h(1)
    as_circuit.t(0)
    as_circuit.t(1)

    start = single(2, "YX", [0, 1])
    assert_close(pp.propagate(as_unitary, start), pp.propagate(as_circuit, start), 1e-14)


def test_a_map_of_a_circuit_on_five_qubits():
    rng = random.Random(8)
    num_qubits = 5
    circuit = pp.Circuit(num_qubits)
    matrices = []
    for paulis, layer in [
        ("X", [(q,) for q in range(num_qubits)]),
        ("Z", [(q,) for q in range(num_qubits)]),
        ("YY", bricklayer_topology(num_qubits)),
    ]:
        for qubits in layer:
            theta = rng.gauss(0.0, 1.0)
            circuit.rotation(paulis, list(qubits), theta)
            matrices.append(rotation_matrix(num_qubits, paulis, qubits, theta))

    unitary = circuit_matrix(matrices)
    as_unitary, as_map = pp.Circuit(num_qubits), pp.Circuit(num_qubits)
    as_unitary.unitary(list(range(num_qubits)), unitary.tolist())
    as_map.transfer_map(list(range(num_qubits)), transfer_matrix(unitary))

    start = pauli_sum(num_qubits, {"".join(rng.choice("XYZ") for _ in range(num_qubits)): 1.0})
    for heisenberg in (True, False):
        expected = pp.propagate(circuit, start, heisenberg=heisenberg, min_abs_coeff=0.0)
        for matrix_gate in (as_unitary, as_map):
            evolved = pp.propagate(matrix_gate, start, heisenberg=heisenberg, min_abs_coeff=0.0)
            assert_close(evolved, expected, 1e-13)
