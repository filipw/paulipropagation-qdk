"""Helpers that take Q# through the QDK: to QIR for the engine, to a state vector as the oracle."""

from __future__ import annotations

import random

import numpy as np
from qdk import qsharp

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def circuit_source(body: list[str]) -> str:
    """A Q# operation ``Circ`` that applies ``body`` to the register ``qs``."""
    statements = "\n".join(f"    {line}" for line in body)
    return f"operation Circ(qs : Qubit[]) : Unit {{\n{statements}\n}}\n"


def compile_to_qir(source: str, num_qubits: int, profile=None) -> str:
    """Compiles ``Circ`` on ``num_qubits`` qubits, followed by a measurement of each, to QIR."""
    qsharp.init(target_profile=profile or qsharp.TargetProfile.Base)
    qsharp.eval(source)
    qsharp.eval(
        f"operation Main() : Result[] {{ use qs = Qubit[{num_qubits}]; Circ(qs); MResetEachZ(qs) }}"
    )
    return str(qsharp.compile("Main()"))


def state_vector(source: str, num_qubits: int) -> np.ndarray:
    """The state after ``Circ`` acts on ``num_qubits`` qubits, from the QDK simulator.

    The result has one axis of length two per qubit, in the order of the register.
    """
    qsharp.init()
    qsharp.eval(source)
    qsharp.eval(f"use qs = Qubit[{num_qubits}]; Circ(qs);")
    dump = qsharp.dump_machine()
    assert dump.qubit_count == num_qubits
    # The QDK numbers basis states with the first qubit as the most significant bit.
    return np.array(dump.as_dense_state(), dtype=complex).reshape((2,) * num_qubits)


def pauli_expectation(state: np.ndarray, paulis: str, qubits: list[int]) -> float:
    """``<state| P |state>`` for the Pauli string with ``paulis[i]`` on ``qubits[i]``."""
    transformed = state
    for pauli, qubit in zip(paulis, qubits):
        transformed = np.moveaxis(
            np.tensordot(_PAULI[pauli], transformed, axes=([1], [qubit])), 0, qubit
        )
    value = np.vdot(state, transformed)
    assert abs(value.imag) < 1e-12
    return float(value.real)


def random_body(rng: random.Random, num_qubits: int, num_gates: int) -> list[str]:
    """Random Q# statements over the gates that the QDK can lower to the base profile."""

    def angle() -> str:
        return repr(rng.uniform(-3.0, 3.0))

    def qubits(count: int) -> list[str]:
        return [f"qs[{q}]" for q in rng.sample(range(num_qubits), count)]

    one_qubit = ["H", "X", "Y", "Z", "S", "Adjoint S", "SX", "Adjoint SX", "T", "Adjoint T"]
    two_qubit = ["CNOT", "CY", "CZ", "SWAP"]
    body = []
    for _ in range(num_gates):
        kind = rng.randrange(9)
        if kind == 0:
            body.append(f"{rng.choice(one_qubit)}({qubits(1)[0]});")
        elif kind == 1:
            a, b = qubits(2)
            body.append(f"{rng.choice(two_qubit)}({a}, {b});")
        elif kind == 2:
            body.append(f"{rng.choice(['Rx', 'Ry', 'Rz', 'R1'])}({angle()}, {qubits(1)[0]});")
        elif kind == 3:
            a, b = qubits(2)
            body.append(f"{rng.choice(['Rxx', 'Ryy', 'Rzz'])}({angle()}, {a}, {b});")
        elif kind == 4:
            count = rng.randint(1, min(3, num_qubits))
            paulis = ", ".join(rng.choice(["PauliI", "PauliX", "PauliY", "PauliZ"]) for _ in range(count))
            body.append(f"Exp([{paulis}], {angle()}, [{', '.join(qubits(count))}]);")
        elif kind == 5:
            a, b, c = qubits(3)
            body.append(f"CCNOT({a}, {b}, {c});")
        elif kind == 6:
            a, b = qubits(2)
            body.append(f"Controlled {rng.choice(['Rx', 'Ry', 'Rz'])}([{a}], ({angle()}, {b}));")
        elif kind == 7:
            a, b = qubits(2)
            body.append(f"Controlled {rng.choice(['H', 'S', 'T'])}([{a}], {b});")
        else:
            body.append(f"Rz({angle()}, {qubits(1)[0]});")
    return body
