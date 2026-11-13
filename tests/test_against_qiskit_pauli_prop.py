"""Checks propagation against the pauli-prop package of Qiskit, where it is installed.

It is a third implementation of the same rules, which shares no code with this package or
with PauliPropagation.jl. Without truncation the two must agree on every term.
"""

import random

import pytest

import paulipropagation_qdk as pp

pauli_prop = pytest.importorskip("pauli_prop")
qiskit = pytest.importorskip("qiskit")

from qiskit.quantum_info import SparsePauliOp  # noqa: E402

# The other package converts complex coefficients without an imaginary part to real ones.
pytestmark = pytest.mark.filterwarnings("ignore:Casting complex values")


def random_circuits(rng: random.Random, n: int, gates: int):
    """The same random circuit of Clifford gates and rotations for both packages."""
    ours, theirs = pp.Circuit(n), qiskit.QuantumCircuit(n)
    for _ in range(gates):
        a, b = rng.sample(range(n), 2)
        theta = rng.uniform(-3, 3)
        kind = rng.choice(["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap", "rx", "ry", "rz", "rxx", "ryy", "rzz"])
        if kind in ("h", "s", "sdg", "x", "y", "z"):
            getattr(ours, "s_adj" if kind == "sdg" else kind)(a)
            getattr(theirs, kind)(a)
        elif kind in ("cx", "cz", "swap"):
            getattr(ours, kind)(a, b)
            getattr(theirs, kind)(a, b)
        elif kind in ("rx", "ry", "rz"):
            getattr(ours, kind)(theta, a)
            getattr(theirs, kind)(theta, a)
        else:
            getattr(ours, kind)(theta, a, b)
            getattr(theirs, kind)(theta, a, b)
    return ours, theirs


def random_terms(rng: random.Random, n: int, count: int) -> list:
    terms = {}
    while len(terms) < count:
        qubits = tuple(sorted(rng.sample(range(n), rng.randint(1, 3))))
        paulis = "".join(rng.choice("XYZ") for _ in qubits)
        terms[(paulis, qubits)] = rng.uniform(-1, 1)
    return [(paulis, list(qubits), coeff) for (paulis, qubits), coeff in terms.items()]


def as_dict(terms) -> dict:
    return {(paulis, tuple(qubits)): coeff for paulis, qubits, coeff in terms}


@pytest.mark.parametrize("heisenberg", [True, False])
def test_every_term_without_truncation(heisenberg):
    rng = random.Random(31)
    for _ in range(12):
        n = rng.randint(3, 7)
        ours, theirs = random_circuits(rng, n, 40)
        terms = random_terms(rng, n, 3)

        start = pp.PauliSum(n)
        for paulis, qubits, coeff in terms:
            start.add(paulis, qubits, coeff)
        mine = pp.propagate(ours, start, min_abs_coeff=0.0, heisenberg=heisenberg)

        operator = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
        frame = "h" if heisenberg else "s"
        # Its largest number of terms counts terms before equal ones are added up, so it
        # is set well above the 4^n Pauli strings that there are.
        evolved, _ = pauli_prop.propagate_through_circuit(operator, theirs, 10**6, 0.0, frame)

        # A label of this package has qubit 0 first. The other package may list a Pauli
        # string more than once.
        expected = {}
        for paulis, qubits, coeff in evolved.to_sparse_list():
            label = ["I"] * n
            for pauli, qubit in zip(paulis, qubits):
                label[qubit] = pauli
            assert abs(coeff.imag) < 1e-12
            label = "".join(label)
            expected[label] = expected.get(label, 0.0) + coeff.real
        result = dict(mine.terms())
        for label in result.keys() | expected.keys():
            assert result.get(label, 0.0) == pytest.approx(expected.get(label, 0.0), abs=1e-12)
        assert len(result) > 1
