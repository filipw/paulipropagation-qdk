"""The cases of ``test/test_cliffordgates.jl`` of PauliPropagation.jl.

The tables of the Julia package are the reference: every Clifford gate here has to map
every Pauli string as the table of the same gate does.
"""

import itertools
import math
import random

import numpy as np
import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    CNOT_MATRIX,
    H_MATRIX,
    PAULIS,
    circuit_matrix,
    clifford_tables,
    embed,
    pauli_sum,
    single,
    table_image,
    weight,
)

TABLES = clifford_tables()

# The gates of this package under the names of the tables.
GATES = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "SX": "sx",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
}


def random_sum(rng: random.Random, num_qubits: int, count: int = 200) -> dict[str, float]:
    labels = {"".join(rng.choice(PAULIS) for _ in range(num_qubits)) for _ in range(count)}
    return {text: 0.5 + rng.random() for text in sorted(labels)}


def images(table, qubits, terms: dict[str, float]) -> dict[str, float]:
    result = {}
    for text, coeff in terms.items():
        image, sign = table_image(table, qubits, text)
        assert image not in result
        result[image] = coeff * sign
    return result


def test_the_tables_hold_the_relations_of_cnot_and_h():
    # (control, target) -> (control, target, sign) under conjugation by CNOT.
    cnot = {
        "II": ("II", 1), "IX": ("IX", 1), "IY": ("ZY", 1), "IZ": ("ZZ", 1),
        "XI": ("XX", 1), "XX": ("XI", 1), "XY": ("YZ", 1), "XZ": ("YY", -1),
        "YI": ("YX", 1), "YX": ("YI", 1), "YY": ("XZ", -1), "YZ": ("XY", 1),
        "ZI": ("ZI", 1), "ZX": ("ZX", 1), "ZY": ("IY", 1), "ZZ": ("IZ", 1),
    }  # fmt: skip
    hadamard = {"I": ("I", 1), "X": ("Z", 1), "Y": ("Y", -1), "Z": ("X", 1)}

    for relations, name, qubits in [(cnot, "CNOT", [0, 1]), (hadamard, "H", [0])]:
        for text, image in relations.items():
            assert table_image(TABLES[name], qubits, text) == image

            circuit = pp.Circuit(len(qubits))
            getattr(circuit, GATES[name])(*qubits)
            evolved = pp.propagate(circuit, single(len(qubits), text, qubits))
            assert dict(evolved.terms()) == {image[0]: float(image[1])}


def test_the_action_on_a_state_is_the_transposed_table():
    for name, table in TABLES.items():
        if name.endswith("_transpose"):
            continue
        transposed = TABLES[f"{name}_transpose"]
        assert len(transposed) == len(table)
        # The transposed table undoes the table.
        for index, (image, sign) in enumerate(table):
            assert transposed[image] == (index, sign)


@pytest.mark.parametrize("num_qubits", [8, 32, 64, 100, 200, 300])
def test_gates_match_the_tables_on_every_width_of_register(num_qubits):
    """Every gate, on qubits in both orders and on both sides of the 64 qubits of a word."""
    rng = random.Random(42)
    terms = random_sum(rng, num_qubits)
    start = pauli_sum(num_qubits, terms)

    last = num_qubits - 1
    one_qubit = [(q,) for q in (0, last, 32, 63, 64, 128, 256) if q < num_qubits]
    two_qubits = [
        pair
        for pair in [(0, 1), (1, 0), (last, 0), (31, 32), (32, 31), (63, 64), (64, 63),
                     (127, 128), (128, 127), (255, 256), (256, 255)]
        if max(pair) < num_qubits
    ]  # fmt: skip

    checked = 0
    for name, method in GATES.items():
        for qubits in one_qubit if len(TABLES[name]) == 4 else two_qubits:
            circuit = pp.Circuit(num_qubits)
            getattr(circuit, method)(*qubits)

            evolved = dict(pp.propagate(circuit, start).terms())
            assert evolved == images(TABLES[name], qubits, terms), (name, qubits)

            state = dict(pp.propagate(circuit, start, heisenberg=False).terms())
            assert state == images(TABLES[f"{name}_transpose"], qubits, terms), (name, qubits)
            checked += 1
    assert checked >= 9 * 2


def test_rotations_by_a_quarter_turn_match_the_tables_of_the_gates_they_stand_for():
    """The Julia package has the square root of Y and exp(i pi/4 ZZ) as Clifford gates."""
    rng = random.Random(7)
    terms = random_sum(rng, 6)
    start = pauli_sum(6, terms)
    for name, add in [
        ("SY", lambda circuit, q: circuit.ry(math.pi / 2, q[0])),
        ("ZZpihalf", lambda circuit, q: circuit.rzz(-math.pi / 2, q[0], q[1])),
    ]:
        for qubits in [(0, 1), (4, 2), (5, 0)]:
            qubits = qubits[: 1 if name == "SY" else 2]
            circuit = pp.Circuit(6)
            add(circuit, qubits)
            evolved = dict(pp.propagate(circuit, start).terms())
            assert evolved == pytest.approx(images(TABLES[name], qubits, terms), abs=1e-15, rel=0)


def all_strings(num_qubits: int) -> pp.PauliSum:
    labels = ["".join(chars) for chars in itertools.product(PAULIS, repeat=num_qubits)]
    return pauli_sum(num_qubits, {text: 1.0 + i / 7 for i, text in enumerate(labels)})


@pytest.mark.parametrize(
    "gates, single_gate",
    [
        ([("h", 1), ("cx", 0, 1), ("h", 1)], ("cz", 0, 1)),
        ([("h", 1), ("cz", 0, 1), ("h", 1)], ("cx", 0, 1)),
        ([("h", 0), ("z", 0), ("h", 0)], ("x", 0)),
        ([("sx", 0), ("sx", 0)], ("x", 0)),
        ([("s", 0), ("s", 0)], ("z", 0)),
    ],
)
def test_gates_compose_to_the_gate_they_make(gates, single_gate):
    composed, direct = pp.Circuit(2), pp.Circuit(2)
    for name, *qubits in gates:
        getattr(composed, name)(*qubits)
    getattr(direct, single_gate[0])(*single_gate[1:])

    start = all_strings(2)
    assert pp.propagate(composed, start).terms() == pp.propagate(direct, start).terms()
    assert (
        pp.propagate(composed, start, heisenberg=False).terms()
        == pp.propagate(direct, start, heisenberg=False).terms()
    )


def test_a_gate_on_more_qubits_acts_as_the_circuit_it_is_made_of():
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    composed = circuit_matrix(
        [embed(3, CNOT_MATRIX, [0, 1]), embed(3, H_MATRIX, [2]), embed(3, cz, [1, 2])]
    )
    qubits = [4, 1, 8]

    as_one_gate = pp.Circuit(10)
    as_one_gate.unitary(qubits, composed.tolist())
    as_circuit = pp.Circuit(10)
    as_circuit.cx(qubits[0], qubits[1])
    as_circuit.h(qubits[2])
    as_circuit.cz(qubits[1], qubits[2])

    start = pauli_sum(10, random_sum(random.Random(3), 10))
    for heisenberg in (True, False):
        ours = dict(pp.propagate(as_one_gate, start, heisenberg=heisenberg).terms())
        reference = dict(pp.propagate(as_circuit, start, heisenberg=heisenberg).terms())
        assert len(reference) == len(start)
        assert ours == pytest.approx(reference, abs=1e-14, rel=0)


def test_a_weight_cap_counts_the_weight_of_the_images():
    terms = random_sum(random.Random(5), 10)
    circuit = pp.Circuit(10)
    circuit.cx(0, 1)

    evolved = dict(pp.propagate(circuit, pauli_sum(10, terms), max_weight=7).terms())
    kept = {
        text: coeff
        for text, coeff in images(TABLES["CNOT"], [0, 1], terms).items()
        if weight(text) <= 7
    }
    assert 0 < len(kept) < len(terms)
    assert evolved == kept


def test_a_clifford_gate_keeps_a_small_coefficient():
    """The gate changes no coefficient but its sign, so a coefficient threshold drops nothing."""
    circuit = pp.Circuit(10)
    circuit.cx(0, 1)
    text = "XZYIZXXYIZ"
    image, sign = table_image(TABLES["CNOT"], [0, 1], text)

    evolved = pp.propagate(circuit, pauli_sum(10, {text: 1e-12}), min_abs_coeff=1e-10)
    assert dict(evolved.terms()) == {image: sign * 1e-12}


def test_gates_on_wrong_qubits_are_rejected():
    circuit = pp.Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        circuit.h(2)
    with pytest.raises(OverflowError):
        circuit.h(-1)
    with pytest.raises(ValueError, match="more than once|twice|duplicate"):
        circuit.cx(1, 1)
    with pytest.raises(TypeError):
        circuit.h(0, 1)
    with pytest.raises(TypeError):
        circuit.cx(0)
    assert len(circuit) == 0
