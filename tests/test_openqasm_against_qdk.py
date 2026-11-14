"""Checks the whole path from OpenQASM against the QDK's own simulator.

One OpenQASM program goes two ways: compiled by the QDK to QIR, read by ``circuit_from_qir``
and propagated without truncation, and run on the QDK simulator, whose state vector gives the
expectation value directly. The two share only the OpenQASM source.
"""

import math
import random

import numpy as np
import pytest
from qdk import code, openqasm, qsharp
from qdk.openqasm import ProgramType, import_openqasm
from qdk_helpers import pauli_expectation
from test_against_qdk import random_paulis

import paulipropagation_qdk as pp

HEADERS = {
    3: 'OPENQASM 3.0;\ninclude "stdgates.inc";\n',
    2: 'OPENQASM 2.0;\ninclude "qelib1.inc";\n',
}

# A gate of the program itself, which the QDK inlines.
CUSTOM_GATE = "gate entangle(theta) a, b { h a; cx a, b; rz(theta) b; cx a, b; }\n"

ONE_QUBIT = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "id"]
ONE_ANGLE = ["rx", "ry", "rz", "p", "phase", "u1"]
TWO_QUBIT = ["cx", "CX", "cy", "cz", "ch", "swap"]
CONTROLLED_ROTATIONS = ["cp", "cphase", "crx", "cry", "crz"]
TWO_QUBIT_ROTATIONS = ["rxx", "ryy", "rzz"]

# The gates of qelib1.inc, by their number of angles and qubits.
VERSION_2 = [
    *((name, 0, 1) for name in ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]),
    *((name, 1, 1) for name in ["rx", "ry", "rz", "u1"]),
    ("u2", 2, 1),
    ("u3", 3, 1),
    *((name, 0, 2) for name in ["cx", "cy", "cz", "ch"]),
    *((name, 1, 2) for name in ["crz", "cu1"]),
    ("cu3", 3, 2),
    ("ccx", 0, 3),
]


def program(version: int, num_qubits: int, body: list[str], measure: bool) -> str:
    """The statements of ``body`` on a register ``q``, with the measurement of all qubits
    that a QIR program ends in, or without it for the state vector."""
    lines = [HEADERS[version]]
    if version == 3:
        lines += [CUSTOM_GATE, f"qubit[{num_qubits}] q;"]
        lines += [f"bit[{num_qubits}] c;"] if measure else []
        lines += [*body, *(["c = measure q;"] if measure else [])]
    else:
        lines += [f"qreg q[{num_qubits}];"]
        lines += [f"creg c[{num_qubits}];"] if measure else []
        lines += [*body, *(["measure q -> c;"] if measure else [])]
    return "\n".join(lines) + "\n"


def state_vector(source: str, num_qubits: int) -> np.ndarray:
    """The state after the program acts on its register, from the QDK simulator, with one
    axis of length two per qubit in the order of the register."""
    qsharp.init()
    import_openqasm(source, name="Program", program_type=ProgramType.Operation)
    qsharp.eval(f"use qs = Qubit[{num_qubits}]; Program(qs);")
    dump = qsharp.dump_machine()
    assert dump.qubit_count == num_qubits
    return np.array(dump.as_dense_state(), dtype=complex).reshape((2,) * num_qubits)


def random_body(rng: random.Random, num_qubits: int, num_gates: int) -> list[str]:
    """Random OpenQASM 3 statements: the gates of stdgates.inc, the built-in gates, the gate
    modifiers, a gate of the program, gates on a whole register and loops."""

    def angle() -> str:
        return repr(rng.uniform(-3.0, 3.0))

    def angles(count: int) -> str:
        return ", ".join(angle() for _ in range(count))

    def qubits(count: int) -> str:
        return ", ".join(f"q[{q}]" for q in rng.sample(range(num_qubits), count))

    body = []
    for _ in range(num_gates):
        kind = rng.randrange(16)
        if kind == 0:
            body.append(f"{rng.choice(ONE_QUBIT)} {qubits(1)};")
        elif kind == 1:
            body.append(f"{rng.choice(ONE_ANGLE)}({angle()}) {qubits(1)};")
        elif kind == 2:
            name, count = rng.choice([("u2", 2), ("u3", 3), ("U", 3)])
            body.append(f"{name}({angles(count)}) {qubits(1)};")
        elif kind == 3:
            body.append(f"{rng.choice(TWO_QUBIT)} {qubits(2)};")
        elif kind == 4:
            body.append(f"{rng.choice(CONTROLLED_ROTATIONS)}({angle()}) {qubits(2)};")
        elif kind == 5:
            body.append(f"cu({angles(4)}) {qubits(2)};")
        elif kind == 6:
            body.append(f"{rng.choice(['ccx', 'cswap'])} {qubits(3)};")
        elif kind == 7:
            body.append(f"{rng.choice(TWO_QUBIT_ROTATIONS)}({angle()}) {qubits(2)};")
        elif kind == 8:
            modifier = rng.choice(["ctrl", "negctrl"])
            gate = rng.choice([f"rx({angle()})", f"ry({angle()})", "h", "s", "t", "z"])
            body.append(f"{modifier} @ {gate} {qubits(2)};")
        elif kind == 9:
            body.append(f"ctrl(2) @ {rng.choice(['x', 'z'])} {qubits(3)};")
        elif kind == 10:
            gate = rng.choice([f"rz({angle()})", f"U({angles(3)})", "t", "s", "sx"])
            body.append(f"inv @ {gate} {qubits(1)};")
        elif kind == 11:
            power = rng.choice([-2, 2, 3])
            gate = rng.choice([f"ry({angle()})", "t", "s", "h"])
            body.append(f"pow({power}) @ {gate} {qubits(1)};")
        elif kind == 12:
            body.append(f"entangle({angle()}) {qubits(2)};")
        elif kind == 13:
            body.append(f"{rng.choice(['h', 's', f'rx({angle()})'])} q;")
        elif kind == 14:
            last = rng.randint(0, num_qubits - 2)
            scale = repr(rng.uniform(-1.0, 1.0))
            body.append(
                f"for int i in [0:{last}] {{ cx q[i], q[i + 1]; ry({scale} * (i + 1)) q[i]; }}"
            )
        else:
            body.append(f"gphase({angle()});")
    return body


def random_body_of_version_2(rng: random.Random, num_qubits: int, num_gates: int) -> list[str]:
    body = []
    for _ in range(num_gates):
        name, num_angles, count = rng.choice(VERSION_2)
        parameters = ",".join(repr(rng.uniform(-3.0, 3.0)) for _ in range(num_angles))
        operands = ",".join(f"q[{q}]" for q in rng.sample(range(num_qubits), count))
        body.append(f"{name}({parameters}) {operands};" if num_angles else f"{name} {operands};")
    return body


def assert_matches_the_state_vector(rng, version, num_qubits, body):
    qir = openqasm.compile(
        program(version, num_qubits, body, measure=True),
        target_profile=qsharp.TargetProfile.Base,
    )
    circuit = pp.circuit_from_qir(str(qir))
    state = state_vector(program(version, num_qubits, body, measure=False), num_qubits)

    # Qubits beyond the register are scratch space of the compiler that ends in the zero state.
    assert circuit.num_qubits >= num_qubits
    total = pp.PauliSum(circuit.num_qubits)
    expected_total = 0.0
    for _ in range(12):
        paulis, qubits = random_paulis(rng, num_qubits)
        coeff = rng.uniform(-1.0, 1.0)

        observable = pp.PauliSum(circuit.num_qubits)
        observable.add(paulis, qubits)
        expected = pauli_expectation(state, paulis, qubits)
        assert pp.expectation(circuit, observable, min_abs_coeff=0.0) == pytest.approx(
            expected, abs=1e-10, rel=0
        ), (paulis, qubits)

        total.add(paulis, qubits, coeff)
        expected_total += coeff * expected

    assert pp.expectation(circuit, total, min_abs_coeff=0.0) == pytest.approx(
        expected_total, abs=1e-10, rel=0
    )
    return circuit


@pytest.mark.parametrize("seed", range(24))
def test_random_openqasm_programs_match_the_qdk_state_vector(seed):
    rng = random.Random(seed)
    num_qubits = rng.randint(3, 7)
    body = random_body(rng, num_qubits, 30)
    assert_matches_the_state_vector(rng, 3, num_qubits, body)


@pytest.mark.parametrize("seed", range(8))
def test_random_openqasm_2_programs_match_the_qdk_state_vector(seed):
    rng = random.Random(100 + seed)
    num_qubits = rng.randint(3, 7)
    body = random_body_of_version_2(rng, num_qubits, 30)
    assert_matches_the_state_vector(rng, 2, num_qubits, body)


def test_every_statement_of_the_random_programs_is_drawn():
    """The 24 programs use every kind of statement and every gate name."""
    text = "\n".join(
        "\n".join(random_body(rng, rng.randint(3, 7), 30)) for rng in map(random.Random, range(24))
    )
    names = [
        *ONE_QUBIT,
        *ONE_ANGLE,
        *TWO_QUBIT,
        *CONTROLLED_ROTATIONS,
        *TWO_QUBIT_ROTATIONS,
    ]
    names += ["u2", "u3", "U", "cu", "ccx", "cswap", "entangle", "gphase"]
    for name in names:
        assert any(line.startswith((f"{name} ", f"{name}(")) for line in text.splitlines()), name
    for modifier in (
        "ctrl @",
        "negctrl @",
        "ctrl(2) @",
        "inv @",
        "pow(",
        "for int",
        " q;",
    ):
        assert modifier in text, modifier


def test_input_parameters_are_bound_when_the_program_is_compiled():
    source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        input int layers;
        input float coupling;
        input float field;
        qubit[4] q;
        output bit[4] c;
        for int layer in [1:layers] {
            for int i in [0:2] { rzz(coupling) q[i], q[i + 1]; }
            rx(field) q;
        }
        c = measure q;
    """
    with pytest.raises(qsharp.QSharpError, match="unbound input parameters"):
        openqasm.compile(source, target_profile=qsharp.TargetProfile.Base)

    qsharp.init(target_profile=qsharp.TargetProfile.Base)
    import_openqasm(source, name="Evolve", program_type=ProgramType.File)
    circuit = pp.circuit_from_qir(str(openqasm.compile(code.qasm_import.Evolve, 3, 0.2, -0.7)))
    assert (circuit.num_qubits, len(circuit)) == (4, 21)
    # An angle of OpenQASM lies between 0 and a full turn.
    expected = ([0.2] * 3 + [2 * math.pi - 0.7] * 4) * 3
    assert circuit.angles == pytest.approx(expected, abs=1e-14, rel=0)

    by_hand = pp.Circuit(4)
    for _ in range(3):
        for i in range(3):
            by_hand.rzz(0.2, i, i + 1)
        for i in range(4):
            by_hand.rx(-0.7, i)
    observable = pp.PauliSum(4)
    observable.add("ZZ", [1, 2])
    assert pp.expectation(circuit, observable, min_abs_coeff=0.0) == pytest.approx(
        pp.expectation(by_hand, observable, min_abs_coeff=0.0), abs=1e-14, rel=0
    )


def test_openqasm_that_is_not_one_circuit_is_rejected():
    branching = (
        HEADERS[3] + "qubit[2] q;\nbit[2] c;\nh q[0];\nc[0] = measure q[0];\n"
        "if (c[0]) { x q[1]; }\nc[1] = measure q[1];\n"
    )
    with pytest.raises(qsharp.QSharpError, match="dynamic bool"):
        openqasm.compile(branching, target_profile=qsharp.TargetProfile.Base)
    adaptive = openqasm.compile(branching, target_profile=qsharp.TargetProfile.Adaptive_RI)
    with pytest.raises(ValueError, match="branching|not supported|unsupported"):
        pp.circuit_from_qir(str(adaptive))
