"""Checks the whole path from Q# against the QDK's own simulator.

One Q# operation goes two ways: compiled to QIR, read by ``circuit_from_qir`` and propagated
without truncation, and run on the QDK simulator, whose state vector gives the expectation
value directly. The two share only the Q# source, so they agree only if the QIR reader, the
qubit order, the gate conventions and the propagation rules are all right.
"""

import random

import pyqir
import pytest
from qdk import qsharp

import paulipropagation_qdk as pp
from qdk_helpers import (
    circuit_source,
    compile_to_qir,
    pauli_expectation,
    random_body,
    state_vector,
)


def random_paulis(rng: random.Random, num_qubits: int) -> tuple[str, list[int]]:
    qubits = rng.sample(range(num_qubits), rng.randint(1, num_qubits))
    return "".join(rng.choice("XYZ") for _ in qubits), qubits


@pytest.mark.parametrize("seed", range(24))
def test_random_qsharp_circuits_match_the_qdk_state_vector(seed):
    rng = random.Random(seed)
    num_qubits = rng.randint(3, 7)
    source = circuit_source(random_body(rng, num_qubits, 40))

    circuit = pp.circuit_from_qir(compile_to_qir(source, num_qubits))
    state = state_vector(source, num_qubits)

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

    # A sum of Pauli strings propagates as one observable.
    assert pp.expectation(circuit, total, min_abs_coeff=0.0) == pytest.approx(
        expected_total, abs=1e-10, rel=0
    )


def ising_body(num_qubits: int, layers: int, dt: float) -> list[str]:
    return [
        f"for _ in 1..{layers} {{",
        f"    for i in 0..{num_qubits - 2} {{ Rzz({dt!r}, qs[i], qs[i + 1]); }}",
        f"    for i in 0..{num_qubits - 1} {{ Rx({dt!r}, qs[i]); }}",
        "}",
    ]


def test_truncated_results_converge_to_the_qdk_state_vector():
    num_qubits = 10
    source = circuit_source(ising_body(num_qubits, layers=6, dt=0.4))
    circuit = pp.circuit_from_qir(compile_to_qir(source, num_qubits))
    expected = pauli_expectation(state_vector(source, num_qubits), "Z", [4])

    observable = pp.PauliSum(num_qubits)
    observable.add("Z", [4])

    def error(**truncation) -> tuple[float, int]:
        evolved = pp.propagate(circuit, observable, **truncation)
        return abs(evolved.overlap_with_zero() - expected), len(evolved)

    exact_error, exact_terms = error(min_abs_coeff=0.0)
    assert exact_error < 1e-10

    previous_terms = 0
    for threshold, bound in [(1e-2, 5e-2), (1e-3, 5e-3), (1e-4, 5e-4), (1e-6, 5e-6)]:
        err, terms = error(min_abs_coeff=threshold)
        assert err < bound, threshold
        assert previous_terms < terms <= exact_terms
        previous_terms = terms

    # Weight truncation is exact once it allows every weight.
    assert error(max_weight=num_qubits, min_abs_coeff=0.0)[0] < 1e-10
    assert error(max_weight=3, min_abs_coeff=0.0)[1] < exact_terms


def test_readme_example_of_the_julia_package_from_qsharp():
    """The 32-qubit example of PauliPropagation.jl, written in Q# and read from QIR."""
    num_qubits, layers, dt = 32, 32, 0.1
    # The gate order of `tfitrottercircuit` on a periodic bricklayer topology.
    zz_layer = [
        f"for i in 0..2..{num_qubits - 2} {{ Rzz({dt}, qs[i], qs[i + 1]); }}",
        f"for i in 1..2..{num_qubits - 3} {{ Rzz({dt}, qs[i], qs[i + 1]); }}",
        f"Rzz({dt}, qs[{num_qubits - 1}], qs[0]);",
    ]
    x_layer = [f"for i in 0..{num_qubits - 1} {{ Rx({dt}, qs[i]); }}"]
    body = [*zz_layer, f"for _ in 1..{layers - 1} {{", *x_layer, *zz_layer, "}", *x_layer]

    circuit = pp.circuit_from_qir(compile_to_qir(circuit_source(body), num_qubits))
    assert (circuit.num_qubits, len(circuit)) == (32, 2048)

    observable = pp.PauliSum(num_qubits)
    observable.add("Z", [15])
    evolved = pp.propagate(circuit, observable, max_weight=6, min_abs_coeff=1e-4)
    assert len(evolved) == 1671
    assert evolved.overlap_with_zero() == pytest.approx(0.15459672824142454, abs=1e-12, rel=0)


def test_openqasm_compiled_by_the_qdk_matches_the_qdk_state_vector():
    from qdk import openqasm

    qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[4] q;
        bit[4] c;
        h q[0];
        cx q[0], q[1];
        rx(0.37) q[2];
        ry(-1.2) q[3];
        cz q[1], q[2];
        t q[1];
        sdg q[2];
        swap q[0], q[3];
        ccx q[0], q[1], q[2];
        rz(0.81) q[2];
        cy q[3], q[1];
        sx q[0];
        crz(0.6) q[2], q[3];
        c = measure q;
    """
    qir = openqasm.compile(qasm, target_profile=qsharp.TargetProfile.Base)
    circuit = pp.circuit_from_qir(str(qir))
    assert circuit.num_qubits == 4

    # The same circuit in Q#, for the state vector.
    source = circuit_source(
        [
            "H(qs[0]);",
            "CNOT(qs[0], qs[1]);",
            "Rx(0.37, qs[2]);",
            "Ry(-1.2, qs[3]);",
            "CZ(qs[1], qs[2]);",
            "T(qs[1]);",
            "Adjoint S(qs[2]);",
            "SWAP(qs[0], qs[3]);",
            "CCNOT(qs[0], qs[1], qs[2]);",
            "Rz(0.81, qs[2]);",
            "CY(qs[3], qs[1]);",
            "SX(qs[0]);",
            "Controlled Rz([qs[2]], (0.6, qs[3]));",
        ]
    )
    state = state_vector(source, 4)

    rng = random.Random(0)
    for _ in range(20):
        paulis, qubits = random_paulis(rng, 4)
        observable = pp.PauliSum(4)
        observable.add(paulis, qubits)
        assert pp.expectation(circuit, observable, min_abs_coeff=0.0) == pytest.approx(
            pauli_expectation(state, paulis, qubits), abs=1e-10, rel=0
        ), (paulis, qubits)


def test_qir_bitcode_reads_like_qir_text():
    source = circuit_source(["H(qs[0]);", "Rzz(0.3, qs[0], qs[1]);", "T(qs[1]);"])
    text = compile_to_qir(source, 2)
    bitcode = pyqir.Module.from_ir(pyqir.Context(), text).bitcode

    observable = pp.PauliSum(2)
    observable.add("XZ", [0, 1])
    from_text = pp.propagate(pp.circuit_from_qir(text), observable)
    from_bitcode = pp.propagate(pp.circuit_from_qir(bitcode), observable)
    assert len(pp.circuit_from_qir(text)) == 3
    assert from_text.terms() == from_bitcode.terms()


def test_gate_after_a_measurement_is_rejected():
    lines = compile_to_qir(circuit_source(["H(qs[0]);"]), 1).splitlines()
    gate = next(line for line in lines if "call" in line and "__quantum__qis__h__body" in line)
    measurement = next(
        i for i, line in enumerate(lines) if "call" in line and "__quantum__qis__m__body" in line
    )
    lines.insert(measurement + 1, gate)
    with pytest.raises(ValueError, match="after it was measured or reset"):
        pp.circuit_from_qir("\n".join(lines))


def test_measurement_that_the_qdk_defers_reads_as_a_wider_circuit():
    """For the base profile the QDK moves a mid-circuit measurement to the end.

    The qubit continues on a fresh one, so the QIR stays one circuit followed by
    measurements, but its qubits no longer number the Q# register.
    """
    qsharp.init(target_profile=qsharp.TargetProfile.Base)
    qsharp.eval(
        "operation Main() : Result[] { use qs = Qubit[2]; H(qs[0]); let r = M(qs[0]); "
        "CNOT(qs[0], qs[1]); [r, M(qs[1])] }"
    )
    circuit = pp.circuit_from_qir(str(qsharp.compile("Main()")))
    assert circuit.num_qubits == 3

    # The two results are random and equal: the qubits measured into them are 0 and 1.
    for paulis, qubits, expected in [("Z", [0], 0.0), ("Z", [1], 0.0), ("ZZ", [0, 1], 1.0)]:
        observable = pp.PauliSum(3)
        observable.add(paulis, qubits)
        assert pp.expectation(circuit, observable) == pytest.approx(expected, abs=1e-12)


def test_branching_on_a_measurement_is_rejected():
    qsharp.init(target_profile=qsharp.TargetProfile.Adaptive_RI)
    qsharp.eval(
        "operation Main() : Result { use qs = Qubit[2]; H(qs[0]); "
        "if M(qs[0]) == One { X(qs[1]); } M(qs[1]) }"
    )
    with pytest.raises(ValueError, match="branching|after it was measured"):
        pp.circuit_from_qir(str(qsharp.compile("Main()")))


def test_unknown_gate_is_rejected():
    text = compile_to_qir(circuit_source(["H(qs[0]);"]), 1)
    assert "__quantum__qis__h__body" in text
    with pytest.raises(ValueError, match="unsupported gate"):
        pp.circuit_from_qir(text.replace("__quantum__qis__h__body", "__quantum__qis__u3__body"))
