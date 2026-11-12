"""Checks the propagation of states, and of observables with complex coefficients, against
the QDK's state vector.

With ``heisenberg=False`` the all-zero state is propagated forwards through a circuit read
from QIR. The coefficient of a Pauli string in the result, times the dimension, is the
expectation value of that string, which the state vector of the QDK simulator gives directly.
"""

import itertools
import random

import pytest

import paulipropagation_qdk as pp
from qdk_helpers import (
    circuit_source,
    compile_to_qir,
    pauli_expectation,
    random_body,
    state_vector,
)
from reference_helpers import computational_state
from test_noise_against_qdk import BODY, NUM_QUBITS, asymmetric_noise


@pytest.mark.parametrize("seed", range(12))
def test_propagated_states_match_the_qdk_state_vector(seed):
    rng = random.Random(seed)
    num_qubits = rng.randint(3, 5)
    source = circuit_source(random_body(rng, num_qubits, 30))
    circuit = pp.circuit_from_qir(compile_to_qir(source, num_qubits))
    state = state_vector(source, num_qubits)

    # Qubits beyond the register are scratch space of the compiler that ends in the zero state.
    total = circuit.num_qubits
    evolved = pp.propagate(
        circuit, computational_state(total), heisenberg=False, min_abs_coeff=0.0
    )

    register = list(range(num_qubits))
    largest = 0.0
    for paulis in itertools.product("IXYZ", repeat=num_qubits):
        expected = pauli_expectation(state, "".join(paulis), register)
        ours = evolved.coeff("".join(paulis), register) * 2**total
        assert ours == pytest.approx(expected, abs=1e-10, rel=0), paulis
        largest = max(largest, abs(expected)) if "".join(paulis).strip("I") else largest
    assert largest > 1e-3

    # The state stays pure: Tr[rho^2] = 1.
    assert evolved.norm() ** 2 * 2**total == pytest.approx(1.0, abs=1e-10, rel=0)
    assert evolved.overlap_with_max_mixed() * 2**total == pytest.approx(1.0, abs=1e-12, rel=0)


@pytest.mark.parametrize("seed", range(8))
def test_truncated_states_converge_to_the_qdk_state_vector(seed):
    rng = random.Random(seed)
    num_qubits = 6
    body = [f"Ry({rng.uniform(-0.4, 0.4)!r}, qs[{q}]);" for q in range(num_qubits)]
    for _ in range(3):
        body += [f"Rzz({rng.uniform(-0.4, 0.4)!r}, qs[{q}], qs[{q + 1}]);" for q in range(num_qubits - 1)]
        body += [f"Rx({rng.uniform(-0.4, 0.4)!r}, qs[{q}]);" for q in range(num_qubits)]
    source = circuit_source(body)
    circuit = pp.circuit_from_qir(compile_to_qir(source, num_qubits))
    state = state_vector(source, num_qubits)
    expected = pauli_expectation(state, "ZZ", [2, 3])

    start = computational_state(num_qubits)
    errors = []
    for threshold in (1e-3, 1e-5, 1e-7, 0.0):
        evolved = pp.propagate(circuit, start, heisenberg=False, min_abs_coeff=threshold)
        errors.append(abs(evolved.coeff("ZZ", [2, 3]) * 2**num_qubits - expected))
    assert errors[-1] < 1e-10
    assert errors[0] < 5e-2 and errors[1] < 5e-4 and errors[2] < 5e-6
    assert errors[0] > errors[2]


@pytest.mark.parametrize("seed", range(8))
def test_observables_with_complex_coefficients_match_the_qdk_state_vector(seed):
    rng = random.Random(100 + seed)
    num_qubits = rng.randint(3, 6)
    source = circuit_source(random_body(rng, num_qubits, 40))
    circuit = pp.circuit_from_qir(compile_to_qir(source, num_qubits))
    state = state_vector(source, num_qubits)

    observable = pp.PauliSum(circuit.num_qubits, dtype=complex)
    expected = 0.0
    for _ in range(10):
        qubits = rng.sample(range(num_qubits), rng.randint(1, num_qubits))
        paulis = "".join(rng.choice("XYZ") for _ in qubits)
        coeff = complex(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        observable.add(paulis, qubits, coeff)
        expected += coeff * pauli_expectation(state, paulis, qubits)

    value = pp.expectation(circuit, observable, min_abs_coeff=0.0)
    assert isinstance(value, complex)
    assert abs(expected.imag) > 1e-4
    assert value == pytest.approx(expected, abs=1e-10, rel=0)


def test_the_lowering_operator_has_a_complex_expectation_value():
    """(X + iY) / 2 on one qubit is |0><1|, whose expectation value is not real."""
    source = circuit_source(["Ry(0.9, qs[0]);", "Rz(0.7, qs[0]);", "CNOT(qs[0], qs[1]);", "Rx(0.4, qs[1]);"])
    circuit = pp.circuit_from_qir(compile_to_qir(source, 2))
    state = state_vector(source, 2)

    for q in (0, 1):
        lowering = pp.PauliSum(2, dtype=complex)
        lowering.add("X", [q], 0.5)
        lowering.add("Y", [q], 0.5j)
        # <psi| (|0><1| on qubit q) |psi> from the amplitudes.
        moved = state if q == 0 else state.T
        expected = (moved[0].conj() * moved[1]).sum()
        assert pp.expectation(circuit, lowering, min_abs_coeff=0.0) == pytest.approx(
            expected, abs=1e-12, rel=0
        )
    assert abs(expected.imag) > 1e-2


def test_a_noisy_state_gives_the_expectation_values_of_the_noisy_circuit():
    """Propagating the state forwards and the observable backwards are the same trace."""
    qir = compile_to_qir(circuit_source(BODY), NUM_QUBITS)
    circuit = pp.circuit_from_qir(qir, noise=asymmetric_noise())
    assert circuit.num_noise_channels > 20
    total = circuit.num_qubits

    evolved = pp.propagate(
        circuit, computational_state(total), heisenberg=False, min_abs_coeff=0.0
    )
    # Noise leaves the trace and lowers the purity.
    assert evolved.overlap_with_max_mixed() * 2**total == pytest.approx(1.0, abs=1e-12, rel=0)
    assert evolved.norm() ** 2 * 2**total < 0.9

    register = list(range(NUM_QUBITS))
    checked = 0
    for paulis in itertools.product("IXYZ", repeat=NUM_QUBITS):
        observable = pp.PauliSum(total)
        observable.add("".join(paulis), register)
        backwards = pp.expectation(circuit, observable, min_abs_coeff=0.0)
        forwards = evolved.coeff("".join(paulis), register) * 2**total
        assert forwards == pytest.approx(backwards, abs=1e-12, rel=0), paulis
        checked += abs(backwards) > 1e-3
    assert checked > 20
