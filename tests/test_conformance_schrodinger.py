"""The cases of ``test/test_schrodinger.jl`` of PauliPropagation.jl.

With ``heisenberg=False`` a Pauli sum is a state that the circuit acts on, first gate first.
"""

import math
import random

import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    bind,
    count_parameters,
    hardware_efficient_circuit,
    label,
    pauli_sum,
    rectangle_topology,
    single,
    staircase_topology,
    tilted_tfi_trotter_circuit,
)


@pytest.mark.parametrize("seed", range(5))
def test_both_pictures_give_the_same_value(seed):
    rng = random.Random(seed)
    num_qubits = 5
    topology = staircase_topology(num_qubits, periodic=True)

    def depolarize(circuit: pp.Circuit) -> None:
        for q in range(num_qubits):
            circuit.depolarizing(0.1, q)

    first = hardware_efficient_circuit(num_qubits, 2, topology)
    second = tilted_tfi_trotter_circuit(num_qubits, 3, topology)
    thetas1 = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(first))]
    thetas2 = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(second))]

    whole = bind(num_qubits, [*first, depolarize, *second], thetas1 + thetas2)
    at_the_start = single(num_qubits, "Z", [1])
    at_the_end = single(num_qubits, "XY", [0, 2])

    def forward(circuit, state):
        return pp.propagate(circuit, state, heisenberg=False, min_abs_coeff=0.0)

    def backward(circuit, observable):
        return pp.propagate(circuit, observable, heisenberg=True, min_abs_coeff=0.0)

    all_forward = forward(whole, at_the_start).scalar_product(at_the_end)
    all_backward = at_the_start.scalar_product(backward(whole, at_the_end))

    noise = pp.Circuit(num_qubits)
    depolarize(noise)
    half_way = forward(noise, forward(bind(num_qubits, first, thetas1), at_the_start))
    met_in_the_middle = half_way.scalar_product(
        backward(bind(num_qubits, second, thetas2), at_the_end)
    )

    assert abs(all_forward) > 1e-4
    assert all_forward == pytest.approx(all_backward, abs=1e-14, rel=0)
    assert all_forward == pytest.approx(met_in_the_middle, abs=1e-14, rel=0)


@pytest.mark.parametrize("forward_is_heisenberg", [False, True])
def test_the_other_picture_undoes_a_unitary_circuit(forward_is_heisenberg):
    rng = random.Random(int(forward_is_heisenberg))
    num_qubits = 6
    gates = hardware_efficient_circuit(num_qubits, 3, rectangle_topology(3, 2))
    circuit = bind(
        num_qubits, gates, [rng.gauss(0.0, 1.0) for _ in range(count_parameters(gates))]
    )
    text = "".join(rng.choice("XYZ") for _ in range(num_qubits))
    start = pauli_sum(num_qubits, {text: 1.0})

    there = pp.propagate(circuit, start, heisenberg=forward_is_heisenberg, min_abs_coeff=1e-12)
    assert len(there) > 1000
    back = pp.propagate(circuit, there, heisenberg=not forward_is_heisenberg, min_abs_coeff=1e-12)

    back = dict(back.terms())
    assert back.pop(text) == pytest.approx(1.0, abs=1e-9, rel=0)
    assert math.sqrt(sum(coeff**2 for coeff in back.values())) < 1e-9


def random_sum(rng: random.Random, num_qubits: int, count: int = 40) -> pp.PauliSum:
    labels = {"".join(rng.choice("IXYZ") for _ in range(num_qubits)) for _ in range(count)}
    return pauli_sum(num_qubits, {text: rng.uniform(-1.0, 1.0) for text in sorted(labels)})


def test_a_rotation_acts_on_a_state_with_the_opposite_angle():
    rng = random.Random(1)
    for _ in range(20):
        paulis = "".join(rng.choice("IXYZ") for _ in range(4))
        theta = rng.gauss(0.0, 1.0)
        on_state, on_observable = pp.Circuit(4), pp.Circuit(4)
        on_state.rotation(paulis, [0, 1, 2, 3], theta)
        on_observable.rotation(paulis, [0, 1, 2, 3], -theta)

        start = random_sum(rng, 4)
        state = pp.propagate(on_state, start, heisenberg=False, min_abs_coeff=0.0)
        observable = pp.propagate(on_observable, start, heisenberg=True, min_abs_coeff=0.0)
        assert state.terms() == observable.terms()


@pytest.mark.parametrize(
    "name", ["depolarizing", "dephasing", "pauli_x_noise", "pauli_y_noise", "pauli_z_noise"]
)
def test_pauli_noise_acts_the_same_in_both_pictures(name):
    rng = random.Random(name)
    circuit = pp.Circuit(4)
    getattr(circuit, name)(0.1, 2)
    start = random_sum(rng, 4)

    state = pp.propagate(circuit, start, heisenberg=False, min_abs_coeff=0.0)
    observable = pp.propagate(circuit, start, heisenberg=True, min_abs_coeff=0.0)
    assert state.terms() == observable.terms()
    assert state.terms() != start.terms()


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.7, 1.0])
def test_amplitude_damping_acts_on_a_state(gamma):
    """I -> I + gamma Z, X -> sqrt(1 - gamma) X, Y -> sqrt(1 - gamma) Y, Z -> (1 - gamma) Z.

    The Julia package has no action of this channel on a state; these are the values of the
    channel that takes the excited state to the ground state with probability gamma.
    """
    c = 0.37
    circuit = pp.Circuit(3)
    circuit.amplitude_damping(gamma, 1)

    def out(pauli: str) -> dict[str, float]:
        start = single(3, "X" + pauli + "Y", [0, 1, 2], c)
        return dict(pp.propagate(circuit, start, heisenberg=False, min_abs_coeff=0.0).terms())

    def terms(**coeffs) -> dict[str, float]:
        return {label(3, "X" + pauli + "Y", [0, 1, 2]): coeff for pauli, coeff in coeffs.items()}

    assert out("I") == pytest.approx(terms(I=c, Z=c * gamma), abs=1e-15, rel=0)
    assert out("X") == pytest.approx(terms(X=c * math.sqrt(1 - gamma)), abs=1e-15, rel=0)
    assert out("Y") == pytest.approx(terms(Y=c * math.sqrt(1 - gamma)), abs=1e-15, rel=0)
    assert out("Z") == pytest.approx(terms(Z=c * (1 - gamma)), abs=1e-15, rel=0)


def test_a_transfer_map_acts_on_a_state_as_its_transpose():
    rng = random.Random(4)
    matrix = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(4)]
    transposed = [[matrix[j][i] for j in range(4)] for i in range(4)]
    as_given, as_transposed = pp.Circuit(3), pp.Circuit(3)
    as_given.transfer_map([1], matrix)
    as_transposed.transfer_map([1], transposed)

    start = random_sum(rng, 3)
    state = pp.propagate(as_given, start, heisenberg=False, min_abs_coeff=0.0)
    observable = pp.propagate(as_transposed, start, heisenberg=True, min_abs_coeff=0.0)
    assert state.terms() == observable.terms()
    assert state.terms() != pp.propagate(as_given, start, min_abs_coeff=0.0).terms()
