"""The cases of ``test/test_noisechannels.jl`` of PauliPropagation.jl."""

import random

import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    bind,
    bricklayer_topology,
    count_parameters,
    hardware_efficient_circuit,
    single,
)

NOISE = ["depolarizing", "dephasing", "pauli_x_noise", "pauli_y_noise", "pauli_z_noise"]

# The Pauli operators whose coefficient each channel multiplies by one minus its strength.
DAMPED = {
    "depolarizing": "XYZ",
    "dephasing": "XY",
    "pauli_x_noise": "YZ",
    "pauli_y_noise": "XZ",
    "pauli_z_noise": "XY",
}


def damping(pauli: str, strength: float, q: int):
    """A map that multiplies the coefficient of one Pauli operator on a qubit."""
    diagonal = [1.0 if other != pauli else 1.0 - strength for other in "IXYZ"]
    matrix = [[diagonal[i] if i == j else 0.0 for j in range(4)] for i in range(4)]
    return lambda circuit: circuit.transfer_map([q], matrix)


def channel_and_dampings(name: str, seed: int) -> tuple[float, float, float]:
    """The value of a random circuit with a channel, with dampings instead, and without."""
    rng = random.Random(f"{name}-{seed}")
    num_qubits = rng.randint(2, 5)
    gates = hardware_efficient_circuit(
        num_qubits, 2, bricklayer_topology(num_qubits, periodic=True)
    )
    thetas = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(gates))]
    start = single(num_qubits, [rng.choice("XYZ") for _ in range(num_qubits)], range(num_qubits))

    where = rng.randrange(len(gates))
    q = rng.randrange(num_qubits)
    strength = rng.uniform(0.0, 0.2)

    with_channel = list(gates)
    with_channel.insert(where, lambda circuit: getattr(circuit, name)(strength, q))
    with_dampings = list(gates)
    for pauli in DAMPED[name]:
        with_dampings.insert(where, damping(pauli, strength, q))

    def value(layers) -> float:
        circuit = bind(num_qubits, layers, thetas)
        return pp.propagate(circuit, start, min_abs_coeff=0.0).overlap_with_zero()

    return value(with_channel), value(with_dampings), value(gates)


@pytest.mark.parametrize("name", NOISE)
def test_pauli_noise_is_a_damping_of_each_operator_it_does_not_commute_with(name):
    changed = 0
    for seed in range(12):
        with_channel, with_dampings, without = channel_and_dampings(name, seed)
        assert with_channel == pytest.approx(with_dampings, abs=1e-14, rel=0), seed
        changed += abs(with_channel - without) > 1e-6
    # Where the channel lands it may not reach the observable, but mostly it does.
    assert changed >= 6


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.7, 1.0])
def test_amplitude_damping_values(gamma):
    """I -> I, X -> sqrt(1 - gamma) X, Y -> sqrt(1 - gamma) Y, Z -> (1 - gamma) Z + gamma I."""
    c = 0.37
    circuit = pp.Circuit(1)
    circuit.amplitude_damping(gamma, 0)

    def out(pauli: str) -> dict[str, float]:
        return dict(pp.propagate(circuit, single(1, pauli, [0], c), min_abs_coeff=0.0).terms())

    assert out("I") == {"I": c}
    for pauli in "XY":
        assert out(pauli) == pytest.approx({pauli: c * (1 - gamma) ** 0.5}, abs=1e-15, rel=0)
    assert out("Z") == pytest.approx({"Z": c * (1 - gamma), "I": c * gamma}, abs=1e-15, rel=0)


def test_amplitude_damping_leaves_other_qubits_alone():
    gamma = 0.4
    circuit = pp.Circuit(3)
    circuit.amplitude_damping(gamma, 1)

    out = pp.propagate(circuit, single(3, "XZY", [0, 1, 2], 0.5), min_abs_coeff=0.0)
    assert dict(out.terms()) == pytest.approx(
        {"XZY": 0.5 * (1 - gamma), "XIY": 0.5 * gamma}, abs=1e-15, rel=0
    )


@pytest.mark.parametrize("name", NOISE)
@pytest.mark.parametrize("strength", [0.0, 0.25, 0.6, 1.0])
def test_pauli_noise_values(name, strength):
    c = 0.37
    circuit = pp.Circuit(1)
    getattr(circuit, name)(strength, 0)
    for pauli in "IXYZ":
        expected = c * (1 - strength) if pauli in DAMPED[name] else c
        out = pp.propagate(circuit, single(1, pauli, [0], c), min_abs_coeff=0.0)
        assert dict(out.terms()) == pytest.approx({pauli: expected}, abs=1e-15, rel=0)


@pytest.mark.parametrize("name", [*NOISE, "amplitude_damping"])
def test_noise_strength_is_validated(name):
    circuit = pp.Circuit(2)
    for strength in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            getattr(circuit, name)(strength, 0)
    assert len(circuit) == 0

    # The ends of the range are valid.
    for strength in (0.0, 1.0):
        getattr(circuit, name)(strength, 0)
    assert len(circuit) == 2
    pp.propagate(circuit, single(2, "X", [0]))


@pytest.mark.parametrize("name", [*NOISE, "amplitude_damping"])
def test_noise_qubit_is_validated(name):
    circuit = pp.Circuit(3)
    with pytest.raises(OverflowError):
        getattr(circuit, name)(0.1, -1)
    with pytest.raises(ValueError, match="out of range"):
        getattr(circuit, name)(0.1, 3)
    assert len(circuit) == 0
