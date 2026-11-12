"""Checks noise read from a QDK ``NoiseConfig`` against the QDK's noisy simulators.

The simulators draw random Pauli errors shot by shot. Averaged over many shots, the
measurement results give the expectation values of products of Z, which have to match the
values that propagation through the noisy circuit gives exactly.
"""

from collections import Counter
from itertools import combinations

import pytest
from qdk.simulation import NoiseConfig, run_qir

import paulipropagation_qdk as pp
from qdk_helpers import circuit_source, compile_to_qir

NUM_QUBITS = 4
SHOTS = 1_000_000
# An average of `SHOTS` values of +1 or -1 has a standard deviation of at most 1 / sqrt(SHOTS).
TOLERANCE = 5 / SHOTS**0.5

# Clifford gates, rotations and a Toffoli gate, so that the expectation values are generic
# and every noise table below is used.
BODY = [
    "H(qs[0]);",
    "Rx(0.7, qs[1]);",
    "CNOT(qs[0], qs[1]);",
    "Ry(-0.9, qs[2]);",
    "S(qs[1]);",
    "CZ(qs[1], qs[2]);",
    "Rzz(0.6, qs[2], qs[3]);",
    "T(qs[3]);",
    "H(qs[3]);",
    "CY(qs[3], qs[0]);",
    "SX(qs[0]);",
    "Rz(1.1, qs[0]);",
    "CCNOT(qs[0], qs[1], qs[2]);",
    "SWAP(qs[1], qs[3]);",
    "Rxx(-0.5, qs[0], qs[2]);",
    "Adjoint T(qs[1]);",
    "Adjoint S(qs[2]);",
    "Ryy(0.8, qs[1], qs[3]);",
    "X(qs[0]);",
    "Y(qs[1]);",
    "Z(qs[2]);",
    "H(qs[1]);",
    "Rx(0.4, qs[2]);",
]


def asymmetric_noise() -> NoiseConfig:
    """Errors that tell the two qubits of a gate apart, on every kind of gate in ``BODY``."""
    noise = NoiseConfig()
    noise.h.set_pauli_noise([("Y", 0.02), ("Z", 0.01)])
    noise.x.set_pauli_noise("Z", 0.03)
    noise.y.set_pauli_noise("X", 0.03)
    noise.z.set_pauli_noise("Y", 0.03)
    noise.s.set_pauli_noise("X", 0.04)
    noise.s_adj.set_pauli_noise("Y", 0.04)
    noise.sx.set_pauli_noise("Z", 0.05)
    noise.t.set_pauli_noise("X", 0.02)
    noise.t_adj.set_pauli_noise("Y", 0.02)
    noise.rx.set_pauli_noise([("Z", 0.03), ("Y", 0.01)])
    noise.ry.set_pauli_noise("X", 0.04)
    noise.rz.set_depolarizing(0.06)
    noise.cx.set_pauli_noise([("XI", 0.06), ("IZ", 0.04), ("YZ", 0.03)])
    noise.cy.set_pauli_noise([("ZX", 0.05), ("IY", 0.02)])
    noise.cz.set_pauli_noise([("XI", 0.02), ("ZY", 0.01)])
    noise.swap.set_pauli_noise([("XI", 0.05), ("ZY", 0.03)])
    noise.rxx.set_pauli_noise([("IZ", 0.05), ("YX", 0.02)])
    noise.ryy.set_pauli_noise([("XI", 0.04), ("ZZ", 0.02)])
    noise.rzz.set_pauli_noise([("IX", 0.06), ("YI", 0.03)])
    return noise


class Reversed:
    """A noise table, or a set of them, with the characters of every error string reversed."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        value = getattr(self._inner, name)
        if isinstance(value, float):
            return getattr(self._inner, name[::-1])
        return Reversed(value)


def z_products() -> list[tuple[int, ...]]:
    return [
        qubits
        for size in range(1, NUM_QUBITS + 1)
        for qubits in combinations(range(NUM_QUBITS), size)
    ]


def propagated(circuit: pp.Circuit) -> dict[tuple[int, ...], float]:
    values = {}
    for qubits in z_products():
        observable = pp.PauliSum(circuit.num_qubits)
        observable.add("Z" * len(qubits), list(qubits))
        values[qubits] = pp.expectation(circuit, observable, min_abs_coeff=0.0)
    return values


def sampled(qir: str, noise, simulator: str) -> dict[tuple[int, ...], float]:
    shots = run_qir(qir, shots=SHOTS, noise=noise, seed=2024, type=simulator)
    assert len(shots) == SHOTS
    counts = Counter(tuple(str(result) for result in shot) for shot in shots)
    assert set().union(*counts) <= {"Zero", "One"}

    def mean(qubits: tuple[int, ...]) -> float:
        total = 0
        for outcome, count in counts.items():
            ones = sum(outcome[q] == "One" for q in qubits)
            total += -count if ones % 2 else count
        return total / SHOTS

    return {qubits: mean(qubits) for qubits in z_products()}


@pytest.fixture(scope="module")
def qir() -> str:
    return compile_to_qir(circuit_source(BODY), NUM_QUBITS)


@pytest.mark.parametrize("simulator", ["clifford", "cpu"])
def test_noisy_expectation_values_match_qdk_shots(qir, simulator):
    noise = asymmetric_noise()
    circuit = pp.circuit_from_qir(qir, noise=noise)
    assert circuit.num_qubits == NUM_QUBITS
    assert circuit.num_noise_channels > len(BODY)

    exact = propagated(circuit)
    shots = sampled(qir, noise, simulator)
    for qubits in z_products():
        assert exact[qubits] == pytest.approx(shots[qubits], abs=TOLERANCE, rel=0), qubits


def test_the_comparison_with_shots_tells_wrong_noise_apart(qir):
    """Without noise, or with the two qubits of each error swapped, the values are far off."""
    noise = asymmetric_noise()
    exact = propagated(pp.circuit_from_qir(qir, noise=noise))

    def largest_difference(circuit: pp.Circuit) -> float:
        other = propagated(circuit)
        return max(abs(exact[qubits] - other[qubits]) for qubits in z_products())

    assert largest_difference(pp.circuit_from_qir(qir)) > 10 * TOLERANCE
    assert largest_difference(pp.circuit_from_qir(qir, noise=Reversed(noise))) > 3 * TOLERANCE


def test_noiseless_expectation_values_match_qdk_shots(qir):
    circuit = pp.circuit_from_qir(qir, noise=NoiseConfig())
    assert circuit.num_noise_channels == 0
    exact = propagated(circuit)
    shots = sampled(qir, None, "cpu")
    for qubits in z_products():
        assert exact[qubits] == pytest.approx(shots[qubits], abs=TOLERANCE, rel=0), qubits


def test_noise_on_measurements_is_dropped(qir):
    """The QDK applies it after recording the result, so it does not change the results."""
    noise = NoiseConfig()
    noise.mz.set_bitflip(0.2)
    noise.mresetz.set_bitflip(0.2)
    assert pp.circuit_from_qir(qir, noise=noise).num_noise_channels == 0

    exact = propagated(pp.circuit_from_qir(qir, noise=noise))
    shots = sampled(qir, noise, "cpu")
    for qubits in z_products():
        assert exact[qubits] == pytest.approx(shots[qubits], abs=TOLERANCE, rel=0), qubits


def test_qubit_loss_is_rejected(qir):
    noise = NoiseConfig()
    noise.cx.set_pauli_noise([("XL", 0.01)])
    with pytest.raises(ValueError, match="qubit loss"):
        pp.circuit_from_qir(qir, noise=noise)

    noise = NoiseConfig()
    noise.rx.loss = 0.01
    with pytest.raises(ValueError, match="qubit loss"):
        pp.circuit_from_qir(qir, noise=noise)
