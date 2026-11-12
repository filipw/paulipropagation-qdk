"""The cases of ``test/test_truncations.jl`` of PauliPropagation.jl.

They run on the circuit of that file: four layers of the hardware-efficient circuit on
eight qubits, with the angles the Julia package draws from its seed 42.
"""

import json

import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    REFERENCE_DIR,
    bind,
    hardware_efficient_circuit,
    pauli_sum,
    single,
    weight,
)


def reference_case(name: str) -> dict:
    return json.loads((REFERENCE_DIR / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def brick_circuit() -> pp.Circuit:
    case = reference_case("upstream_hardware_efficient_8")
    gates = hardware_efficient_circuit(8, 4)
    assert [(g["paulis"], tuple(g["qubits"])) for g in case["circuit"]] == gates
    return bind(8, gates, [g["theta"] for g in case["circuit"]])


@pytest.fixture(scope="module")
def exact(brick_circuit) -> pp.PauliSum:
    return pp.propagate(brick_circuit, single(8, "Z", [3]), min_abs_coeff=0.0)


def test_weight_damping_without_damping_drops_nothing(brick_circuit, exact):
    start = single(8, "Z", [3])
    damped = pp.propagate(brick_circuit, start, min_abs_coeff=0.0, weight_damping=(0.0, 0.0))
    assert damped.terms() == exact.terms()

    def custom(label, coeff):
        return abs(coeff) * 10 ** (-0.0 * weight(label)) < 0.0

    damped = pp.propagate(brick_circuit, start, min_abs_coeff=0.0, custom_truncation=custom)
    assert damped.terms() == exact.terms()


def test_weight_damping_stays_close_to_the_exact_value(brick_circuit, exact):
    gamma, threshold = 0.01, 1e-5
    case = reference_case("upstream_hardware_efficient_damped_8")
    assert case["weight_damping"] == [gamma, threshold]
    start = single(8, "Z", [3])

    damped = pp.propagate(
        brick_circuit, start, min_abs_coeff=threshold, weight_damping=(gamma, threshold)
    )
    assert len(damped) == case["num_terms"] < len(exact)
    assert damped.overlap_with_zero() == pytest.approx(exact.overlap_with_zero(), rel=1e-3)

    # The same rule as a function, which is how the Julia package takes it.
    def custom(label, coeff):
        return abs(coeff) * 10 ** (-gamma * weight(label)) < threshold

    by_function = pp.propagate(
        brick_circuit, start, min_abs_coeff=threshold, custom_truncation=custom
    )
    assert by_function.terms() == damped.terms()


def test_max_abs_coeff():
    coeffs = {"XIII": 0.1, "IYII": -0.7, "IIZI": 0.3}
    assert pauli_sum(4, coeffs).max_abs_coeff() == pytest.approx(0.7, abs=1e-15)

    complex_coeffs = {"XIII": 0.1 + 0.2j, "IYII": -0.7 - 0.3j, "IIZI": 0.3j}
    assert pauli_sum(4, complex_coeffs).max_abs_coeff() == pytest.approx(
        max(abs(c) for c in complex_coeffs.values()), abs=1e-15
    )
    assert pp.PauliSum(4).max_abs_coeff() == 0.0


def test_relative_truncation(brick_circuit):
    start = single(8, "Z", [3])
    min_abs_coeff = 1e-8

    # A fraction of zero adds nothing to the absolute threshold.
    expected = pp.propagate(brick_circuit, start, min_abs_coeff=min_abs_coeff)
    zero = pp.propagate(brick_circuit, start, min_abs_coeff=min_abs_coeff, min_rel_coeff=0.0)
    assert zero.terms() == expected.terms()

    # A larger fraction can only drop more.
    counts = [len(expected)]
    for min_rel_coeff in (1e-4, 1e-3, 1e-2):
        evolved = pp.propagate(
            brick_circuit, start, min_abs_coeff=min_abs_coeff, min_rel_coeff=min_rel_coeff
        )
        counts.append(len(evolved))
    assert counts == sorted(counts, reverse=True)
    assert len(set(counts)) == len(counts)
    assert counts[-1] == reference_case("upstream_hardware_efficient_relative_8")["num_terms"]


def test_relative_threshold_is_a_fraction_of_the_largest_coefficient():
    start = pauli_sum(1, {"X": 1.0, "Y": 0.1})
    # A channel of strength zero changes nothing, and truncates like any other.
    circuit = pp.Circuit(1)
    circuit.depolarizing(0.0, 0)

    kept = pp.propagate(circuit, start, min_abs_coeff=0.0, min_rel_coeff=0.2)
    assert dict(kept.terms()) == {"X": 1.0}
    kept = pp.propagate(circuit, start, min_abs_coeff=0.0, min_rel_coeff=0.05)
    assert dict(kept.terms()) == {"X": 1.0, "Y": 0.1}
