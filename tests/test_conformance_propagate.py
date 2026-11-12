"""The cases of ``test/test_propagate.jl`` of PauliPropagation.jl.

Where that file compares the backends of the Julia package with each other, the tests here
compare the engine with ``reference_propagate``, which follows the same rules on labels.
"""

import json
import random

import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    REFERENCE_DIR,
    bind,
    count_parameters,
    efficient_su2_circuit,
    hardware_efficient_circuit,
    label,
    pauli_sum,
    rectangle_topology,
    reference_propagate,
    single,
    tfi_trotter_circuit,
)

# No rule that counts rotations reaches this, but asking for one makes the engine count.
NEVER = 10**6


def test_propagation_with_a_known_value():
    """The Julia package states 0.21720058439757214, cross-validated with other libraries."""
    case = json.loads((REFERENCE_DIR / "upstream_hardware_efficient_8.json").read_text())
    gates = hardware_efficient_circuit(8, 4)
    assert [(g["paulis"], tuple(g["qubits"])) for g in case["circuit"]] == gates
    circuit = bind(8, gates, [g["theta"] for g in case["circuit"]])
    start = single(8, "Z", [3])

    evolved = pp.propagate(circuit, start, min_abs_coeff=0.0)
    assert evolved.overlap_with_zero() == pytest.approx(0.21720058439757214, abs=1e-13, rel=0)
    assert len(evolved) == case["num_terms"]

    counted = pp.propagate(circuit, start, min_abs_coeff=0.0, max_freq=NEVER)
    assert counted.terms() == evolved.terms()


def random_start(rng: random.Random, num_qubits: int, paulis: str) -> dict[str, float]:
    return {label(num_qubits, rng.choice(paulis), [rng.randrange(num_qubits)]): 1.0}


def assert_same_terms(evolved: pp.PauliSum, reference: dict[str, float]) -> None:
    ours = dict(evolved.terms())
    assert ours.keys() == reference.keys()
    assert ours == pytest.approx(reference, abs=1e-14, rel=0)


@pytest.fixture(scope="module")
def su2():
    return efficient_su2_circuit(6, 3, rectangle_topology(2, 3, periodic=True))


@pytest.mark.parametrize("max_weight", [0, 1, 3, 6])
def test_weight_truncation(su2, max_weight):
    rng = random.Random(max_weight)
    thetas = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(su2))]
    terms = random_start(rng, 6, "IXYZ")
    circuit = bind(6, su2, thetas)

    reference = reference_propagate(6, su2, thetas, terms, max_weight=max_weight)
    evolved = pp.propagate(circuit, pauli_sum(6, terms), min_abs_coeff=0.0, max_weight=max_weight)
    assert_same_terms(evolved, reference)

    counted = pp.propagate(
        circuit, pauli_sum(6, terms), min_abs_coeff=0.0, max_weight=max_weight, max_freq=NEVER
    )
    assert counted.terms() == evolved.terms()

    # The same start on registers wider than one, two and four machine words.
    for width in (100, 200, 300):
        wide_terms = {text + "I" * (width - 6): coeff for text, coeff in terms.items()}
        wide = pp.propagate(
            circuit, pauli_sum(width, wide_terms), min_abs_coeff=0.0, max_weight=max_weight
        )
        assert wide.terms() == [(text + "I" * (width - 6), c) for text, c in evolved.terms()]
        assert wide.overlap_with_zero() == evolved.overlap_with_zero()
        assert wide.overlap_with_plus() == evolved.overlap_with_plus()


@pytest.mark.parametrize("max_freq", [0, 2, 5, 10])
def test_frequency_truncation(su2, max_freq):
    rng = random.Random(100 + max_freq)
    thetas = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(su2))]
    terms = random_start(rng, 6, "XYZ")

    reference = reference_propagate(6, su2, thetas, terms, max_freq=max_freq)
    evolved = pp.propagate(
        bind(6, su2, thetas), pauli_sum(6, terms), min_abs_coeff=0.0, max_freq=max_freq
    )
    assert_same_terms(evolved, reference)
    assert len(reference) < len(reference_propagate(6, su2, thetas, terms))


@pytest.mark.parametrize("max_sins", [0, 2, 5, 10])
def test_sine_truncation(su2, max_sins):
    rng = random.Random(200 + max_sins)
    thetas = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(su2))]
    terms = random_start(rng, 6, "XYZ")

    reference = reference_propagate(6, su2, thetas, terms, max_sins=max_sins)
    evolved = pp.propagate(
        bind(6, su2, thetas), pauli_sum(6, terms), min_abs_coeff=0.0, max_sins=max_sins
    )
    assert_same_terms(evolved, reference)
    if max_sins < 10:
        assert len(reference) < len(reference_propagate(6, su2, thetas, terms))


@pytest.mark.parametrize("min_abs_coeff", [1e-3, 1e-2, 1e-1])
def test_coefficient_truncation(min_abs_coeff):
    gates = efficient_su2_circuit(6, 5, rectangle_topology(2, 3, periodic=True))
    rng = random.Random(str(min_abs_coeff))
    thetas = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(gates))]
    terms = random_start(rng, 6, "XYZ")
    circuit = bind(6, gates, thetas)

    reference = reference_propagate(6, gates, thetas, terms, min_abs_coeff=min_abs_coeff)
    evolved = pp.propagate(circuit, pauli_sum(6, terms), min_abs_coeff=min_abs_coeff)
    assert_same_terms(evolved, reference)
    assert len(reference) < len(reference_propagate(6, gates, thetas, terms))

    counted = pp.propagate(
        circuit, pauli_sum(6, terms), min_abs_coeff=min_abs_coeff, max_sins=NEVER
    )
    assert counted.terms() == evolved.terms()


def test_complex_coefficients():
    gates = hardware_efficient_circuit(4, 2)
    rng = random.Random(9)
    circuit = bind(4, gates, [rng.gauss(0.0, 1.0) for _ in range(count_parameters(gates))])

    # Propagation is linear in the coefficient it starts with.
    z = 1.0 + 0.5j
    real = pp.propagate(circuit, single(4, "Z", [1], 1.0), min_abs_coeff=0.0)
    scaled = pp.propagate(circuit, single(4, "Z", [1], z), min_abs_coeff=0.0)
    assert scaled.dtype is complex
    assert abs(real.overlap_with_zero()) > 1e-3
    assert scaled.overlap_with_zero() == pytest.approx(z * real.overlap_with_zero(), abs=1e-14)
    assert dict(scaled.terms()) == pytest.approx(
        {text: z * coeff for text, coeff in real.terms()}, abs=1e-15, rel=0
    )

    # The largest absolute value of complex coefficients sets the relative threshold.
    truncated = pp.propagate(circuit, single(4, "Z", [1], z), min_rel_coeff=0.01)
    assert 0 < len(truncated) < len(scaled)


def test_counting_truncations_need_no_preparation():
    """A plain sum takes `max_freq` and `max_sins`, and the result is a plain sum."""
    gates = tfi_trotter_circuit(4, 4)
    rng = random.Random(11)
    thetas = [rng.gauss(0.0, 1.0) for _ in range(count_parameters(gates))]
    terms = {label(4, "Z", [1]): 1.0}

    exact = reference_propagate(4, gates, thetas, terms)
    sizes = []
    for max_freq, max_sins in [(4, 2), (8, 3), (12, 2)]:
        evolved = pp.propagate(
            bind(4, gates, thetas), pauli_sum(4, terms), max_freq=max_freq, max_sins=max_sins
        )
        reference = reference_propagate(
            4, gates, thetas, terms, min_abs_coeff=1e-10, max_freq=max_freq, max_sins=max_sins
        )
        assert_same_terms(evolved, reference)
        assert evolved.dtype is float
        sizes.append(len(reference))
    assert sizes[0] < sizes[1] < len(exact)
    assert 1 < sizes[2] < len(exact)
