"""The cases of ``test/test_overlaps.jl`` of PauliPropagation.jl."""

import random

import pytest

import paulipropagation_qdk as pp
from reference_helpers import computational_state, label, pauli_sum, single


@pytest.mark.parametrize(
    "num_qubits, ones, expected",
    [(4, [1, 3], 1.0), (5, [1], -1.0), (4, [2], 1.0), (4, [], 1.0)],
)
def test_overlap_with_a_computational_basis_state(num_qubits, ones, expected):
    identity = single(num_qubits, "I", [num_qubits - 1])
    assert identity.overlap_with_computational(ones) == 1.0

    assert single(num_qubits, "ZZ", [1, 3]).overlap_with_computational(ones) == expected

    assert single(num_qubits, "XZZ", [0, 1, 3]).overlap_with_computational(ones) == 0.0
    assert single(num_qubits, "Y", [3]).overlap_with_computational(ones) == 0.0


def test_scalar_product():
    first = single(4, "IZ", [0, 1], 1.0)
    second = single(4, "XY", [1, 2], -2.0)
    third = single(4, "ZX", [2, 3], 0.2)

    assert first.scalar_product(second) == 0.0
    assert first.scalar_product(third) == 0.0
    assert second.scalar_product(third) == 0.0

    both = pauli_sum(4, {label(4, "IZ", [0, 1]): 1.0, label(4, "XY", [1, 2]): -2.0})
    for kind in (float, complex):
        total = both.astype(kind)
        assert total.scalar_product(third) == third.scalar_product(total) == 0.0
        assert total.scalar_product(first) == first.scalar_product(total) == 1.0
        assert total.scalar_product(second) == second.scalar_product(total) == 4.0
        assert total.scalar_product(both) == both.scalar_product(total) == 5.0


def test_overlap_with_a_state_given_as_a_pauli_sum():
    num_qubits = 3
    observable = pp.PauliSum(num_qubits)
    observable.add("I", [0], 0.2)
    observable.add("X", [1], 0.3)
    observable.add("Z", [2], 0.4)
    observable.add("ZZ", [0, 1], 0.5)

    mixed = single(num_qubits, "I", [0], 1 / 2**num_qubits)
    assert observable.overlap_with_pauli_sum(mixed) == observable.overlap_with_max_mixed() == 0.2
    assert mixed.overlap_with_plus() == 1 / 2**num_qubits

    rng = random.Random(0)
    for _ in range(8):
        bits = [rng.randint(0, 1) for _ in range(num_qubits)]
        ones = [q for q, bit in enumerate(bits) if bit]
        state = computational_state(num_qubits, ones)
        assert observable.overlap_with_pauli_sum(state) == pytest.approx(
            observable.overlap_with_computational(ones), abs=1e-15, rel=0
        )

    zero = computational_state(num_qubits)
    assert observable.overlap_with_pauli_sum(zero) == pytest.approx(1.1, abs=1e-15, rel=0)
    assert observable.overlap_with_zero() == pytest.approx(1.1, abs=1e-15, rel=0)
