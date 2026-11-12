"""The cases of ``test/test_paulirotations.jl`` of PauliPropagation.jl."""

import math
import random

import pytest

import paulipropagation_qdk as pp
from reference_helpers import label, single


def rotated(num_qubits, generator, qubits, theta, paulis, at) -> dict[str, float]:
    circuit = pp.Circuit(num_qubits)
    circuit.rotation(generator, qubits, theta)
    return dict(pp.propagate(circuit, single(num_qubits, paulis, at)).terms())


def expected(num_qubits, *terms) -> dict[str, float]:
    return {label(num_qubits, paulis, qubits): coeff for paulis, qubits, coeff in terms}


@pytest.mark.parametrize("seed", range(10))
def test_rotations_split_the_strings_they_anticommute_with(seed):
    theta = random.Random(seed).gauss(0.0, 1.0)
    cos, sin = math.cos(theta), math.sin(theta)

    def same(result, *terms):
        assert result == pytest.approx(expected(len(next(iter(result))), *terms), abs=1e-15, rel=0)

    # An X rotation of one qubit.
    same(rotated(1, "X", [0], theta, "Z", [0]), ("Z", [0], cos), ("Y", [0], sin))
    same(rotated(1, "X", [0], theta, "Y", [0]), ("Y", [0], cos), ("Z", [0], -sin))
    assert rotated(1, "X", [0], theta, "X", [0]) == {"X": 1.0}

    # A YY rotation of two qubits.
    same(rotated(2, "YY", [0, 1], theta, "Z", [1]), ("Z", [1], cos), ("YX", [0, 1], -sin))
    assert rotated(2, "YY", [0, 1], theta, "Y", [1]) == {"IY": 1.0}
    same(rotated(2, "YY", [0, 1], theta, "X", [1]), ("X", [1], cos), ("YZ", [0, 1], sin))

    # A Z rotation of the middle one of three qubits.
    assert rotated(3, "Z", [1], theta, "Z", [1]) == {"IZI": 1.0}
    same(rotated(3, "Z", [1], theta, "Y", [1]), ("Y", [1], cos), ("X", [1], sin))
    same(rotated(3, "Z", [1], theta, "X", [1]), ("X", [1], cos), ("Y", [1], -sin))
