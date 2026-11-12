"""Compares propagated Pauli sums, term by term, with results from PauliPropagation.jl.

The reference files in ``tests/reference`` are written by ``tools/julia/generate_reference.jl``.
They hold circuits built by the Julia package itself, so the gates, their order, and the
truncation after every gate have to agree for the terms to match.
"""

import json
import math
from pathlib import Path

import pytest

import paulipropagation_qdk as pp

REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = sorted(path.stem for path in REFERENCE_DIR.glob("*.json"))

# Clifford gates of PauliPropagation.jl that this package has under another name.
SAME_GATE = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "SX": "sx",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
}

# Noise channels of PauliPropagation.jl, which take a strength and a qubit.
NOISE = {
    "DepolarizingNoise": "depolarizing",
    "PauliXNoise": "pauli_x_noise",
    "PauliYNoise": "pauli_y_noise",
    "PauliZNoise": "pauli_z_noise",
    "AmplitudeDampingNoise": "amplitude_damping",
}

# Coefficients agree to rounding. The slack is for the two reference gates that are
# rotations here: a rotation by pi/2 leaves cos(pi/2) = 6e-17 where a Clifford gate leaves 0.
COEFF_TOL = 1e-13


def build_circuit(case) -> pp.Circuit:
    circuit = pp.Circuit(case["num_qubits"])
    for gate in case["circuit"]:
        kind, qubits = gate["gate"], gate["qubits"]
        if kind == "rotation":
            circuit.rotation(gate["paulis"], qubits, gate["theta"])
        elif kind in NOISE:
            getattr(circuit, NOISE[kind])(gate["strength"], qubits[0])
        elif kind == "SY":
            # The square root of Y is exp(-i pi/4 Y) up to a phase.
            circuit.ry(math.pi / 2, qubits[0])
        elif kind == "ZZpihalf":
            # The reference table for this gate is that of exp(+i pi/4 ZZ).
            circuit.rzz(-math.pi / 2, qubits[0], qubits[1])
        else:
            getattr(circuit, SAME_GATE[kind])(*qubits)
    return circuit


def test_reference_cases_exist():
    assert "readme_tfi_32" in CASES


@pytest.mark.parametrize("name", CASES)
def test_terms_match_the_julia_reference(name):
    case = json.loads((REFERENCE_DIR / f"{name}.json").read_text())

    observable = pp.PauliSum(case["num_qubits"])
    for term in case["observable"]:
        observable.add(term["paulis"], term["qubits"], term["coeff"])

    result = pp.propagate(
        build_circuit(case),
        observable,
        max_weight=case["max_weight"],
        min_abs_coeff=case["min_abs_coeff"],
    )

    ours = dict(result.terms())
    theirs = {label: coeff for label, coeff in case["terms"]}
    assert len(theirs) > 1000, "the reference case is too small to mean much"

    for label in ours.keys() | theirs.keys():
        assert ours.get(label, 0.0) == pytest.approx(
            theirs.get(label, 0.0), abs=COEFF_TOL, rel=0
        ), label
    assert result.overlap_with_zero() == pytest.approx(
        case["overlap_with_zero"], abs=1e-12, rel=0
    )


def test_readme_example_of_the_julia_package():
    """The README of PauliPropagation.jl states 0.154596728241 for this circuit."""
    case = json.loads((REFERENCE_DIR / "readme_tfi_32.json").read_text())
    observable = pp.PauliSum(32)
    observable.add("Z", [15])
    value = pp.expectation(
        build_circuit(case), observable, max_weight=6, min_abs_coeff=1e-4
    )
    assert value == pytest.approx(0.154596728241, abs=1e-12, rel=0)
