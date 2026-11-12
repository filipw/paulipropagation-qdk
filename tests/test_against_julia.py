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


def number(value) -> float | complex:
    """A coefficient of a reference file: a real number, or the two parts of a complex one."""
    return complex(*value) if isinstance(value, list) else value


def build_circuit(case, matrices_as_unitaries: bool = False) -> pp.Circuit:
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
        elif kind == "T":
            circuit.t(qubits[0])
        elif kind == "TransferMap":
            if matrices_as_unitaries and "unitary" in gate:
                circuit.unitary(qubits, [[number(v) for v in row] for row in gate["unitary"]])
            else:
                circuit.transfer_map(qubits, gate["matrix"])
        else:
            getattr(circuit, SAME_GATE[kind])(*qubits)
    return circuit


def run_case(case, **options) -> pp.PauliSum:
    """Propagates the sum of a reference case with its truncations and in its picture."""
    coeffs = [number(term["coeff"]) for term in case["observable"]]
    dtype = complex if any(isinstance(c, complex) for c in coeffs) else float
    start = pp.PauliSum(case["num_qubits"], dtype=dtype)
    for term, coeff in zip(case["observable"], coeffs):
        start.add(term["paulis"], term["qubits"], coeff)

    damping = case.get("weight_damping")
    return pp.propagate(
        build_circuit(case, **options),
        start,
        max_weight=case["max_weight"],
        min_abs_coeff=case["min_abs_coeff"],
        min_rel_coeff=case.get("min_rel_coeff"),
        max_freq=case.get("max_freq"),
        max_sins=case.get("max_sins"),
        weight_damping=tuple(damping) if damping else None,
        heisenberg=case.get("heisenberg", True),
    )


def assert_matches(result: pp.PauliSum, case, tolerance: float = COEFF_TOL) -> None:
    assert result.overlap_with_zero() == pytest.approx(
        number(case["overlap_with_zero"]), abs=1e-12, rel=0
    )
    assert case["num_terms"] > 50, "the reference case is too small to mean much"
    if case["terms"] is None:
        # A case too large to store term by term.
        assert len(result) == case["num_terms"]
        return

    ours = dict(result.terms())
    theirs = {label: number(coeff) for label, coeff in case["terms"]}
    for label in ours.keys() | theirs.keys():
        assert ours.get(label, 0.0) == pytest.approx(
            theirs.get(label, 0.0), abs=tolerance, rel=0
        ), label


def test_reference_cases_exist():
    assert "readme_tfi_32" in CASES


@pytest.mark.parametrize("name", CASES)
def test_terms_match_the_julia_reference(name):
    case = json.loads((REFERENCE_DIR / f"{name}.json").read_text())
    assert_matches(run_case(case), case)


def test_unitary_matrices_match_the_transfer_maps_of_the_reference():
    """The reference derives a transfer map from each unitary; here the unitary is the gate."""
    case = json.loads((REFERENCE_DIR / "matrix_gates_6.json").read_text())
    assert sum("unitary" in gate for gate in case["circuit"]) >= 10
    assert_matches(run_case(case, matrices_as_unitaries=True), case, tolerance=1e-12)


def test_readme_example_of_the_julia_package():
    """The README of PauliPropagation.jl states 0.154596728241 for this circuit."""
    case = json.loads((REFERENCE_DIR / "readme_tfi_32.json").read_text())
    observable = pp.PauliSum(32)
    observable.add("Z", [15])
    value = pp.expectation(
        build_circuit(case), observable, max_weight=6, min_abs_coeff=1e-4
    )
    assert value == pytest.approx(0.154596728241, abs=1e-12, rel=0)
