"""Compares propagated Pauli sums, term by term, with results from PauliPropagation.jl.

The reference files in ``tests/reference`` are written by ``tools/julia/generate_reference.jl``.
They hold circuits built by the Julia package itself, so the gates, their order, and the
truncation after every gate have to agree for the terms to match.
"""

import json
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

# Clifford gates of PauliPropagation.jl that are products of Clifford gates here, in the order
# of the circuit. The square root of Y is exp(-i pi/4 Y) up to a phase, and the table of
# ZZpihalf is that of exp(+i pi/4 ZZ).
PRODUCTS = {
    "SY": [("h", (0,)), ("x", (0,))],
    "ZZpihalf": [("s_adj", (0,)), ("s_adj", (1,)), ("cz", (0, 1))],
}
# The factors of ZZpihalf commute, and only the last of them changes the weight of a Pauli
# string. It has to act first: a truncation by weight after it then sees the string that the
# reference sees after its one gate. An observable meets the gates of a circuit from the end,
# a state from the start.
ACTS_FIRST_ON_STATES = {"ZZpihalf": [("cz", (0, 1)), ("s_adj", (0,)), ("s_adj", (1,))]}

# Coefficients agree to rounding.
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
        elif kind in PRODUCTS:
            products = PRODUCTS if case.get("heisenberg", True) else PRODUCTS | ACTS_FIRST_ON_STATES
            for name, places in products[kind]:
                getattr(circuit, name)(*(qubits[place] for place in places))
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


def test_rounding_stays_small_through_a_deep_circuit():
    """The reference propagated this circuit with numbers of 256 bits, so the difference from
    it is the rounding of this package, which stays below 1e-14 through six thousand
    rotations."""
    case = json.loads((REFERENCE_DIR / "deep_circuit_exact_5.json").read_text())
    rotations = sum(gate["gate"] == "rotation" for gate in case["circuit"])
    assert rotations > 5000 and case["min_abs_coeff"] == 0.0
    ours = dict(run_case(case).terms())
    theirs = {label: number(coeff) for label, coeff in case["terms"]}
    assert len(theirs) > 1000
    worst = max(abs(ours.get(label, 0.0) - theirs.get(label, 0.0)) for label in ours.keys() | theirs.keys())
    assert worst < 1e-14
    assert sum(coeff**2 for coeff in ours.values()) == pytest.approx(1.25, abs=1e-12, rel=0)


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
