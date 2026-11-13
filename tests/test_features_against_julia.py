"""Compares merged translations, gradients, error estimates and surrogates with results from
PauliPropagation.jl.

The reference files in ``tests/reference/features`` are written by
``tools/julia/generate_features.jl``.
"""

import json
import math
from pathlib import Path

import pytest
from test_against_julia import build_circuit

import paulipropagation_qdk as pp

FEATURES_DIR = Path(__file__).parent / "reference" / "features"


def cases(prefix: str) -> list[str]:
    return sorted(path.stem for path in FEATURES_DIR.glob(f"{prefix}*.json"))


def load(name: str) -> dict:
    return json.loads((FEATURES_DIR / f"{name}.json").read_text())


def observable_of(case) -> pp.PauliSum:
    observable = pp.PauliSum(case["num_qubits"])
    for term in case["observable"]:
        observable.add(term["paulis"], term["qubits"], term["coeff"])
    return observable


def assert_same_terms(ours: pp.PauliSum, theirs: list, tolerance: float) -> None:
    ours, theirs = dict(ours.terms()), dict(map(tuple, theirs))
    for label in ours.keys() | theirs.keys():
        assert ours.get(label, 0.0) == pytest.approx(theirs.get(label, 0.0), abs=tolerance, rel=0), label


def test_reference_cases_exist():
    for prefix, count in [("translation_merge", 4), ("gradient", 5), ("mse", 6), ("surrogate", 3)]:
        assert len(cases(prefix)) == count


@pytest.mark.parametrize("name", cases("translation_merge"))
def test_merged_translations_match_the_julia_reference(name):
    case = load(name)
    n = case["num_qubits"]
    start = pp.PauliSum(n)
    for label, coeff in case["terms"]:
        start.add(label, list(range(n)), coeff)

    merged = start.translation_merge(*case["grid"]) if case["grid"] else start.translation_merge()

    # The same Pauli string stands for each set of translations, with the coefficients added up.
    assert len(merged) == len(case["merged"]) < len(start)
    assert_same_terms(merged, case["merged"], 1e-12)


@pytest.mark.parametrize("name", cases("gradient"))
def test_gradients_match_the_julia_reference(name):
    case = load(name)
    circuit = build_circuit(case)
    value, derivatives = pp.gradient(
        circuit,
        observable_of(case),
        state=case["state"],
        max_weight=case["max_weight"],
        min_abs_coeff=case["min_abs_coeff"],
    )
    assert circuit.angles == [gate["theta"] for gate in case["circuit"] if gate["gate"] == "rotation"]
    assert value == pytest.approx(case["value"], abs=1e-12, rel=0)
    assert len(derivatives) == len(case["gradient"]) >= 40
    assert max(map(abs, case["gradient"])) > 0.05
    # A truncation by coefficients near the threshold can go either way on a last digit.
    assert derivatives == pytest.approx(case["gradient"], abs=1e-11, rel=0)


@pytest.mark.parametrize("name", cases("mse"))
def test_error_estimates_agree_with_the_julia_reference(name):
    # Two estimates from a million random paths each agree to their statistical error.
    case = load(name)
    samples = case["samples"]
    estimate = pp.estimate_mse(
        build_circuit(case),
        observable_of(case),
        samples,
        thetas=case["thetas"],
        seed=1,
        **case["limits"],
    )
    theirs = case["mse"]
    assert theirs > 1e-3
    error = math.sqrt(2 * theirs * (1 - theirs) / samples)
    assert abs(estimate - theirs) < 6 * error, (estimate, theirs, error)


@pytest.mark.parametrize("name", cases("surrogate"))
def test_surrogates_match_the_julia_reference(name):
    case = load(name)
    circuit = build_circuit(case)
    surrogate = pp.Surrogate(circuit, observable_of(case), **case["limits"])
    assert surrogate.num_parameters == len(circuit.angles)
    for evaluation in case["evaluations"]:
        evaluated = surrogate.evaluate(evaluation["thetas"])
        assert len(evaluated) == len(evaluation["terms"])
        assert_same_terms(evaluated, evaluation["terms"], 1e-13)
