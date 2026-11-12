"""Checks every truncation of ``propagate`` on circuits read from QIR.

A truncation that drops nothing has to leave the exact value, which the QDK's state vector
gives, and one that drops more has to move away from it gradually.
"""

import pytest

import paulipropagation_qdk as pp
from qdk_helpers import circuit_source, compile_to_qir, pauli_expectation, state_vector

NUM_QUBITS = 8


def ising_body(layers: int, dt: float) -> list[str]:
    return [
        f"for _ in 1..{layers} {{",
        f"    for i in 0..{NUM_QUBITS - 2} {{ Rzz({dt!r}, qs[i], qs[i + 1]); }}",
        f"    for i in 0..{NUM_QUBITS - 1} {{ Rx({dt!r}, qs[i]); }}",
        "}",
    ]


@pytest.fixture(scope="module")
def ising():
    source = circuit_source(ising_body(layers=5, dt=0.35))
    circuit = pp.circuit_from_qir(compile_to_qir(source, NUM_QUBITS))
    expected = pauli_expectation(state_vector(source, NUM_QUBITS), "Z", [3])
    observable = pp.PauliSum(NUM_QUBITS)
    observable.add("Z", [3])
    return circuit, observable, expected


def errors(ising, settings: list[dict]) -> list[float]:
    circuit, observable, expected = ising
    return [
        abs(pp.expectation(circuit, observable, min_abs_coeff=0.0, **setting) - expected)
        for setting in settings
    ]


def sizes(ising, settings: list[dict]) -> list[int]:
    circuit, observable, _ = ising
    return [len(pp.propagate(circuit, observable, min_abs_coeff=0.0, **s)) for s in settings]


@pytest.mark.parametrize(
    "loose",
    [
        {"max_weight": NUM_QUBITS},
        {"max_freq": 10**6},
        {"max_sins": 10**6},
        {"min_rel_coeff": 0.0},
        {"weight_damping": (0.0, 0.0)},
        {"weight_damping": (5.0, 0.0)},
        {"custom_truncation": lambda label, coeff: False},
    ],
    ids=lambda loose: next(iter(loose)),
)
def test_a_truncation_that_drops_nothing_leaves_the_exact_value(ising, loose):
    assert errors(ising, [{}, loose]) == pytest.approx([0.0, 0.0], abs=1e-10)
    assert len(set(sizes(ising, [{}, loose]))) == 1


@pytest.mark.parametrize(
    "name, values, bounds",
    [
        ("max_weight", [2, 4, 6], [0.2, 2e-2, 1e-3]),
        ("max_sins", [2, 6, 12], [0.2, 2e-2, 1e-3]),
        ("max_freq", [10, 20, 30], [0.3, 5e-2, 1e-3]),
        ("min_rel_coeff", [1e-2, 1e-4, 1e-6], [0.1, 1e-3, 1e-5]),
        ("weight_damping", [(0.5, 1e-3), (0.5, 1e-5), (0.5, 1e-7)], [0.2, 1e-2, 1e-4]),
    ],
)
def test_a_tighter_truncation_keeps_fewer_terms_and_a_looser_one_converges(
    ising, name, values, bounds
):
    settings = [{name: value} for value in values]
    kept = sizes(ising, [*settings, {}])
    assert kept == sorted(kept) and len(set(kept)) == len(kept), kept
    found = errors(ising, settings)
    assert all(error < bound for error, bound in zip(found, bounds)), found
    assert found[0] > found[-1] > 0.0, found


def test_a_function_truncates_like_the_rule_it_states(ising):
    circuit, observable, _ = ising
    by_rule = pp.propagate(circuit, observable, max_weight=3)
    seen = []

    def heavy(label: str, coeff: float) -> bool:
        seen.append((len(label), type(coeff)))
        return len(label) - label.count("I") > 3

    by_function = pp.propagate(circuit, observable, custom_truncation=heavy)
    assert by_function.terms() == by_rule.terms()
    assert len(by_rule) < len(pp.propagate(circuit, observable))
    assert set(seen) == {(NUM_QUBITS, float)}

    def small(label: str, coeff: float) -> bool:
        return abs(coeff) < 1e-3

    by_rule = pp.propagate(circuit, observable, min_abs_coeff=1e-3)
    by_function = pp.propagate(circuit, observable, min_abs_coeff=0.0, custom_truncation=small)
    assert by_function.terms() == by_rule.terms()


def test_a_function_gets_complex_coefficients_of_a_complex_sum(ising):
    circuit, _, _ = ising
    observable = pp.PauliSum(NUM_QUBITS, dtype=complex)
    observable.add("Z", [3], 1j)
    kinds = set()

    def record(label: str, coeff: complex) -> bool:
        kinds.add(type(coeff))
        return False

    pp.propagate(circuit, observable, custom_truncation=record)
    assert kinds == {complex}


def test_an_error_in_the_function_is_raised(ising):
    circuit, observable, _ = ising
    calls = 0

    def broken(label: str, coeff: float) -> bool:
        nonlocal calls
        calls += 1
        if calls == 50:
            raise KeyError("no such rule")
        return False

    with pytest.raises(KeyError, match="no such rule"):
        pp.propagate(circuit, observable, custom_truncation=broken)
    # The function is not asked again after it failed.
    assert calls == 50
    assert len(observable) == 1

    with pytest.raises(TypeError, match="callable"):
        pp.propagate(circuit, observable, custom_truncation=3)


@pytest.mark.parametrize(
    "setting",
    [
        {"min_abs_coeff": -1e-3},
        {"min_abs_coeff": float("nan")},
        {"min_rel_coeff": -0.1},
        {"min_rel_coeff": float("nan")},
        {"weight_damping": (0.1, -1.0)},
        {"weight_damping": (float("nan"), 1e-3)},
    ],
    ids=str,
)
def test_thresholds_that_are_not_thresholds_are_rejected(ising, setting):
    circuit, observable, _ = ising
    with pytest.raises(ValueError):
        pp.propagate(circuit, observable, **setting)
    with pytest.raises(ValueError):
        pp.expectation(circuit, observable, **setting)


def test_counts_that_are_not_counts_are_rejected(ising):
    circuit, observable, _ = ising
    for name in ("max_weight", "max_freq", "max_sins"):
        with pytest.raises(OverflowError):
            pp.propagate(circuit, observable, **{name: -1})
        with pytest.raises(TypeError):
            pp.propagate(circuit, observable, **{name: 1.5})


def test_sums_of_the_wrong_kind_are_rejected(ising):
    circuit, _, _ = ising
    with pytest.raises(ValueError, match="dtype must be float or complex"):
        pp.PauliSum(3, dtype=int)
    real = pp.PauliSum(3)
    with pytest.raises(TypeError):
        real.add("X", [0], 1j)
    assert len(real) == 0
    with pytest.raises(ValueError, match="qubits"):
        pp.propagate(circuit, real)
    with pytest.raises(ValueError, match="3 and 4 qubits"):
        real.scalar_product(pp.PauliSum(4))
