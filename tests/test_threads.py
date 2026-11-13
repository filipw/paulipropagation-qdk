"""Checks that the number of threads changes nothing but the time a propagation takes.

A sum that is large enough is divided over the threads, and one that is small is not. Either
way every coefficient has to come out bit for bit as it does on one thread, and as it does
in the propagation written out in ``reference_helpers``.
"""

import pytest

import paulipropagation_qdk as pp
from reference_helpers import (
    bind,
    count_parameters,
    hardware_efficient_circuit,
    rectangle_topology,
    reference_propagate,
    single,
    tfi_trotter_circuit,
)

THREADS = [1, 2, 3, 8]


def large_case(dtype=float):
    """A circuit under which a single Pauli string spreads to tens of thousands."""
    num_qubits = 16
    gates = tfi_trotter_circuit(num_qubits, 8, rectangle_topology(4, 4))
    thetas = [0.3 + 0.01 * index for index in range(count_parameters(gates))]
    observable = single(num_qubits, "Z", [5])
    if dtype is complex:
        observable = observable.astype(complex)
        observable.add("XY", [2, 9], 0.5 - 0.25j)
    return bind(num_qubits, gates, thetas), observable


@pytest.mark.parametrize("dtype", [float, complex])
def test_a_large_sum_is_the_same_on_any_number_of_threads(dtype):
    circuit, observable = large_case(dtype)
    options = {"max_weight": 6, "min_abs_coeff": 1e-6}
    expected = pp.propagate(circuit, observable, threads=1, **options).terms()
    assert len(expected) > 20_000, "the sum is too small to be divided over threads"
    for threads in THREADS[1:]:
        assert pp.propagate(circuit, observable, threads=threads, **options).terms() == expected
    assert pp.propagate(circuit, observable, **options).terms() == expected


@pytest.mark.parametrize(
    "options",
    [
        {"min_rel_coeff": 1e-4, "max_weight": 6},
        {"max_freq": 30, "max_sins": 12, "max_weight": 6, "min_abs_coeff": 1e-6},
        {"weight_damping": (0.05, 1e-5), "max_weight": 6},
        {"heisenberg": False, "max_weight": 5, "min_abs_coeff": 1e-5},
    ],
    ids=lambda options: "+".join(options),
)
def test_every_kind_of_truncation_is_the_same_on_any_number_of_threads(options):
    circuit, observable = large_case()
    expected = pp.propagate(circuit, observable, threads=1, **options).terms()
    assert len(expected) > 5_000, "the sum is too small to be divided over threads"
    for threads in THREADS[1:]:
        assert pp.propagate(circuit, observable, threads=threads, **options).terms() == expected


def test_a_custom_rule_is_asked_from_every_thread():
    circuit, observable = large_case()
    asked = []

    def rule(label: str, coeff: float) -> bool:
        asked.append(label)
        return label.count("Y") > 2

    options = {"max_weight": 4, "min_abs_coeff": 1e-4}
    expected = pp.propagate(circuit, observable, threads=1, custom_truncation=rule, **options)
    alone = len(asked)
    assert alone > 10_000
    together = pp.propagate(circuit, observable, threads=4, custom_truncation=rule, **options)
    assert together.terms() == expected.terms()
    assert len(asked) == 2 * alone
    assert len(expected) < len(pp.propagate(circuit, observable, **options))


def test_an_error_in_a_custom_rule_ends_the_propagation_on_any_number_of_threads():
    circuit, observable = large_case()
    calls = 0

    def rule(label: str, coeff: float) -> bool:
        nonlocal calls
        calls += 1
        if calls > 20_000:
            raise RuntimeError("no more")
        return False

    for threads in (1, 4):
        calls = 0
        with pytest.raises(RuntimeError, match="no more"):
            pp.propagate(circuit, observable, threads=threads, custom_truncation=rule, max_weight=5)


def test_threads_agree_with_the_reference_propagation():
    num_qubits = 8
    gates = hardware_efficient_circuit(num_qubits, 3)
    thetas = [0.2 + 0.07 * index for index in range(count_parameters(gates))]
    circuit = bind(num_qubits, gates, thetas)
    terms = {"IIIZIIII": 1.0, "XXIIIIII": -0.5}
    observable = pp.PauliSum(num_qubits)
    for text, coeff in terms.items():
        observable.add(text, list(range(num_qubits)), coeff)

    expected = reference_propagate(num_qubits, gates, thetas, terms, max_weight=5, min_abs_coeff=1e-6)
    assert len(expected) > 2_000
    for threads in THREADS:
        result = dict(pp.propagate(circuit, observable, threads=threads, max_weight=5, min_abs_coeff=1e-6).terms())
        assert result.keys() == expected.keys()
        for text, coeff in expected.items():
            assert result[text] == pytest.approx(coeff, abs=1e-13, rel=0), text


def test_expectation_takes_the_number_of_threads():
    circuit, observable = large_case()
    options = {"max_weight": 5, "min_abs_coeff": 1e-5}
    expected = pp.expectation(circuit, observable, threads=1, **options)
    assert expected != 0.0
    assert pp.expectation(circuit, observable, threads=5, **options) == expected
    assert pp.expectation(circuit, observable, **options) == expected


@pytest.mark.parametrize("threads", [0, -1])
def test_there_is_at_least_one_thread(threads):
    circuit, observable = large_case()
    with pytest.raises((ValueError, OverflowError)):
        pp.propagate(circuit, observable, threads=threads)
