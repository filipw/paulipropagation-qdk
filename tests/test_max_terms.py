"""Checks `max_terms` against a propagation of one gate at a time through the public API,
with the largest terms picked in Python after every gate."""

import random

import pytest
from reference_helpers import pauli_sum

import paulipropagation_qdk as pp


def random_gates(rng: random.Random, n: int, count: int) -> list:
    """Gates as the name of the method of `Circuit` and its arguments."""
    gates = []
    for _ in range(count):
        a, b = rng.sample(range(n), 2)
        theta = rng.uniform(-3, 3)
        kind = rng.randrange(8)
        if kind == 0:
            gates.append((rng.choice(["h", "s", "sx", "t"]), (a,)))
        elif kind == 1:
            gates.append((rng.choice(["cx", "cz", "swap"]), (a, b)))
        elif kind == 2:
            gates.append(("amplitude_damping", (rng.uniform(0, 0.5), a)))
        elif kind == 3:
            gates.append(("depolarizing", (rng.uniform(0, 0.2), a)))
        elif kind in (4, 5):
            gates.append((rng.choice(["rx", "ry", "rz"]), (theta, a)))
        else:
            gates.append((rng.choice(["rxx", "ryy", "rzz"]), (theta, a, b)))
    return gates


def circuit_of(n: int, gates: list) -> pp.Circuit:
    circuit = pp.Circuit(n)
    for name, arguments in gates:
        getattr(circuit, name)(*arguments)
    return circuit


def place(label: str) -> tuple[int, int]:
    """Where a Pauli string stands in the order that decides among equally large terms:
    by the qubits with X or Y, then by the qubits with Y or Z, each as a binary number."""
    x = sum(1 << q for q, op in enumerate(label) if op in "XY")
    z = sum(1 << q for q, op in enumerate(label) if op in "YZ")
    return x, z


def gate_by_gate(n: int, gates: list, start: pp.PauliSum, max_terms: int, heisenberg: bool, **options):
    """The terms after every gate cut down to the largest, and how often that happened."""
    dtype = complex if isinstance(start.terms()[0][1], complex) else float
    terms, cuts = dict(start.terms()), 0
    for gate in reversed(gates) if heisenberg else gates:
        after = pp.propagate(circuit_of(n, [gate]), pauli_sum(n, terms, dtype=dtype), heisenberg=heisenberg, **options)
        by_size = sorted(after.terms(), key=lambda term: (-abs(term[1]), place(term[0])))
        cuts += len(by_size) > max_terms
        terms = dict(by_size[:max_terms])
    return terms, cuts


@pytest.mark.parametrize("heisenberg", [True, False])
@pytest.mark.parametrize("options", [
    {"min_abs_coeff": 0.0},
    {"max_weight": 4, "min_abs_coeff": 1e-6},
    {"min_rel_coeff": 1e-3, "weight_damping": (0.2, 1e-4)},
])
def test_the_largest_terms_are_kept_after_every_gate(heisenberg, options):
    rng = random.Random(41)
    cuts = 0
    for dtype in (float, float, complex):
        n = rng.randint(4, 6)
        gates = random_gates(rng, n, 45)
        coeff = (lambda: rng.uniform(-1, 1)) if dtype is float else (lambda: complex(rng.uniform(-1, 1), rng.uniform(-1, 1)))
        start = pauli_sum(n, {"".join(rng.choice("IXYZ") for _ in range(n)): coeff() for _ in range(3)}, dtype=dtype)
        for max_terms in (1, 12, 150):
            expected, cut = gate_by_gate(n, gates, start, max_terms, heisenberg, **options)
            capped = pp.propagate(circuit_of(n, gates), start, max_terms=max_terms, heisenberg=heisenberg, **options)
            assert dict(capped.terms()) == expected
            assert len(capped) <= max_terms
            cuts += cut
    assert cuts > 60


def trotter(n: int, layers: int) -> pp.Circuit:
    circuit = pp.Circuit(n)
    for _ in range(layers):
        for q in range(n - 1):
            circuit.rzz(0.35, q, q + 1)
        for q in range(n):
            circuit.rx(0.45, q)
    return circuit


def test_a_bounded_sum_on_many_qubits():
    n = 60
    circuit = trotter(n, 12)
    observable = pp.PauliSum(n)
    observable.add("Z", [n // 2])
    plain = pp.propagate(circuit, observable, max_weight=6, min_abs_coeff=1e-6)
    assert len(plain) > 10_000

    capped = pp.propagate(circuit, observable, max_weight=6, min_abs_coeff=1e-6, max_terms=5_000)
    assert len(capped) == 5_000
    for threads in (1, 2, 5):
        again = pp.propagate(circuit, observable, max_weight=6, min_abs_coeff=1e-6, max_terms=5_000, threads=threads)
        assert again.terms() == capped.terms()
    # The largest terms carry the value: the bounded sum is close, and closer with more terms.
    exact = plain.overlap_with_zero()
    errors = [
        abs(pp.expectation(circuit, observable, max_weight=6, min_abs_coeff=1e-6, max_terms=max_terms) - exact)
        for max_terms in (500, 5_000, 12_000)
    ]
    assert errors[0] > errors[1] > errors[2] > 0
    assert errors[1] < 0.01
    assert capped.overlap_with_zero() == pp.expectation(circuit, observable, max_weight=6, min_abs_coeff=1e-6, max_terms=5_000)

    # A number of terms that is never reached changes nothing.
    for max_terms in (len(plain), 10 * len(plain), 2**62):
        unbounded = pp.propagate(circuit, observable, max_weight=6, min_abs_coeff=1e-6, max_terms=max_terms)
        assert unbounded.terms() == plain.terms()


def test_a_custom_rule_and_what_is_rejected():
    circuit = trotter(8, 4)
    observable = pp.PauliSum(8)
    observable.add("Z", [3])
    rule = lambda text, coeff: "Y" in text and abs(coeff) < 0.05  # noqa: E731
    plain = pp.propagate(circuit, observable, custom_truncation=rule)
    capped = pp.propagate(circuit, observable, custom_truncation=rule, max_terms=10 * len(plain))
    assert capped.terms() == plain.terms()
    assert 0 < len(pp.propagate(circuit, observable, custom_truncation=rule, max_terms=10)) <= 10 < len(plain)

    with pytest.raises(ValueError, match="positive"):
        pp.propagate(circuit, observable, max_terms=0)
    with pytest.raises(ValueError, match="max_freq or max_sins"):
        pp.propagate(circuit, observable, max_terms=10, max_freq=3)
    with pytest.raises(ValueError, match="max_freq or max_sins"):
        pp.expectation(circuit, observable, max_terms=10, max_sins=3)
    with pytest.raises(OverflowError):
        pp.propagate(circuit, observable, max_terms=-1)
