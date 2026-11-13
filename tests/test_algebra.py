"""Checks sums, products and commutators of Pauli sums against numpy matrices, and merged
symmetries against the symmetry written out on labels."""

import itertools
import random

import numpy as np
import pytest
from reference_helpers import PAULIS, label_matrix, pauli_sum

import paulipropagation_qdk as pp


def random_terms(rng: random.Random, n: int, count: int, complex_coeffs: bool = False) -> dict:
    terms = {}
    for _ in range(count):
        text = "".join(rng.choice(PAULIS) for _ in range(n))
        coeff = rng.uniform(-1, 1)
        terms[text] = complex(coeff, rng.uniform(-1, 1)) if complex_coeffs else coeff
    return terms


def matrix(sum_: pp.PauliSum) -> np.ndarray:
    return sum(coeff * label_matrix(text) for text, coeff in sum_.terms())


@pytest.mark.parametrize("kinds", list(itertools.product([False, True], repeat=2)))
def test_arithmetic_matches_matrices(kinds):
    rng = random.Random(str(kinds))
    for n in (1, 2, 3, 4):
        a = pauli_sum(n, random_terms(rng, n, 6, kinds[0]), dtype=complex if kinds[0] else float)
        b = pauli_sum(n, random_terms(rng, n, 9, kinds[1]), dtype=complex if kinds[1] else float)
        ma, mb = matrix(a), matrix(b)

        assert np.allclose(matrix(a.product(b)), ma @ mb, atol=1e-12)
        assert np.allclose(matrix(a.commutator(b)), ma @ mb - mb @ ma, atol=1e-12)
        assert np.allclose(matrix(a + b), ma + mb, atol=1e-12)
        assert np.allclose(matrix(a - b), ma - mb, atol=1e-12)
        assert np.allclose(matrix(-a), -ma, atol=1e-12)
        assert np.allclose(matrix(a * 0.5), 0.5 * ma, atol=1e-12)
        assert np.allclose(matrix(2 * a), 2 * ma, atol=1e-12)
        assert np.allclose(matrix((1 - 2j) * a), (1 - 2j) * ma, atol=1e-12)

        assert a.product(b).dtype is complex and a.commutator(b).dtype is complex
        assert (a + b).dtype is (complex if any(kinds) else float)
        assert (a * 0.5).dtype is a.dtype and (a * 1j).dtype is complex


def test_the_factors_are_left_as_they_are():
    a = pauli_sum(2, {"XI": 1.0, "ZZ": 0.5})
    b = pauli_sum(2, {"YI": 2.0})
    before = a.terms(), b.terms()
    a + b, a - b, a * 3, a.product(b), a.commutator(b), a.translation_merge()
    assert (a.terms(), b.terms()) == before
    assert dict(a.product(b).terms()) == {"ZI": 2j, "XZ": -1j}
    assert dict(a.commutator(b).terms()) == {"ZI": 4j, "XZ": -2j}


def test_sums_on_different_registers_do_not_combine():
    a, b = pp.PauliSum(2), pp.PauliSum(3)
    for combine in (a.__add__, a.__sub__, a.product, a.commutator):
        with pytest.raises(ValueError, match="different numbers of qubits"):
            combine(b)
    with pytest.raises(ValueError):
        b.translation_merge(2, 2)
    with pytest.raises(ValueError):
        b.translation_merge(3)


def test_a_sum_and_a_circuit_describe_themselves():
    total = pp.PauliSum(3)
    total.add("XZ", [0, 2], 0.5)
    total.add("Y", [1])
    assert repr(total) == "PauliSum(num_qubits=3, terms=2, dtype=float)"
    assert repr(total.astype(complex)) == "PauliSum(num_qubits=3, terms=2, dtype=complex)"
    assert total.max_abs_coeff() == 1.0
    assert pp.PauliSum(3).max_abs_coeff() == 0.0
    assert total.coeff("XZ", [0, 2]) == 0.5
    assert total.coeff("Z", [0]) == 0.0
    with pytest.raises(ValueError, match="out of range"):
        total.coeff("Z", [3])

    circuit = pp.Circuit(3)
    circuit.h(0)
    circuit.rzz(0.1, 0, 2)
    assert repr(circuit) == "Circuit(num_qubits=3, gates=2)"


def as_integer(text: str) -> int:
    """The integer that PauliPropagation.jl holds a Pauli string in: two bits per qubit."""
    return sum(PAULIS.index(char) << (2 * q) for q, char in enumerate(text))


def merged_by_hand(terms: dict, images) -> dict:
    merged = {}
    for text, coeff in terms.items():
        lowest = min(images(text), key=as_integer)
        merged[lowest] = merged.get(lowest, 0.0) + coeff
    return merged


def assert_same(ours: pp.PauliSum, expected: dict) -> None:
    ours = dict(ours.terms())
    assert ours.keys() == expected.keys()
    for text, coeff in expected.items():
        assert ours[text] == pytest.approx(coeff, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 7, 70, 130])
def test_translations_on_a_ring(n):
    rng = random.Random(n)
    terms = {}
    for _ in range(300):
        chars = ["I"] * n
        for q in rng.sample(range(n), min(n, rng.randint(0, 3))):
            chars[q] = rng.choice("XYZ")
        terms["".join(chars)] = rng.uniform(-1, 1)
    merged = pauli_sum(n, terms).translation_merge()
    assert_same(merged, merged_by_hand(terms, lambda t: [t[s:] + t[:s] for s in range(n)]))


@pytest.mark.parametrize("grid", [(2, 3), (3, 3), (4, 2), (1, 6), (6, 1), (9, 8)])
def test_translations_on_a_torus(grid):
    nx, ny = grid
    n = nx * ny
    rng = random.Random(n)
    terms = {}
    for _ in range(300):
        chars = ["I"] * n
        for q in rng.sample(range(n), rng.randint(0, 3)):
            chars[q] = rng.choice("XYZ")
        terms["".join(chars)] = rng.uniform(-1, 1)

    def images(text):
        for dx in range(nx):
            for dy in range(ny):
                moved = ["I"] * n
                for q, char in enumerate(text):
                    moved[((q // nx + dy) % ny) * nx + (q % nx + dx) % nx] = char
                yield "".join(moved)

    assert_same(pauli_sum(n, terms).translation_merge(nx, ny), merged_by_hand(terms, images))


def test_a_ring_of_single_operators_becomes_one_term():
    # The example of PauliPropagation.jl: Z on its qubits 3 and 6 of 6.
    total = pp.PauliSum(6)
    total.add("Z", [2])
    total.add("Z", [5])
    assert total.translation_merge().terms() == [("ZIIIII", 2.0)]
    # On a torus every site is a translation of the first one, too.
    assert total.translation_merge(2, 3).terms() == [("ZIIIII", 2.0)]


def test_a_merged_sum_has_the_expectation_value_of_the_whole():
    # A circuit that is the same on every site, and the magnetization of the ring: the terms
    # that are translations of each other have the same value in the all-zero state.
    n, layers = 10, 4
    layer, circuit = pp.Circuit(n), pp.Circuit(n)
    for target in [layer] + [circuit] * layers:
        for q in range(n):
            target.rx(0.3, q)
        for q in range(n):
            target.rzz(0.4, q, (q + 1) % n)
    magnetization = pp.PauliSum(n)
    for q in range(n):
        magnetization.add("Z", [q], 1 / n)

    whole = pp.propagate(circuit, magnetization, max_weight=6)
    merged = magnetization
    for _ in range(layers):
        merged = pp.propagate(layer, merged, max_weight=6).translation_merge()
    assert len(merged) * 5 < len(whole)
    assert merged.overlap_with_zero() == pytest.approx(whole.overlap_with_zero(), abs=1e-9)


def test_a_symmetry_given_as_a_function():
    rng = random.Random(3)
    n = 6
    terms = random_terms(rng, n, 200)
    merged = pauli_sum(n, terms).symmetry_merge(lambda text: min(text, text[::-1], key=as_integer))
    assert_same(merged, merged_by_hand(terms, lambda text: [text, text[::-1]]))
    assert len(merged) < len(terms)

    complex_terms = random_terms(rng, n, 50, complex_coeffs=True)
    merged = pauli_sum(n, complex_terms, dtype=complex).symmetry_merge(lambda text: "I" * n)
    assert merged.terms() == [("I" * n, pytest.approx(sum(complex_terms.values())))]


def test_a_symmetry_function_that_fails():
    total = pauli_sum(3, {"XYZ": 1.0})

    def fails(text):
        raise KeyError(text)

    with pytest.raises(KeyError):
        total.symmetry_merge(fails)
    with pytest.raises(ValueError):
        total.symmetry_merge(lambda text: "XQZ")
    with pytest.raises(ValueError):
        total.symmetry_merge(lambda text: "XY")
    with pytest.raises(TypeError):
        total.symmetry_merge(lambda text: 3)
