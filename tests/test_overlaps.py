"""Checks overlaps with states against circuits that prepare those states from zero."""

import random

import pytest

import paulipropagation_qdk as pp

NUM_QUBITS = 5


def random_circuit(rng: random.Random, prefix=()) -> pp.Circuit:
    circuit = pp.Circuit(NUM_QUBITS)
    for gate, qubit in prefix:
        getattr(circuit, gate)(qubit)
    for _ in range(40):
        a, b = rng.sample(range(NUM_QUBITS), 2)
        kind = rng.randrange(5)
        if kind == 0:
            circuit.h(a)
        elif kind == 1:
            circuit.cx(a, b)
        elif kind == 2:
            circuit.rotation(
                "".join(rng.choice("XYZ") for _ in range(2)), [a, b], rng.uniform(-3, 3)
            )
        elif kind == 3:
            circuit.s(a)
        else:
            circuit.rx(rng.uniform(-3, 3), a)
    return circuit


def random_observable(rng: random.Random) -> pp.PauliSum:
    observable = pp.PauliSum(NUM_QUBITS)
    for _ in range(4):
        qubits = rng.sample(range(NUM_QUBITS), rng.randint(1, NUM_QUBITS))
        observable.add("".join(rng.choice("XYZ") for _ in qubits), qubits, rng.uniform(-1, 1))
    return observable


@pytest.mark.parametrize("seed", range(10))
def test_overlaps_match_state_preparation(seed):
    rng = random.Random(seed)
    observable = random_observable(rng)
    state = rng.getstate()

    evolved = pp.propagate(random_circuit(rng), observable, min_abs_coeff=0.0)

    def with_prefix(prefix) -> float:
        rng.setstate(state)
        return pp.expectation(random_circuit(rng, prefix), observable, min_abs_coeff=0.0)

    assert evolved.overlap_with_zero() == pytest.approx(with_prefix([]), abs=1e-12)

    hadamards = [("h", q) for q in range(NUM_QUBITS)]
    assert evolved.overlap_with_plus() == pytest.approx(with_prefix(hadamards), abs=1e-12)

    ones = rng.sample(range(NUM_QUBITS), 3)
    assert evolved.overlap_with_computational(ones) == pytest.approx(
        with_prefix([("x", q) for q in ones]), abs=1e-12
    )
    assert evolved.overlap_with_computational([]) == evolved.overlap_with_zero()


def test_overlap_with_a_state_given_as_a_pauli_sum():
    rng = random.Random(5)
    evolved = pp.propagate(random_circuit(rng), random_observable(rng), min_abs_coeff=0.0)

    # |0><0| on every qubit is the product of (I + Z) / 2.
    zero_state = pp.PauliSum(NUM_QUBITS)
    for mask in range(2**NUM_QUBITS):
        qubits = [q for q in range(NUM_QUBITS) if mask >> q & 1]
        zero_state.add("Z" * len(qubits), qubits, 2.0**-NUM_QUBITS)
    assert evolved.overlap_with_pauli_sum(zero_state) == pytest.approx(
        evolved.overlap_with_zero(), abs=1e-12
    )

    # The maximally mixed state is I / 2^n.
    mixed = pp.PauliSum(NUM_QUBITS)
    mixed.add("", [], 2.0**-NUM_QUBITS)
    assert evolved.overlap_with_pauli_sum(mixed) == pytest.approx(
        evolved.overlap_with_max_mixed(), abs=1e-12
    )
    assert evolved.scalar_product(evolved) == pytest.approx(evolved.norm() ** 2, abs=1e-12)

    with pytest.raises(ValueError):
        evolved.scalar_product(pp.PauliSum(NUM_QUBITS + 1))
    with pytest.raises(ValueError):
        evolved.overlap_with_computational([NUM_QUBITS])
