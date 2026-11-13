"""Checks that a surrogate evaluated at any angles is the propagation with those angles."""

import random

import pytest
from qdk_helpers import circuit_source, compile_to_qir
from test_montecarlo import mixed_circuit

import paulipropagation_qdk as pp


@pytest.mark.parametrize("limits", [
    {},
    {"max_weight": 3},
    {"max_freq": 12},
    {"max_sins": 4},
    {"max_weight": 5, "max_freq": 18, "max_sins": 6},
])
def test_a_surrogate_is_the_propagation_for_any_angles(limits):
    rng = random.Random(3)
    n = 8
    circuit = mixed_circuit(rng, n, 60)
    observable = pp.PauliSum(n)
    observable.add("Z", [3])
    observable.add("XY", [1, 6], -0.4)

    surrogate = pp.Surrogate(circuit, observable, **limits)
    assert surrogate.num_parameters == circuit.num_rotations
    assert surrogate.num_qubits == n and surrogate.num_nodes > 100
    for _ in range(5):
        angles = [rng.uniform(-3, 3) for _ in circuit.angles]
        expected = pp.propagate(circuit.with_angles(angles), observable, min_abs_coeff=0.0, **limits)
        evaluated = surrogate.evaluate(angles)
        # Every coefficient is made of the same products and sums.
        assert evaluated.terms() == expected.terms()
        assert len(surrogate) == len(expected) > 50
    assert surrogate.evaluate(circuit.angles).terms() == pp.propagate(circuit, observable, min_abs_coeff=0.0, **limits).terms()


def test_a_surrogate_with_a_rule_of_its_own():
    rng = random.Random(9)
    n = 6
    circuit = mixed_circuit(rng, n, 60)
    observable = pp.PauliSum(n)
    observable.add("Z", [2])
    rule = lambda text: text[0] != "I" and text.count("I") < 3  # noqa: E731

    surrogate = pp.Surrogate(circuit, observable, custom_truncation=rule)
    expected = pp.propagate(circuit, observable, min_abs_coeff=0.0, custom_truncation=lambda text, _: rule(text))
    assert surrogate.evaluate(circuit.angles).terms() == expected.terms()
    assert len(surrogate) < len(pp.Surrogate(circuit, observable))


def test_a_surrogate_of_a_circuit_from_qir():
    source = circuit_source([
        "for i in 0..4 { Rx(0.2, qs[i]); }",
        "for i in 0..3 { Rzz(0.3, qs[i], qs[i + 1]); }",
        "T(qs[2]); CNOT(qs[1], qs[3]);",
        "for i in 0..4 { Ry(0.4, qs[i]); }",
    ])
    circuit = pp.circuit_from_qir(compile_to_qir(source, 5))
    observable = pp.PauliSum(5)
    observable.add("Z", [2])
    surrogate = pp.Surrogate(circuit, observable, max_weight=3)
    assert surrogate.num_parameters == 15
    expected = pp.propagate(circuit, observable, max_weight=3, min_abs_coeff=0.0)
    assert surrogate.evaluate(circuit.angles).terms() == expected.terms()
    value, _ = pp.gradient(circuit, observable, min_abs_coeff=0.0)
    assert pp.Surrogate(circuit, observable).evaluate(circuit.angles).overlap_with_zero() == pytest.approx(value, abs=1e-12)


def test_a_surrogate_rejects_what_it_cannot_hold():
    circuit = pp.Circuit(2)
    circuit.rx(0.3, 0)
    circuit.rzz(0.2, 0, 1)
    observable = pp.PauliSum(2)
    observable.add("Z", [0])
    surrogate = pp.Surrogate(circuit, observable)
    assert repr(surrogate) == "Surrogate(num_qubits=2, parameters=2, terms=2, nodes=2)"
    with pytest.raises(ValueError, match="one angle for every rotation"):
        surrogate.evaluate([0.1])
    with pytest.raises(ValueError):
        surrogate.evaluate([0.1, float("inf")])
    with pytest.raises(ValueError, match="real coefficients"):
        pp.Surrogate(circuit, observable.astype(complex))
    with pytest.raises(TypeError):
        pp.Surrogate(circuit, observable, custom_truncation=1)
    with pytest.raises(TypeError):
        pp.Surrogate(circuit, observable, min_abs_coeff=1e-3)

    def fails(text):
        raise KeyError(text)

    with pytest.raises(KeyError):
        pp.Surrogate(circuit, observable, custom_truncation=fails)
    circuit.amplitude_damping(0.1, 0)
    with pytest.raises(ValueError, match="Clifford gates and Pauli rotations"):
        pp.Surrogate(circuit, observable)
