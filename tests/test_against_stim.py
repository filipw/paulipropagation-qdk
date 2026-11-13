"""Compares Clifford circuits with the stabilizer simulator Stim, on up to 512 qubits.

A Clifford circuit takes a Pauli string to one Pauli string with a sign, which Stim gives
from a tableau of the circuit without a Pauli sum or a state vector. Every Clifford gate of
this package and every rotation by a quarter or half turn is compared: the string and the
sign for observables and for states, and expectation values in the state that the circuit
prepares. The tests run where Stim is installed.
"""

import math
import random

import pytest
from reference_helpers import single
from test_published_results import KICKED_ISING_OBSERVABLES, heavy_hex_bonds

import paulipropagation_qdk as pp

stim = pytest.importorskip("stim")

# Gates of this package and the gates of Stim that they are.
SAME_GATE = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "s_adj": "S_DAG",
    "sx": "SQRT_X",
    "sx_adj": "SQRT_X_DAG",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
}
ONE_QUBIT = ["h", "x", "y", "z", "s", "s_adj", "sx", "sx_adj"]
TWO_QUBIT = ["cx", "cy", "cz", "swap"]

SIZES = [1, 2, 5, 32, 33, 64, 65, 128, 129, 300, 512]


def add_rotation(circuit: pp.Circuit, theirs, paulis: str, qubits: list[int], turns: int) -> None:
    """A rotation about a Pauli string by `turns` quarter turns, which is the square root of
    the string for one of them, up to a phase."""
    circuit.rotation(paulis, qubits, turns * math.pi / 2)
    product = "*".join(f"{pauli}{q}" for pauli, q in zip(paulis, qubits))
    for name in {1: ["SPP"], 2: ["SPP", "SPP"], 3: ["SPP_DAG"]}[turns % 4]:
        theirs.append_from_stim_program_text(f"{name} {product}")


def random_clifford_circuit(rng: random.Random, n: int, gates: int):
    """The same random circuit for this package and for Stim."""
    circuit, theirs = pp.Circuit(n), stim.Circuit()
    for _ in range(gates):
        kind = rng.randrange(4)
        if kind == 0 or n == 1 and kind == 1:
            name, qubits = rng.choice(ONE_QUBIT), [rng.randrange(n)]
        elif kind == 1:
            name, qubits = rng.choice(TWO_QUBIT), rng.sample(range(n), 2)
        else:
            qubits = rng.sample(range(n), rng.randint(1, min(n, 4)))
            paulis = "".join(rng.choice("XYZ") for _ in qubits)
            add_rotation(circuit, theirs, paulis, qubits, rng.choice([1, 2, 3, -1, 5]))
            continue
        getattr(circuit, name)(*qubits)
        theirs.append(SAME_GATE[name], qubits)
    # Stim sizes a tableau by the qubits it has seen.
    theirs.append("I", [n - 1])
    return circuit, theirs


def random_string(rng: random.Random, n: int) -> str:
    weight = min(n, rng.choice([1, 2, 7, n]))
    qubits = set(rng.sample(range(n), weight))
    return "".join(rng.choice("XYZ") if q in qubits else "I" for q in range(n))


def to_stim(text: str):
    return stim.PauliString(text.replace("I", "_"))


def from_stim(string) -> tuple[str, float]:
    """A Pauli string of Stim as a label and a sign."""
    assert string.sign in (1, -1)
    return str(string)[1:].replace("_", "I"), string.sign.real


def only_term(result: pp.PauliSum) -> tuple[str, float]:
    ((text, coeff),) = result.terms()
    return text, coeff


@pytest.mark.parametrize("n", SIZES)
def test_pauli_strings_through_clifford_circuits(n):
    rng = random.Random(n)
    circuit, theirs = random_clifford_circuit(rng, n, 6 * n + 60)
    assert circuit.num_rotations > 20
    forwards = theirs.to_tableau()
    backwards = forwards.inverse()
    signs = set()
    for _ in range(25):
        text = random_string(rng, n)
        start = single(n, text, range(n))
        # An observable goes to U† P U, a state to U P U†.
        observable = only_term(pp.propagate(circuit, start))
        assert observable == from_stim(backwards(to_stim(text)))
        state = only_term(pp.propagate(circuit, start, heisenberg=False))
        assert state == from_stim(forwards(to_stim(text)))
        signs |= {observable[1], state[1]}
    # A gate on one qubit can change the sign of all three operators.
    assert n == 1 or signs == {1.0, -1.0}


@pytest.mark.parametrize("n", SIZES)
def test_expectation_values_in_stabilizer_states(n):
    rng = random.Random(1000 + n)
    circuit, theirs = random_clifford_circuit(rng, n, 6 * n + 60)
    simulator = stim.TableauSimulator()
    simulator.do(theirs)
    forwards = theirs.to_tableau()

    values = set()
    for _ in range(25):
        # The image of a product of Z operators is 1 or -1 in the state of the circuit, and
        # a random string is hardly ever anything but 0.
        ones = "".join(rng.choice("IZ") for _ in range(n - 1)) + "Z"
        stabilizer, sign = from_stim(forwards(to_stim(ones)))
        for text in (stabilizer, random_string(rng, n)):
            value = pp.expectation(circuit, single(n, text, range(n)))
            assert value == simulator.peek_observable_expectation(to_stim(text))
            values.add(value)
        assert pp.expectation(circuit, single(n, stabilizer, range(n))) == sign
    # One qubit has one stabilizer, with one sign.
    assert n == 1 or {1.0, -1.0} <= values
    assert n < 5 or 0.0 in values


@pytest.mark.parametrize(("steps", "last_kick", "qubits_of", "sines"), KICKED_ISING_OBSERVABLES)
def test_the_kicked_ising_experiment_at_the_clifford_point(steps, last_kick, qubits_of, sines):
    """Rotations about X by pi/2 and about ZZ by -pi/2 on the heavy-hex lattice of 127 qubits,
    as rotations here and as the square roots of X and of ZZ in Stim."""
    kicks, bonds = stim.Circuit(), stim.Circuit()
    kicks.append("SQRT_X", range(127))
    for a, b in heavy_hex_bonds():
        bonds.append("SQRT_ZZ_DAG", [a, b])
    bonds.append("I", [126])
    layers = [kicks, bonds] * steps + [kicks] * last_kick

    circuit, theirs = pp.Circuit(127), stim.Circuit()
    for layer in layers:
        theirs += layer
        if layer is kicks:
            for q in range(127):
                circuit.rx(math.pi / 2, q)
        else:
            for a, b in heavy_hex_bonds():
                circuit.rzz(-math.pi / 2, a, b)
    simulator = stim.TableauSimulator()
    simulator.do(theirs)

    chars = ["I"] * 127
    for pauli, qubits in qubits_of.items():
        for q in qubits:
            chars[q] = pauli
    text = "".join(chars)
    expected = simulator.peek_observable_expectation(to_stim(text))
    assert abs(expected) == 1
    assert pp.expectation(circuit, single(127, text, range(127))) == expected

    # The path of the observable takes a sine at every kick that does not commute with it:
    # going back through the layers, those are the qubits on which it has Y or Z.
    string, met = to_stim(text), 0
    for layer in reversed(layers):
        if layer is kicks:
            met += sum(string[q] in (2, 3) for q in range(127))
        string = layer.to_tableau().inverse()(string)
    assert met == sines
