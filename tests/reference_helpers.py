"""References that share no code with the engine: the circuits and topologies that
PauliPropagation.jl builds, its Clifford tables, dense matrices, and a propagation that works
on labels a term at a time.
"""

import cmath
import itertools
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

import paulipropagation_qdk as pp

REFERENCE_DIR = Path(__file__).parent / "reference"
PAULIS = "IXYZ"

# A gate of a circuit without its angles: the Pauli operators of a rotation and its qubits,
# "CNOT" and its qubits, or a function that adds gates of its own to a circuit.
Layer = tuple[str, tuple[int, ...]] | Callable[[pp.Circuit], None]


# Topologies, with qubits counted from zero.


def bricklayer_topology(num_qubits: int, periodic: bool = False) -> list[tuple[int, int]]:
    if num_qubits == 1:
        return []
    if num_qubits == 2:
        return [(0, 1)]
    pairs = [(q, q + 1) for q in range(0, num_qubits - 1, 2)]
    if periodic and num_qubits % 2 == 1:
        pairs.append((num_qubits - 1, 0))
    pairs += [(q, q + 1) for q in range(1, num_qubits - 1, 2)]
    if periodic and num_qubits % 2 == 0:
        pairs.append((num_qubits - 1, 0))
    return pairs


def staircase_topology(num_qubits: int, periodic: bool = False) -> list[tuple[int, int]]:
    pairs = [(q, q + 1) for q in range(num_qubits - 1)]
    if periodic and num_qubits > 2:
        pairs.append((num_qubits - 1, 0))
    return pairs


def rectangle_topology(nx: int, ny: int, periodic: bool = False) -> list[tuple[int, int]]:
    pairs = []
    for row in range(ny):
        for column in range(nx):
            if row < ny - 1:
                pairs.append((row * nx + column, (row + 1) * nx + column))
            if column < nx - 1:
                pairs.append((row * nx + column, row * nx + column + 1))
    if periodic:
        last_row = (ny - 1) * nx
        pairs += [(column, last_row + column) for column in range(nx)]
        pairs += [(row * nx, row * nx + nx - 1) for row in range(ny)]
        pairs = [pair for pair in dict.fromkeys(pairs) if pair[0] != pair[1]]
    return pairs


# Circuits. Each returns the gates without their angles, as the Julia package does.


def hardware_efficient_circuit(
    num_qubits: int, layers: int, topology: Sequence[tuple[int, int]] | None = None
) -> list[Layer]:
    if topology is None:
        topology = bricklayer_topology(num_qubits)
    gates: list[Layer] = []
    for _ in range(layers):
        for q in range(num_qubits):
            gates += [("X", (q,)), ("Z", (q,)), ("X", (q,))]
        gates += [("YY", pair) for pair in topology]
    return gates


def efficient_su2_circuit(
    num_qubits: int, layers: int, topology: Sequence[tuple[int, int]] | None = None
) -> list[Layer]:
    if topology is None:
        topology = bricklayer_topology(num_qubits)
    gates: list[Layer] = []
    for _ in range(layers):
        for q in range(num_qubits):
            gates += [("Y", (q,)), ("Z", (q,))]
        gates += [("CNOT", pair) for pair in topology]
    return gates


def tfi_trotter_circuit(
    num_qubits: int, layers: int, topology: Sequence[tuple[int, int]] | None = None
) -> list[Layer]:
    if topology is None:
        topology = bricklayer_topology(num_qubits)
    zz_layer: list[Layer] = [("ZZ", pair) for pair in topology]
    x_layer: list[Layer] = [("X", (q,)) for q in range(num_qubits)]
    if layers == 0:
        return []
    gates = list(zz_layer)
    for _ in range(layers - 1):
        gates += x_layer + zz_layer
    return gates + x_layer


def tilted_tfi_trotter_circuit(
    num_qubits: int, layers: int, topology: Sequence[tuple[int, int]] | None = None
) -> list[Layer]:
    if topology is None:
        topology = bricklayer_topology(num_qubits)
    gates: list[Layer] = []
    for _ in range(layers):
        gates += [("ZZ", pair) for pair in topology]
        gates += [("Z", (q,)) for q in range(num_qubits)]
        gates += [("X", (q,)) for q in range(num_qubits)]
    return gates


def is_rotation(gate: Layer) -> bool:
    return not callable(gate) and gate[0] != "CNOT"


def count_parameters(gates: Sequence[Layer]) -> int:
    return sum(is_rotation(gate) for gate in gates)


def bind(num_qubits: int, gates: Sequence[Layer], thetas: Sequence[float]) -> pp.Circuit:
    """The circuit of `gates`, with the angles of its rotations taken from `thetas` in order."""
    assert len(thetas) == count_parameters(gates)
    angles = iter(thetas)
    circuit = pp.Circuit(num_qubits)
    for gate in gates:
        if callable(gate):
            gate(circuit)
        elif is_rotation(gate):
            paulis, qubits = gate
            circuit.rotation(paulis, list(qubits), next(angles))
        else:
            circuit.cx(*gate[1])
    return circuit


# Pauli strings as labels, where `label[q]` is the operator on qubit `q`.


def label(num_qubits: int, paulis: str, qubits: Sequence[int]) -> str:
    chars = ["I"] * num_qubits
    for pauli, q in zip(paulis, qubits, strict=True):
        chars[q] = pauli
    return "".join(chars)


def pauli_sum(num_qubits: int, terms: dict[str, float | complex], dtype=None) -> pp.PauliSum:
    """The sum with the given coefficients of labels."""
    if dtype is None:
        dtype = complex if any(isinstance(c, complex) for c in terms.values()) else float
    result = pp.PauliSum(num_qubits, dtype=dtype)
    for text, coeff in terms.items():
        assert len(text) == num_qubits
        result.add(text, list(range(num_qubits)), coeff)
    return result


def single(num_qubits: int, paulis: str, qubits: Sequence[int], coeff=1.0) -> pp.PauliSum:
    return pauli_sum(num_qubits, {label(num_qubits, paulis, qubits): coeff})


def computational_state(num_qubits: int, ones: Sequence[int] = ()) -> pp.PauliSum:
    """The basis state whose qubits in `ones` are 1: the product of (I + Z) / 2 or (I - Z) / 2."""
    terms = {}
    for chars in itertools.product("IZ", repeat=num_qubits):
        sign = (-1) ** sum(chars[q] == "Z" for q in ones)
        terms["".join(chars)] = sign / 2**num_qubits
    return pauli_sum(num_qubits, terms)


def weight(text: str) -> int:
    return len(text) - text.count("I")


def anticommute(a: str, b: str) -> bool:
    return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b, strict=True)) % 2 == 1


# The product of two single-qubit Pauli operators as (operator, power of i).
PRODUCT = {
    ("X", "Y"): ("Z", 1),
    ("Y", "Z"): ("X", 1),
    ("Z", "X"): ("Y", 1),
    ("Y", "X"): ("Z", 3),
    ("Z", "Y"): ("X", 3),
    ("X", "Z"): ("Y", 3),
}


def multiply(a: str, b: str) -> tuple[str, int]:
    """The product of two Pauli strings as (label, power of i)."""
    chars, power = [], 0
    for x, y in zip(a, b, strict=True):
        if x == "I" or y == "I" or x == y:
            chars.append("I" if x == y else (x if y == "I" else y))
        else:
            op, phase = PRODUCT[x, y]
            chars.append(op)
            power += phase
    return "".join(chars), power % 4


# Clifford tables of PauliPropagation.jl.


def clifford_tables() -> dict[str, list[tuple[int, int]]]:
    """For each gate, the image and the sign of every Pauli string on its qubits.

    A Pauli string is numbered by its operators as base-4 digits with I, X, Y, Z as 0 to 3
    and the first qubit of the gate as the lowest digit. A name ending in `_transpose` is
    the action of the gate on a state.
    """
    tables = json.loads((REFERENCE_DIR / "tables" / "clifford_maps.json").read_text())
    return {name: [tuple(entry) for entry in table] for name, table in tables.items()}


def table_image(table: Sequence[tuple[int, int]], qubits: Sequence[int], text: str) -> tuple[str, int]:
    """The image of a label under a table whose gate acts on `qubits`, and its sign."""
    index = sum(PAULIS.index(text[q]) << (2 * i) for i, q in enumerate(qubits))
    image, sign = table[index]
    chars = list(text)
    for i, q in enumerate(qubits):
        chars[q] = PAULIS[(image >> (2 * i)) & 3]
    return "".join(chars), sign


# Dense matrices, with qubit 0 as the most significant bit of an index.

MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
H_MATRIX = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
T_MATRIX = np.diag([1, cmath.exp(1j * math.pi / 4)])
CNOT_MATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def label_matrix(text: str) -> np.ndarray:
    matrix = np.eye(1, dtype=complex)
    for char in text:
        matrix = np.kron(matrix, MATRICES[char])
    return matrix


def rotation_matrix(num_qubits: int, paulis: str, qubits: Sequence[int], theta: float) -> np.ndarray:
    """exp(-i theta/2 P) on a register of `num_qubits` qubits."""
    generator = label_matrix(label(num_qubits, paulis, qubits))
    return math.cos(theta / 2) * np.eye(2**num_qubits) - 1j * math.sin(theta / 2) * generator


def embed(num_qubits: int, matrix: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """A matrix on `qubits`, whose first qubit is its most significant bit, on the register."""
    k = len(qubits)
    full = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for column in range(2**num_qubits):
        bits = [(column >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        local = sum(bits[q] << (k - 1 - i) for i, q in enumerate(qubits))
        for local_row in range(2**k):
            amplitude = matrix[local_row, local]
            if amplitude == 0:
                continue
            new_bits = list(bits)
            for i, q in enumerate(qubits):
                new_bits[q] = (local_row >> (k - 1 - i)) & 1
            row = sum(bit << (num_qubits - 1 - q) for q, bit in enumerate(new_bits))
            full[row, column] += amplitude
    return full


def circuit_matrix(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """The unitary of gates given in the order they act."""
    total = np.eye(matrices[0].shape[0], dtype=complex)
    for matrix in matrices:
        total = matrix @ total
    return total


def indexed_label(num_qubits: int, index: int) -> str:
    """The Pauli string whose operator on qubit `q` is digit `q` of `index` in base 4."""
    return "".join(PAULIS[(index >> (2 * q)) & 3] for q in range(num_qubits))


def transfer_matrix(*kraus: np.ndarray) -> list[list[float]]:
    """The action of a unitary, or of a channel with these Kraus operators, on an observable.

    Column `j` holds the coefficients of `sum_k K_k† P_j K_k` in the Pauli basis.
    """
    num_qubits = kraus[0].shape[0].bit_length() - 1
    paulis = [label_matrix(indexed_label(num_qubits, i)) for i in range(4**num_qubits)]
    evolved = [sum(k.conj().T @ pauli @ k for k in kraus) for pauli in paulis]
    return [
        [float(np.trace(row @ column).real) / 2**num_qubits for column in evolved]
        for row in paulis
    ]


# A propagation that follows the rules a term at a time, on labels.


def reference_propagate(
    num_qubits: int,
    gates: Sequence[Layer],
    thetas: Sequence[float],
    terms: dict[str, float],
    *,
    max_weight: int | None = None,
    min_abs_coeff: float = 0.0,
    max_freq: int | None = None,
    max_sins: int | None = None,
) -> dict[str, float]:
    """Propagates an observable backwards through rotations and CNOT gates.

    Every term carries the number of rotations that split it and the number of sine
    branches it took. Where two paths meet, the coefficients add and each count takes the
    smaller value.
    """
    cnot_table = clifford_tables()["CNOT"]
    paths = {text: (coeff, 0, 0) for text, coeff in terms.items()}

    def merge(into: dict, text: str, path: tuple[float, int, int]) -> None:
        if text in into:
            coeff, freq, sins = into[text]
            path = (coeff + path[0], min(freq, path[1]), min(sins, path[2]))
        into[text] = path

    def dropped(text: str, path: tuple[float, int, int]) -> bool:
        coeff, freq, sins = path
        return (
            abs(coeff) < min_abs_coeff
            or (max_weight is not None and weight(text) > max_weight)
            or (max_freq is not None and freq > max_freq)
            or (max_sins is not None and sins > max_sins)
        )

    angles = list(thetas)
    for gate in reversed(gates):
        after: dict[str, tuple[float, int, int]] = {}
        if is_rotation(gate):
            theta = angles.pop()
            generator = label(num_qubits, *gate)
            for text in sorted(paths):
                coeff, freq, sins = paths[text]
                if not anticommute(generator, text):
                    merge(after, text, (coeff, freq, sins))
                    continue
                # U† P U = cos(theta) P + sin(theta) i G P.
                image, power = multiply(generator, text)
                sign = {3: 1.0, 1: -1.0}[power]
                merge(after, text, (coeff * math.cos(theta), freq + 1, sins))
                merge(after, image, (coeff * math.sin(theta) * sign, freq + 1, sins + 1))
            truncates = True
        else:
            assert gate[0] == "CNOT"
            for text in sorted(paths):
                coeff, freq, sins = paths[text]
                image, sign = table_image(cnot_table, gate[1], text)
                merge(after, image, (coeff * sign, freq, sins))
            # The gate changes no coefficient, so it truncates only by what it changes.
            truncates = max_weight is not None
        paths = {
            text: path for text, path in after.items() if not (truncates and dropped(text, path))
        }
    assert not angles
    return {text: path[0] for text, path in paths.items()}
