"""Turns the noise tables of a QDK ``NoiseConfig`` into Pauli channels."""

from __future__ import annotations

from itertools import product

# Names of the noise tables of a NoiseConfig, with the number of qubits of their gate.
GATE_QUBITS = {
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "s_adj": 1,
    "t": 1,
    "t_adj": 1,
    "sx": 1,
    "sx_adj": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "cx": 2,
    "cy": 2,
    "cz": 2,
    "rxx": 2,
    "ryy": 2,
    "rzz": 2,
    "swap": 2,
}


class NoiseModel:
    """The Pauli errors that follow each gate, read once from a ``NoiseConfig``.

    The QDK simulators draw one error from the table of a gate after applying the gate, and
    character ``i`` of an error string acts on qubit argument ``i`` of the gate. An entry
    with qubit loss has no counterpart as a Pauli channel and is rejected.
    """

    def __init__(self, noise_config) -> None:
        self._errors: dict[str, list[tuple[str, float]]] = {}
        for gate, num_qubits in GATE_QUBITS.items():
            table = getattr(noise_config, gate, None)
            if table is None:
                continue
            errors = []
            for chars in product("IXYZL", repeat=num_qubits):
                string = "".join(chars)
                probability = getattr(table, string.lower())
                if probability == 0 or set(string) == {"I"}:
                    continue
                if "L" in string:
                    raise ValueError(
                        f"the noise table of {gate} has qubit loss ({string}), "
                        "which is not supported"
                    )
                errors.append((string, probability))
            if errors:
                self._errors[gate] = errors

    def errors(self, gate: str) -> list[tuple[str, float]]:
        """The errors after ``gate`` as ``(Pauli string, probability)`` pairs."""
        return self._errors.get(gate, [])
