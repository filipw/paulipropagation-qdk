"""Reads a circuit from QIR, as the QDK compiles it for the base profile."""

from __future__ import annotations

import pyqir

from ._native import Circuit

_QIS = "__quantum__qis__"

# QIR intrinsic -> Circuit method, for gates whose arguments are all qubits.
_QUBIT_GATES = {
    "h__body": "h",
    "x__body": "x",
    "y__body": "y",
    "z__body": "z",
    "s__body": "s",
    "s__adj": "s_adj",
    "sx__body": "sx",
    "sx__adj": "sx_adj",
    "t__body": "t",
    "t__adj": "t_adj",
    "cx__body": "cx",
    "cnot__body": "cx",
    "cy__body": "cy",
    "cz__body": "cz",
    "swap__body": "swap",
    "ccx__body": "ccx",
}

# QIR intrinsic -> Circuit method, for gates whose first argument is an angle.
_ROTATIONS = {
    "rx__body": "rx",
    "ry__body": "ry",
    "rz__body": "rz",
    "rxx__body": "rxx",
    "ryy__body": "ryy",
    "rzz__body": "rzz",
}

# Intrinsics after which a qubit no longer takes part in the circuit.
_MEASUREMENTS = {"m__body", "mz__body", "mresetz__body", "reset__body"}

# Intrinsics that do nothing to the state.
_IGNORED = {"barrier__body"}


def _qubit(value: pyqir.Value, name: str) -> int:
    qubit = pyqir.ptr_id(value)
    if qubit is None:
        raise ValueError(
            f"{name} takes a qubit that is not a constant; "
            "only statically allocated qubits are supported"
        )
    return qubit


def circuit_from_qir(qir: str | bytes) -> Circuit:
    """Reads the circuit of a QIR program.

    ``qir`` is LLVM IR text or bitcode, for example ``str(qsharp.compile(...))`` with the
    base profile.

    The program has to be one unitary circuit, optionally followed by measurements: every
    gate is a Clifford gate, a T gate or a Pauli rotation with a constant angle, and no
    gate acts on a qubit after it was measured or reset. The trailing measurements are
    dropped, since an observable is evaluated on the state before them.

    :raises ValueError: if the program branches, measures mid-circuit, or uses an
        instruction that is not supported.
    """
    context = pyqir.Context()
    if isinstance(qir, (bytes, bytearray)):
        module = pyqir.Module.from_bitcode(context, bytes(qir))
    else:
        module = pyqir.Module.from_ir(context, str(qir))

    entry_points = [f for f in module.functions if pyqir.is_entry_point(f)]
    if len(entry_points) != 1:
        raise ValueError(
            f"expected one entry point in the QIR module, found {len(entry_points)}"
        )
    entry = entry_points[0]

    num_qubits = pyqir.required_num_qubits(entry)
    if num_qubits is None:
        raise ValueError("the entry point does not declare its number of qubits")

    circuit = Circuit(num_qubits)
    used: set[int] = set()
    finished: set[int] = set()

    def gate_qubits(values, name: str) -> list[int]:
        qubits = [_qubit(v, name) for v in values]
        for q in qubits:
            if q in finished:
                raise ValueError(
                    f"{name} acts on qubit {q} after it was measured or reset; "
                    "mid-circuit measurement and qubit reuse are not supported"
                )
        used.update(qubits)
        return qubits

    for block in entry.basic_blocks:
        terminator = block.terminator
        if terminator is not None and (
            isinstance(terminator, pyqir.Switch)
            or (terminator.opcode == pyqir.Opcode.BR and len(terminator.operands) > 1)
        ):
            raise ValueError("programs with branching control flow are not supported")

        for instruction in block.instructions:
            if not isinstance(instruction, pyqir.Call):
                if instruction.opcode in (pyqir.Opcode.BR, pyqir.Opcode.RET):
                    continue
                raise ValueError(
                    f"unsupported instruction in the entry point: {instruction}"
                )

            name = instruction.callee.name
            if name.startswith("__quantum__rt__"):
                continue
            if not name.startswith(_QIS):
                raise ValueError(f"unsupported call to {name}")
            intrinsic = name[len(_QIS) :]
            args = instruction.args

            if intrinsic in _QUBIT_GATES:
                getattr(circuit, _QUBIT_GATES[intrinsic])(*gate_qubits(args, name))
            elif intrinsic in _ROTATIONS:
                angle = args[0]
                if not isinstance(angle, pyqir.FloatConstant):
                    raise ValueError(f"{name} takes an angle that is not a constant")
                getattr(circuit, _ROTATIONS[intrinsic])(
                    angle.value, *gate_qubits(args[1:], name)
                )
            elif intrinsic in _MEASUREMENTS:
                qubit = _qubit(args[0], name)
                # Resetting a qubit that no gate has touched leaves it in the zero state.
                if intrinsic != "reset__body" or qubit in used:
                    finished.add(qubit)
            elif intrinsic in _IGNORED:
                continue
            else:
                raise ValueError(f"unsupported gate {name}")

    return circuit
