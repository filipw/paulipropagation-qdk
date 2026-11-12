"""Pauli propagation for circuits compiled with the Microsoft Quantum Development Kit.

An observable is evolved backwards through a circuit as a truncated sum of Pauli strings,
which gives expectation values without holding a quantum state::

    from qdk import qsharp
    import paulipropagation_qdk as pp

    qsharp.init(target_profile=qsharp.TargetProfile.Base)
    qsharp.eval(source)
    circuit = pp.circuit_from_qir(str(qsharp.compile("Main()")))

    observable = pp.PauliSum(circuit.num_qubits)
    observable.add("Z", [0])
    value = pp.expectation(circuit, observable, max_weight=6, min_abs_coeff=1e-4)
"""

from ._native import Circuit, PauliSum, propagate
from ._qir import circuit_from_qir

__all__ = ["Circuit", "PauliSum", "circuit_from_qir", "expectation", "propagate"]


def expectation(
    circuit: Circuit,
    observable: PauliSum,
    *,
    max_weight: int | None = None,
    min_abs_coeff: float = 1e-10,
) -> float:
    """The expectation value of ``observable`` after ``circuit`` acts on the all-zero state.

    The result is exact only if the truncation drops nothing; see :func:`propagate`.
    """
    evolved = propagate(
        circuit, observable, max_weight=max_weight, min_abs_coeff=min_abs_coeff
    )
    return evolved.overlap_with_zero()
