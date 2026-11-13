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

from collections.abc import Callable

from ._native import (
    Circuit,
    PauliSum,
    Surrogate,
    estimate_mse,
    gradient,
    mcpropagate,
    mcsample,
    propagate,
    resample,
)
from ._qir import circuit_from_qir

__all__ = [
    "Circuit",
    "PauliSum",
    "Surrogate",
    "circuit_from_qir",
    "estimate_mse",
    "expectation",
    "gradient",
    "mcpropagate",
    "mcsample",
    "propagate",
    "resample",
]


def expectation(
    circuit: Circuit,
    observable: PauliSum,
    *,
    max_weight: int | None = None,
    min_abs_coeff: float = 1e-10,
    min_rel_coeff: float | None = None,
    max_freq: int | None = None,
    max_sins: int | None = None,
    weight_damping: tuple[float, float] | None = None,
    custom_truncation: Callable[[str, float | complex], bool] | None = None,
    max_terms: int | None = None,
    threads: int | None = None,
) -> float | complex:
    """The expectation value of ``observable`` after ``circuit`` acts on the all-zero state.

    The result is exact only if the truncations drop nothing; see :func:`propagate` for
    what each of them drops, and for ``threads``.
    """
    evolved = propagate(
        circuit,
        observable,
        max_weight=max_weight,
        min_abs_coeff=min_abs_coeff,
        min_rel_coeff=min_rel_coeff,
        max_freq=max_freq,
        max_sins=max_sins,
        weight_damping=weight_damping,
        custom_truncation=custom_truncation,
        max_terms=max_terms,
        threads=threads,
    )
    return evolved.overlap_with_zero()
