"""Times this package and the pauli-prop package of Qiskit on the same propagation.

    pip install pauli-prop
    python benchmarks/qiskit_pauli_prop.py

The circuit is the evolution of a transverse-field Ising chain of 100 qubits in 20 steps,
and the observable is Z on qubit 50. The only truncation that both packages have is the one
by the size of the coefficients, so that is what is compared, at three thresholds. Only the
propagation is timed, with the circuit already in the form that each package works on, and
the best of three runs is printed. pauli-prop works on one thread.
"""

from __future__ import annotations

import time

import numpy as np
from pauli_prop import circuit_to_rotation_gates, propagate_through_rotation_gates
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import paulipropagation_qdk as pp

QUBITS, LAYERS, SITE = 100, 20, 50
COUPLING, FIELD = 0.2, 0.3
REPETITIONS = 3
# pauli-prop needs a largest number of terms, which here is far above what is reached.
MAX_TERMS = 2_000_000


def best_of(run) -> tuple[float, object]:
    best, result = float("inf"), None
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - start)
    return best, result


def main() -> None:
    ours, theirs = pp.Circuit(QUBITS), QuantumCircuit(QUBITS)
    for _ in range(LAYERS):
        for q in range(QUBITS - 1):
            ours.rzz(COUPLING, q, q + 1)
            theirs.rzz(COUPLING, q, q + 1)
        for q in range(QUBITS):
            ours.rx(FIELD, q)
            theirs.rx(FIELD, q)
    observable = pp.PauliSum(QUBITS)
    observable.add("Z", [SITE])
    operator = SparsePauliOp.from_sparse_list([("Z", [SITE], 1.0)], num_qubits=QUBITS)
    rotations = circuit_to_rotation_gates(theirs)

    print(f"{'threshold':>9}  {'package':<28} {'terms':>8}  {'<Z>':>13}  {'seconds':>8}")
    for threshold in (1e-4, 1e-5, 1e-6):
        seconds, (evolved, _) = best_of(
            lambda: propagate_through_rotation_gates(operator, rotations, MAX_TERMS, threshold, "h")
        )
        diagonal = ~evolved.paulis.x.any(axis=1)
        value = float(np.real(evolved.coeffs[diagonal].sum()))
        print(f"{threshold:>9g}  {'pauli-prop':<28} {len(evolved):>8}  {value:>13.9f}  {seconds:>8.3f}")
        for threads in (1, 8):
            seconds, evolved = best_of(
                lambda: pp.propagate(ours, observable, min_abs_coeff=threshold, threads=threads)
            )
            name = f"this package, {threads} thread{'s' if threads > 1 else ''}"
            print(f"{threshold:>9g}  {name:<28} {len(evolved):>8}  {evolved.overlap_with_zero():>13.9f}  {seconds:>8.3f}")


if __name__ == "__main__":
    main()
