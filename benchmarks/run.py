"""Times this package on the benchmark circuits that ``tools/julia/benchmark.jl`` writes.

    python benchmarks/run.py [--threads 1,8] [case ...]

Prints the best of three runs for every case and number of threads next to the timings of
PauliPropagation.jl from ``benchmarks/results/julia.json``: those on one thread next to ours
on one thread, the others next to ours on the largest number of threads. Checks that the
number of terms and the overlap with the zero state are the same on any number of threads
and agree with the Julia results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import paulipropagation_qdk as pp

ROOT = Path(__file__).parent
REPETITIONS = 3

NOISE = {
    "DepolarizingNoise": "depolarizing",
    "PauliXNoise": "pauli_x_noise",
    "PauliYNoise": "pauli_y_noise",
    "PauliZNoise": "pauli_z_noise",
    "AmplitudeDampingNoise": "amplitude_damping",
}


def build_circuit(case) -> pp.Circuit:
    circuit = pp.Circuit(case["num_qubits"])
    for gate in case["circuit"]:
        kind, qubits = gate["gate"], gate["qubits"]
        if kind == "rotation":
            circuit.rotation(gate["paulis"], qubits, gate["theta"])
        elif kind in NOISE:
            getattr(circuit, NOISE[kind])(gate["strength"], qubits[0])
        else:
            raise ValueError(f"no benchmark mapping for the gate {kind}")
    return circuit


def run(name: str, julia: dict | None, thread_counts: list[int]) -> None:
    case = json.loads((ROOT / "cases" / f"{name}.json").read_text())
    circuit = build_circuit(case)
    observable = pp.PauliSum(case["num_qubits"])
    for term in case["observable"]:
        observable.add(term["paulis"], term["qubits"], term["coeff"])
    truncation = {"max_weight": case["max_weight"], "min_abs_coeff": case["min_abs_coeff"]}

    best = {}
    outcomes = set()
    for threads in thread_counts:
        best[threads] = float("inf")
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            result = pp.propagate(circuit, observable, threads=threads, **truncation)
            best[threads] = min(best[threads], time.perf_counter() - start)
        outcomes.add((len(result), result.overlap_with_zero()))
        del result
    terms, overlap = next(iter(outcomes))

    print(f"{name}: {case['num_qubits']} qubits, {len(circuit)} gates, {terms} terms")
    for threads in thread_counts:
        label = f"this package, {threads} thread{'s' if threads > 1 else ''}"
        print(f"  {label:<28} {best[threads]:8.4f} s   overlap {overlap!r}")
    if len(outcomes) > 1:
        print(f"  MISMATCH between numbers of threads: {sorted(outcomes)}")
    if julia is None:
        return
    for label, seconds in sorted(julia["seconds"].items(), key=lambda item: -item[1]):
        ours = best[1] if "1 thread" in label and 1 in best else best[max(best)]
        print(f"  {'julia: ' + label:<28} {seconds:8.4f} s   {seconds / ours:6.2f} x")
    if terms != julia["terms"]:
        print(f"  MISMATCH: {julia['terms']} terms in the Julia result")
    if abs(overlap - julia["overlap_with_zero"]) > 1e-10:
        print(f"  MISMATCH: overlap {julia['overlap_with_zero']:.15g} in the Julia result")


def main() -> None:
    results = ROOT / "results" / "julia.json"
    julia = json.loads(results.read_text())["cases"] if results.exists() else {}
    arguments = sys.argv[1:]
    thread_counts = [1, 8]
    if arguments and arguments[0] == "--threads":
        thread_counts = [int(count) for count in arguments[1].split(",")]
        arguments = arguments[2:]
    names = arguments or sorted(path.stem for path in (ROOT / "cases").glob("*.json"))
    for name in names:
        run(name, julia.get(name), thread_counts)


if __name__ == "__main__":
    main()
