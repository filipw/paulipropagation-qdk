"""Times this package on the benchmark circuits that ``tools/julia/benchmark.jl`` writes.

    python benchmarks/run.py [--threads 1,8] [case ...]

Prints the best of three runs for every case and number of threads next to the timings of
PauliPropagation.jl from ``benchmarks/results/julia.json``: those on one thread next to ours
on one thread, the others next to ours on the largest number of threads. Checks that the
number of terms and the overlap with the zero state are the same on any number of threads
and agree with the Julia results. The timings, the number of terms and the overlap go to
``benchmarks/results/paulipropagation_qdk.json``.

Without names, every case is run but the large ones, which need tens of gigabytes, from the
smallest result to the largest: on a laptop a small case is slower right after a large one.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import paulipropagation_qdk as pp

ROOT = Path(__file__).parent
REPETITIONS = 3
LARGE = {"noisy_ising_6x6_large"}

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


def processor() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        query = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip()
    return platform.processor() or platform.machine()


def run(name: str, julia: dict | None, thread_counts: list[int]) -> dict:
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
    report = {
        "gates": len(circuit),
        "terms": terms,
        "overlap_with_zero": overlap,
        "seconds": {str(threads): best[threads] for threads in thread_counts},
    }
    if julia is None:
        return report
    for label, seconds in sorted(julia["seconds"].items(), key=lambda item: -item[1]):
        ours = best[1] if "1 thread" in label and 1 in best else best[max(best)]
        print(f"  {'julia: ' + label:<28} {seconds:8.4f} s   {seconds / ours:6.2f} x")
    if terms != julia["terms"]:
        print(f"  MISMATCH: {julia['terms']} terms in the Julia result")
    if abs(overlap - julia["overlap_with_zero"]) > 1e-10:
        print(f"  MISMATCH: overlap {julia['overlap_with_zero']:.15g} in the Julia result")
    return report


def main() -> None:
    results = ROOT / "results" / "julia.json"
    julia = json.loads(results.read_text())["cases"] if results.exists() else {}
    path = ROOT / "results" / "paulipropagation_qdk.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["version"] = version("paulipropagation-qdk")
    report["processor"] = processor()
    report["cores"] = os.cpu_count()
    arguments = sys.argv[1:]
    thread_counts = [1, 8]
    if arguments and arguments[0] == "--threads":
        thread_counts = [int(count) for count in arguments[1].split(",")]
        arguments = arguments[2:]
    names = arguments or sorted(
        (case.stem for case in (ROOT / "cases").glob("*.json") if case.stem not in LARGE),
        key=lambda name: (julia.get(name, {}).get("terms", 0), name),
    )
    for name in names:
        report.setdefault("cases", {})[name] = run(name, julia.get(name), thread_counts)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
