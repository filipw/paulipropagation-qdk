"""Times the cuPauliProp library of NVIDIA cuQuantum on the benchmark circuits.

    pip install cuquantum-python-cu12
    python benchmarks/cupauliprop.py [--threads 40] [case ...]

Needs an NVIDIA GPU. Every case is propagated with cuPauliProp, under the truncation of the
case after every gate, and with this package on ``--threads`` threads, by default one per
processor. cuPauliProp is run in two ways: with buffers that it allocates for every gate, and
with two buffers allocated once that the gates write to in turn, as the samples of cuQuantum
do. Only the propagation is timed, and the best of three runs is kept.

Printed, and written to ``benchmarks/results/cupauliprop.json``, are the timings, the number of
terms, the overlap with the zero state and the norm of both results, and how far the two are
apart: the number of terms that only one of them has, the largest coefficient among those,
and the largest difference between the coefficients of the others. Results of more than two
million terms are compared by their number of terms, overlap and norm only.

Without names, every case is run but the large ones, which need tens of gigabytes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import cuquantum
import numpy as np
from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliNoiseChannel,
    PauliRotationGate,
    Truncation,
    get_num_packed_integers,
)
from run import LARGE, build_circuit

import paulipropagation_qdk as pp

ROOT = Path(__file__).parent
REPETITIONS = 3
# The largest result whose terms are compared one by one.
TERM_LIMIT = 2_000_000

# The Pauli errors of the noise gates of a case, with their probabilities at strength 1.
NOISE = {
    "DepolarizingNoise": {"X": 0.25, "Y": 0.25, "Z": 0.25},
    "PauliXNoise": {"X": 0.5},
    "PauliYNoise": {"Y": 0.5},
    "PauliZNoise": {"Z": 0.5},
}


def packed(strings: list[str], num_qubits: int) -> np.ndarray:
    """One row per Pauli string, of the bits that say where it has X or Y and then of those
    that say where it has Z or Y, in 64-bit words with qubit 0 in the lowest bit."""
    words = get_num_packed_integers(num_qubits)
    letters = np.frombuffer("".join(strings).encode(), dtype=np.uint8).reshape(-1, num_qubits)
    bits = np.zeros((len(strings), 2, 64 * words), dtype=np.uint8)
    bits[:, 0, :num_qubits] = (letters == ord("X")) | (letters == ord("Y"))
    bits[:, 1, :num_qubits] = (letters == ord("Z")) | (letters == ord("Y"))
    return np.packbits(bits, axis=2, bitorder="little").view("<u8").reshape(len(strings), -1)


def dense(paulis: str, qubits: list[int], num_qubits: int) -> str:
    letters = ["I"] * num_qubits
    for pauli, qubit in zip(paulis, qubits):
        letters[qubit] = pauli
    return "".join(letters)


def gates_of(case) -> list:
    gates = []
    for gate in case["circuit"]:
        kind, qubits = gate["gate"], gate["qubits"]
        if kind == "rotation":
            gates.append(PauliRotationGate(gate["theta"], list(gate["paulis"]), qubits))
        elif kind in NOISE:
            errors = {pauli: gate["strength"] * share for pauli, share in NOISE[kind].items()}
            errors["I"] = 1.0 - sum(errors.values())
            gates.append(PauliNoiseChannel(qubits, errors))
        else:
            raise ValueError(f"no benchmark mapping for the gate {kind}")
    return gates


class Propagation:
    """The propagation of the observable of a case through its circuit with cuPauliProp."""

    def __init__(self, handle: LibraryHandle, case) -> None:
        self.handle = handle
        self.num_qubits = case["num_qubits"]
        self.gates = gates_of(case)
        self.truncation = Truncation(
            pauli_weight_cutoff=case["max_weight"], pauli_coeff_cutoff=case["min_abs_coeff"]
        )
        self.options = PauliExpansionOptions(memory_limit="80%", blocking=True)
        strings = [dense(t["paulis"], t["qubits"], self.num_qubits) for t in case["observable"]]
        self.xz = packed(strings, self.num_qubits)
        self.coeffs = np.array([t["coeff"] for t in case["observable"]], dtype=np.float64)

    def allocating(self) -> tuple[float, PauliExpansion, int]:
        """With buffers that cuPauliProp allocates for every gate. Returns the seconds, the
        result and the largest number of terms after a gate."""
        expansion = PauliExpansion(
            self.handle,
            self.num_qubits,
            len(self.coeffs),
            cp.asarray(self.xz),
            cp.asarray(self.coeffs),
            options=self.options,
        )
        most = expansion.num_terms
        start = time.perf_counter()
        # The observable passes the gates from the last one to the first.
        for gate in reversed(self.gates):
            expansion = expansion.apply_gate(gate, truncation=self.truncation, adjoint=True)
            most = max(most, expansion.num_terms)
        cp.cuda.get_current_stream().synchronize()
        return time.perf_counter() - start, expansion, most

    def capacity(self, most: int) -> int:
        """The number of terms that a buffer has to hold when no gate gets more than ``most``."""
        rehearsal = PauliExpansion.empty(
            self.handle, self.num_qubits, most, dtype="float64", options=self.options
        )
        required = 0
        for gate in self.gates:
            info = rehearsal.apply_gate(gate, truncation=self.truncation, adjoint=True)
            required = max(required, info.num_terms_required)
        return required

    def preallocated(self, capacity: int) -> tuple[float, PauliExpansion]:
        """With two buffers of ``capacity`` terms that the gates write to in turn."""
        expansions = []
        for num_terms in (len(self.coeffs), 0):
            xz = cp.zeros((capacity, self.xz.shape[1]), dtype=cp.uint64)
            coeffs = cp.zeros(capacity, dtype=cp.float64)
            if num_terms:
                xz[:num_terms] = cp.asarray(self.xz)
                coeffs[:num_terms] = cp.asarray(self.coeffs)
            expansions.append(
                PauliExpansion(
                    self.handle, self.num_qubits, num_terms, xz, coeffs, options=self.options
                )
            )
        source, target = expansions
        start = time.perf_counter()
        for gate in reversed(self.gates):
            source.apply_gate(gate, truncation=self.truncation, expansion_out=target, adjoint=True)
            source, target = target, source
        cp.cuda.get_current_stream().synchronize()
        return time.perf_counter() - start, source


def release() -> None:
    cp.get_default_memory_pool().free_all_blocks()


def difference(ours: pp.PauliSum, xz: np.ndarray, coeffs: np.ndarray) -> dict:
    """How far the terms of the result of this package and those of cuPauliProp are apart."""
    strings, values = zip(*ours.terms())
    mine = dict(zip((row.tobytes() for row in packed(strings, ours.num_qubits)), values))
    theirs = dict(zip((row.tobytes() for row in xz), coeffs.tolist()))
    common = mine.keys() & theirs.keys()
    unmatched = [
        abs(c) for terms in (mine, theirs) for key, c in terms.items() if key not in common
    ]
    return {
        "only_this_package": len(mine) - len(common),
        "only_cupauliprop": len(theirs) - len(common),
        "largest_unmatched_coeff": max(unmatched, default=0.0),
        "largest_coeff_difference": max(
            (abs(mine[key] - theirs[key]) for key in common), default=0.0
        ),
    }


def run(handle: LibraryHandle, name: str, threads: int | None) -> dict:
    case = json.loads((ROOT / "cases" / f"{name}.json").read_text())
    propagation = Propagation(handle, case)

    observable = pp.PauliSum(case["num_qubits"])
    for term in case["observable"]:
        observable.add(term["paulis"], term["qubits"], term["coeff"])
    ours = pp.propagate(
        build_circuit(case),
        observable,
        threads=threads,
        max_weight=case["max_weight"],
        min_abs_coeff=case["min_abs_coeff"],
    )
    print(f"{name}: {case['num_qubits']} qubits, {len(propagation.gates)} gates")
    print(f"  {'this package':<28} {len(ours):>11} terms   overlap {ours.overlap_with_zero()!r}")

    # The first run is not timed: it tells how many terms the buffers have to hold.
    _, result, most = propagation.allocating()
    significand, exponent = result.trace_with_zero_state()
    coeffs = cp.asnumpy(result.valid_coeffs)
    report = {
        "gates": len(propagation.gates),
        "terms": result.num_terms,
        "most_terms": most,
        "overlap_with_zero": float(significand * np.exp2(exponent)),
        "norm": float(np.linalg.norm(coeffs)),
        "this_package": {
            "terms": len(ours),
            "overlap_with_zero": ours.overlap_with_zero(),
            "norm": ours.norm(),
        },
    }
    print(
        f"  {'cuPauliProp':<28} {report['terms']:>11} terms   overlap {report['overlap_with_zero']!r}"
    )
    if max(len(ours), result.num_terms) <= TERM_LIMIT:
        report["difference"] = difference(ours, cp.asnumpy(result.valid_xz_bits), coeffs)
        print(f"  {report['difference']}")
    del ours, result, coeffs
    release()

    capacity = propagation.capacity(most)
    report["capacity"] = capacity
    seconds = {"buffers per gate": float("inf"), "two buffers": float("inf")}
    for _ in range(REPETITIONS):
        elapsed, result, _ = propagation.allocating()
        seconds["buffers per gate"] = min(seconds["buffers per gate"], elapsed)
        del result
        release()
        elapsed, result = propagation.preallocated(capacity)
        seconds["two buffers"] = min(seconds["two buffers"], elapsed)
        if result.num_terms != report["terms"]:
            print(f"  MISMATCH: {result.num_terms} terms with two buffers")
        del result
        release()
    report["seconds"] = seconds
    for label, elapsed in seconds.items():
        print(f"  {'cuPauliProp, ' + label:<28} {elapsed:8.4f} s")
    return report


def gpu() -> str:
    query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    return subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    arguments = sys.argv[1:]
    threads = None
    if arguments and arguments[0] == "--threads":
        threads = int(arguments[1])
        arguments = arguments[2:]
    names = arguments or sorted(
        case.stem for case in (ROOT / "cases").glob("*.json") if case.stem not in LARGE
    )

    path = ROOT / "results" / "cupauliprop.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["version"] = cuquantum.__version__
    report["gpu"] = gpu()
    report["threads_of_this_package"] = threads or os.cpu_count()
    handle = LibraryHandle()
    failed = []
    for name in names:
        try:
            report.setdefault("cases", {})[name] = run(handle, name, threads)
        except Exception as error:  # a case that the GPU cannot hold must not end the others
            print(f"  FAILED: {type(error).__name__}: {error}")
            failed.append(name)
            release()
        path.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
