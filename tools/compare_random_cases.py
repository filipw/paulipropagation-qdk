"""Compares this package with PauliPropagation.jl on random circuits, term by term.

    python tools/compare_random_cases.py --first 0 --count 100000 --jobs 8

``tools/julia/random_cases.jl`` makes a case from every seed and propagates it with
PauliPropagation.jl. The same case is propagated here, and the two results must agree in
every coefficient, in the overlaps, and where the case has them in the sum with its
translations merged and in the gradient. Julia has to be installed, with the environment of
``tools/julia`` instantiated.

The seeds are divided among ``--jobs`` Julia processes. A case that differs is written to
``--failures`` as it was read, and the exit code is 1 if there is one.
"""

import argparse
import collections
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_against_julia import build_circuit, number  # noqa: E402

import paulipropagation_qdk as pp  # noqa: E402

# Two coefficients agree if they differ by no more than this times the largest coefficient
# of the reference, or times 1 if that is smaller. A number that adds up several terms, like
# an overlap, may differ by that for every term.
TOLERANCE = 1e-11


def start_sum(case) -> pp.PauliSum:
    coeffs = [number(term["coeff"]) for term in case["observable"]]
    dtype = complex if any(isinstance(c, complex) for c in coeffs) else float
    start = pp.PauliSum(case["num_qubits"], dtype=dtype)
    for term, coeff in zip(case["observable"], coeffs):
        start.add(term["paulis"], term["qubits"], coeff)
    return start


class Comparison:
    """Collects what differs by more than a tolerance, and the largest difference seen."""

    def __init__(self, counts: collections.Counter, scale: float) -> None:
        self.counts, self.scale, self.problems = counts, scale, []

    def check(self, kind: str, name: str, ours, theirs, slack: float = 1.0) -> None:
        """Compares two numbers. `slack` is the number of terms that were added up to them."""
        difference = abs(ours - theirs) / self.scale
        key = f"largest difference of {kind}"
        self.counts[key] = max(self.counts[key], difference)
        if not difference <= TOLERANCE * slack:
            self.problems.append(f"{name}: {ours!r} here, {theirs!r} there")

    def check_terms(self, kind: str, ours: dict, theirs: dict, slack: float = 1.0) -> None:
        for label in ours.keys() | theirs.keys():
            self.check(kind, label, ours.get(label, 0.0), theirs.get(label, 0.0), slack)


def compared_derivatives(case) -> list[int]:
    """The places, among the rotations of the circuit here, of the angles that the reference
    differentiates. A T gate is a rotation here, and the reference does not differentiate a
    rotation whose angle is frozen."""
    places, rotations = [], 0
    for gate in case["circuit"]:
        if gate["gate"] == "rotation" and not gate.get("frozen"):
            places.append(rotations)
        if gate["gate"] in ("rotation", "T"):
            rotations += 1
    return places


def check(case, counts: collections.Counter) -> list[str]:
    """What differs between the reference result of the case and the result here."""
    circuit = build_circuit(case, matrices_as_unitaries=case["seed"] % 2 == 0)
    start = start_sum(case)
    damping = case["weight_damping"]
    result = pp.propagate(
        circuit,
        start,
        max_weight=case["max_weight"],
        min_abs_coeff=case["min_abs_coeff"],
        min_rel_coeff=case["min_rel_coeff"],
        max_freq=case["max_freq"],
        max_sins=case["max_sins"],
        weight_damping=tuple(damping) if damping else None,
        heisenberg=case["heisenberg"],
    )

    theirs = {label: number(coeff) for label, coeff in case["terms"]}
    comparison = Comparison(counts, max([1.0, *map(abs, theirs.values())]))
    comparison.check_terms("a coefficient", dict(result.terms()), theirs)
    counts["terms"] += len(theirs)

    overlaps = {
        "overlap_with_zero": result.overlap_with_zero(),
        "overlap_with_plus": result.overlap_with_plus(),
        "overlap_with_computational": result.overlap_with_computational(case["ones"]),
        "overlap_with_max_mixed": result.overlap_with_max_mixed(),
    }
    added = max(1, len(theirs))
    for name, value in overlaps.items():
        comparison.check("an overlap", name, value, number(case[name]), added)

    if "merged" in case:
        grid = case["grid"]
        merged = result.translation_merge(*grid) if grid else result.translation_merge()
        expected = {label: number(coeff) for label, coeff in case["merged"]}
        # A merged coefficient adds up to one term for every translation.
        comparison.check_terms("a merged coefficient", dict(merged.terms()), expected, case["num_qubits"])
        counts["merged sums"] += 1

    if "gradient" in case:
        reference = case["gradient"]
        value, derivatives = pp.gradient(
            circuit,
            start,
            state=reference["state"],
            max_weight=case["max_weight"],
            min_abs_coeff=case["min_abs_coeff"],
        )
        places = compared_derivatives(case)
        comparison.check("an overlap", "gradient value", value, reference["value"], added)
        if len(places) != len(reference["derivatives"]):
            comparison.problems.append("gradient: a different number of angles")
        else:
            for place, expected in zip(places, reference["derivatives"]):
                comparison.check("a derivative", f"derivative {place}", derivatives[place], expected, added)
        counts["gradients"] += 1
        counts["derivatives"] += len(places)

    counts["cases"] += 1
    counts["observables" if case["heisenberg"] else "states"] += 1
    counts["gates"] += len(case["circuit"])
    for gate in case["circuit"]:
        counts["gate " + gate["gate"]] += 1
    for option in ("max_weight", "min_rel_coeff", "max_freq", "max_sins", "weight_damping"):
        counts["with " + option] += case[option] is not None
    counts["with min_abs_coeff"] += case["min_abs_coeff"] > 0
    counts["complex"] += result.dtype is complex
    counts["on more than 32 qubits"] += case["num_qubits"] > 32
    counts["largest sum"] = max(counts["largest sum"], len(theirs))
    return comparison.problems


def check_file(path: Path, counts: collections.Counter, failures) -> None:
    with path.open() as lines:
        for line in lines:
            case = json.loads(line)
            if "error" in case:
                counts["not propagated by the reference: " + case["error"][:120]] += 1
                continue
            try:
                problems = check(case, counts)
            except Exception as err:  # a case that cannot be run here is a difference
                problems = [f"{type(err).__name__}: {err}"]
            if problems:
                counts["different"] += 1
                print(f"seed {case['seed']}: {len(problems)} differences, first: {problems[0]}", flush=True)
                failures.write(json.dumps({"seed": case["seed"], "problems": problems[:20], "case": case}) + "\n")
                failures.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=0, help="the first seed")
    parser.add_argument("--count", type=int, default=10_000, help="the number of cases")
    parser.add_argument("--jobs", type=int, default=4, help="the number of Julia processes")
    parser.add_argument("--per-file", type=int, default=1000, help="the number of cases in a file")
    parser.add_argument("--growth", type=int, default=13, help="see tools/julia/random_cases.jl")
    parser.add_argument("--failures", type=Path, default=Path("random_case_failures.jsonl"))
    parser.add_argument("--julia", default="julia", help="the Julia executable")
    args = parser.parse_args()

    counts: collections.Counter = collections.Counter()
    started = time.monotonic()
    share = -(-args.count // args.jobs)
    with tempfile.TemporaryDirectory() as directory, args.failures.open("w") as failures:
        workers = []
        for job in range(args.jobs):
            first = args.first + job * share
            count = min(share, args.first + args.count - first)
            if count <= 0:
                break
            command = [args.julia, f"--project={ROOT / 'tools' / 'julia'}", "--threads=1"]
            command += [str(ROOT / "tools" / "julia" / "random_cases.jl")]
            command += [str(first), str(count), str(args.per_file), directory, str(args.growth)]
            workers.append(subprocess.Popen(command))

        reported = time.monotonic()
        while True:
            running = [worker for worker in workers if worker.poll() is None]
            complete = sorted(Path(directory).glob("*.jsonl"))
            for path in complete:
                check_file(path, counts, failures)
                path.unlink()
            if not running and not complete:
                break
            if not complete:
                time.sleep(0.5)
            if time.monotonic() - reported > 60:
                reported = time.monotonic()
                print(f"{counts['cases']} cases, {counts['different']} different", flush=True)
        failed = [worker.returncode for worker in workers if worker.returncode != 0]

    print(f"seeds {args.first} to {args.first + args.count - 1} in {time.monotonic() - started:.0f} s")
    for name, value in sorted(counts.items()):
        print(f"  {name}: {value:.3g}" if isinstance(value, float) else f"  {name}: {value}")
    if failed:
        print(f"{len(failed)} Julia processes failed")
    different = counts["different"]
    print("no differences" if not different else f"{different} cases differ, see {args.failures}")
    return 1 if different or failed else 0


if __name__ == "__main__":
    sys.exit(main())
