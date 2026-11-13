# How this package compares with other Pauli propagation software

Pauli propagation is implemented in several packages. This page says what each of them is
for, where they overlap with this one, and where they do not. It covers
[PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl) 0.9.2, the
Qiskit package [pauli-prop](https://github.com/Qiskit/pauli-prop) 0.2.1, and
[cuPauliProp](https://github.com/NVIDIA/cuQuantum/tree/main/python/cuquantum/pauliprop) of
NVIDIA cuQuantum 26.11.

## In short

- **PauliPropagation.jl** is the reference implementation of the method, in Julia. This
  package follows its algorithms, conventions and defaults, and its results are checked
  against it term by term. Use it if you work in Julia.
- **pauli-prop** is the propagation routine under the error mitigation add-ons of Qiskit. It
  takes Qiskit circuits, truncates by the size of coefficients and by a largest number of
  terms, and has a routine of its own for conjugating one operator by another.
- **cuPauliProp** is a library of building blocks for Pauli propagation on NVIDIA GPUs. It has
  no circuit front end; a simulator is written on top of it.
- **This package** brings the feature set of PauliPropagation.jl to circuits compiled with the
  QDK, from Q# or OpenQASM through QIR, with a Rust engine that uses all cores of a CPU.

All four share the core of the method: a Pauli rotation turns a Pauli string that anticommutes
with its generator into two, equal strings are added up, and small terms are dropped. They
differ in the circuits they read, the truncations and methods around that core, and the
hardware they run on.

## Features

| | This package | PauliPropagation.jl | pauli-prop (Qiskit) | cuPauliProp (cuQuantum) |
| --- | --- | --- | --- | --- |
| Interface | Python, Rust | Julia | Python | Python, C |
| Engine | Rust | Julia | Rust | CUDA library, distributed as a binary |
| Runs on | CPU, all cores | CPU, all cores; a CUDA extension for rotations | CPU, one thread | NVIDIA GPU |
| Platforms | macOS, Linux, Windows | wherever Julia runs | macOS, Linux, Windows | Linux with an NVIDIA GPU |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | BSD-3-Clause for the Python layer |
| Circuits from | QIR (Q#, OpenQASM through the QDK), or built gate by gate | built gate by gate; Yao circuits through an extension | Qiskit `QuantumCircuit` | none: gates are applied one by one |
| Clifford gates | yes | yes | yes, moved to one end of the circuit first | yes |
| Pauli rotations | on any qubits | on any qubits | on any qubits | on any qubits |
| Gates from matrices | unitaries and transfer matrices on up to 5 qubits | unitaries and transfer matrices | no | no |
| Noise | Pauli channels on up to 6 qubits, depolarizing, dephasing, amplitude damping, any channel as a transfer matrix; the Pauli errors of a QDK `NoiseConfig` | depolarizing, dephasing, Pauli noise, amplitude damping, transfer matrices | Pauli-Lindblad channels | Pauli channels on 1 or 2 qubits, amplitude damping |
| Observables and states | both pictures | both pictures | both pictures | both pictures |
| Complex coefficients | yes | yes | no | yes |
| Truncation by coefficient | absolute, and relative to the largest | absolute, and relative to the largest | absolute | absolute |
| Truncation by weight | yes, also as a damping of the coefficient | yes, also as a damping of the coefficient | no | yes |
| Truncation by path (`max_freq`, `max_sins`) | yes | yes | no | no |
| Largest number of terms | yes | no | yes | no |
| A rule of your own | yes | yes | no | no |
| Gradients for all angles in one pass | yes | yes | no | yes |
| Surrogate for many angles | yes | yes | no | no |
| Resampling and Monte Carlo paths | yes | yes | no | no |
| Estimate of the truncation error | yes | yes | sum of the dropped coefficients | no |
| Merging by symmetry | translations on rings and tori, or any function | translations, or any function | no | no |
| Imaginary-time evolution | no | yes | no | no |
| Operator conjugated by an operator | no | no | yes, with a largest number of terms | no |
| Same result on any number of threads | bit for bit | not stated | one thread | not stated |

"Not stated" means that the documentation of the package does not say. The entries for
cuPauliProp are taken from its Python interface, `cuquantum.pauliprop.experimental`; it was
not run for this comparison.

## What is specific to each

**This package.** Circuits come from the QDK: a Q# or OpenQASM program is compiled to QIR and
read from there, and a `NoiseConfig` of the QDK becomes Pauli channels on the gates that the
QDK simulators would apply them to, so that results can be checked against those simulators
on small circuits. The register is limited to 512 qubits. Every coefficient is the same bit
for bit on any number of threads. See the [README](../README.md) for how the results are
verified.

**PauliPropagation.jl.** The widest range of methods, and the place where new ones appear
first. It has what this package leaves out: imaginary-time evolution, a visualization of
Pauli sums, counting of terms during a propagation, an interface to Yao, and registers of any
size.

**pauli-prop.** Made for the error mitigation workflows of Qiskit: lightcone shading,
propagated noise absorption and operator backpropagation are built on it. Its
`propagate_through_operator` conjugates one Pauli-sum operator by another and generates only
the largest contributions, which none of the other packages have. Its noise model is the
Pauli-Lindblad model that those workflows learn from hardware. Two details matter when
comparing numbers with it: its largest number of terms counts terms before equal Pauli strings
are added up, so it has to be set well above the size of the result if it is not meant to
act, and an operator that it returns can list a Pauli string more than once.

**cuPauliProp.** The only one of the four that is built for GPUs, and for runs that are
distributed over several processes. It offers the operations of the method, applying a gate to an expansion, truncating,
sorting, taking traces and differentiating backwards, and leaves the simulator around them
to its user. It needs an NVIDIA GPU and Linux.

## Agreement

Results of this package are compared with two of the others in its tests:

- with PauliPropagation.jl term by term on circuits that it built, under every truncation,
  with noise, as states, with gradients, surrogates, merged symmetries and error estimates
  (`tests/test_against_julia.py`, `tests/test_features_against_julia.py`);
- with pauli-prop term by term on random circuits of Clifford gates and rotations without
  truncation, as observables and as states (`tests/test_against_qiskit_pauli_prop.py`, which
  runs where pauli-prop is installed).

Truncated results of pauli-prop and of this package differ slightly at the same threshold,
because the two apply it at different points of a gate, and they approach each other as the
threshold is lowered.

## Timings

Timings are given only where the same propagation can be run and the results agree.

**PauliPropagation.jl.** Four circuits with truncation by weight and by coefficient, on which
the two packages give the same number of terms and the same expectation value. Best of three
runs on an Apple M4 Pro, against the fastest configuration of PauliPropagation.jl for the
number of threads; `tools/julia/benchmark.jl` and `benchmarks/run.py` reproduce it.

| Case | Terms | This package, 1 thread | Julia, 1 thread | This package, 8 threads | Julia, 8 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transverse-field Ising chain, 32 qubits, 32 layers | 1,671 | 1.5 ms | 1.6 ms | 1.5 ms | 1.9 ms |
| Kicked Ising on the 127-qubit heavy-hex lattice, 20 layers | 6,149 | 49 ms | 57 ms | 19 ms | 20 ms |
| Tilted-field Ising on 6 x 6 qubits, 14 layers | 1,036,142 | 0.66 s | 0.65 s | 0.11 s | 0.12 s |
| The same lattice with depolarizing noise, 12 layers | 36,935,368 | 17.5 s | 17.6 s | 3.4 s | 3.6 s |

**pauli-prop.** One circuit, the evolution of a transverse-field Ising chain of 100 qubits in
20 steps with the observable Z on one qubit, truncated by the size of the coefficients, which
is the truncation that both packages have. Only the propagation is timed, with the circuit
already converted for each package. Best of three runs on the same machine;
`benchmarks/qiskit_pauli_prop.py` reproduces it.

| Threshold | Terms, pauli-prop | Terms, this package | pauli-prop | This package, 1 thread | This package, 8 threads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 44,037 | 39,398 | 0.27 s | 0.023 s | 0.009 s |
| 1e-5 | 138,626 | 128,403 | 0.75 s | 0.069 s | 0.018 s |
| 1e-6 | 353,187 | 333,977 | 1.83 s | 0.175 s | 0.036 s |

The numbers of terms differ by a few percent for the reason given above; the expectation
values agree to 6e-4 at the first threshold and to 3e-6 at the last. This is one circuit on
one machine, and pauli-prop is written for a different use, in which the largest number of
terms rather than a threshold bounds the work.

**cuPauliProp** needs hardware that this comparison was not run on, so there are no timings
for it.
