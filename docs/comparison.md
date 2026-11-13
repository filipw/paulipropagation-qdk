# Comparison with other Pauli propagation software

| Package | Version | What it is |
| --- | --- | --- |
| paulipropagation-qdk | | the feature set of PauliPropagation.jl for circuits compiled with the QDK, with a Rust engine on all CPU cores |
| [PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl) | 0.9.2 | the reference implementation of the method, in Julia |
| [pauli-prop](https://github.com/Qiskit/pauli-prop) | 0.2.1 | the propagation routine under the error mitigation add-ons of Qiskit |
| [cuPauliProp](https://github.com/NVIDIA/cuQuantum/tree/main/python/cuquantum/pauliprop) | cuQuantum 26.11 | building blocks for Pauli propagation on NVIDIA GPUs, without a circuit front end |

All four share the core of the method: a rotation splits a Pauli string that anticommutes with
its generator into two, equal strings are added up, small terms are dropped.

## Features

| | paulipropagation-qdk | PauliPropagation.jl | pauli-prop (Qiskit) | cuPauliProp (cuQuantum) |
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

- "Not stated": the documentation of the package does not say.
- cuPauliProp was not run. Its entries are from its Python interface,
  `cuquantum.pauliprop.experimental`.

## Specific to each

| Package | |
| --- | --- |
| paulipropagation-qdk | circuits from Q# and OpenQASM through QIR; a QDK `NoiseConfig` as Pauli channels on the gates that the QDK simulators apply them to; bit-for-bit results on any number of threads; registers of up to 512 qubits |
| PauliPropagation.jl | the widest range of methods, and where new ones appear first: imaginary-time evolution, visualization of Pauli sums, counting of terms during a propagation, an interface to Yao, registers of any size |
| pauli-prop | `propagate_through_operator`, which conjugates one Pauli-sum operator by another and generates only the largest contributions; the Pauli-Lindblad noise model that the Qiskit workflows learn from hardware; lightcone shading, propagated noise absorption and operator backpropagation are built on it |
| cuPauliProp | GPUs, and runs distributed over several processes; it offers the operations (apply a gate, truncate, sort, trace, differentiate backwards) and leaves the simulator to its user |

## Agreement

| With | Compared | Where |
| --- | --- | --- |
| PauliPropagation.jl | every term under every truncation, with noise, as states; gradients, surrogates, merged symmetries, error estimates; random cases | [verification.md](verification.md) |
| pauli-prop | every term on random circuits of Clifford gates and rotations without truncation, in both pictures | `tests/test_against_qiskit_pauli_prop.py` |
| cuPauliProp | not run | |

When comparing numbers with pauli-prop:

- At the same threshold, truncated results differ slightly, because the two apply it at
  different points of a gate. They approach each other as the threshold is lowered.
- Its largest number of terms counts terms before equal strings are added up, so it has to be
  set well above the size of the result if it is not meant to act.
- An operator that it returns can list a Pauli string more than once.

## Timings

Only where the same propagation can be run and the results agree. Best of three runs on an
Apple M4 Pro (eight performance cores).

### PauliPropagation.jl

Truncation by weight and by coefficient; both packages give the same number of terms and the
same expectation value. Julia in its fastest configuration for the number of threads.

```bash
julia -t 8 --project=tools/julia tools/julia/benchmark.jl
python benchmarks/run.py --threads 1,8
```

| Case | Terms | paulipropagation-qdk, 1 thread | Julia, 1 thread | paulipropagation-qdk, 8 threads | Julia, 8 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transverse-field Ising chain, 32 qubits, 32 layers | 1,671 | 1.5 ms | 1.6 ms | 1.5 ms | 1.9 ms |
| Kicked Ising on the 127-qubit heavy-hex lattice, 20 layers | 6,149 | 49 ms | 57 ms | 19 ms | 20 ms |
| Tilted-field Ising on 6 x 6 qubits, 14 layers | 1,036,142 | 0.66 s | 0.65 s | 0.11 s | 0.12 s |
| The same lattice with depolarizing noise, 12 layers | 36,935,368 | 17.5 s | 17.6 s | 3.4 s | 3.6 s |

### pauli-prop

Transverse-field Ising chain of 100 qubits, 20 steps, observable Z on one qubit, truncated by
the size of the coefficients. Only the propagation is timed.

```bash
python benchmarks/qiskit_pauli_prop.py
```

| Threshold | Terms, pauli-prop | Terms, paulipropagation-qdk | pauli-prop | paulipropagation-qdk, 1 thread | paulipropagation-qdk, 8 threads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 44,037 | 39,398 | 0.27 s | 0.023 s | 0.009 s |
| 1e-5 | 138,626 | 128,403 | 0.75 s | 0.069 s | 0.018 s |
| 1e-6 | 353,187 | 333,977 | 1.83 s | 0.175 s | 0.036 s |

- Expectation values agree to 6e-4 at the first threshold and to 3e-6 at the last.
- One circuit on one machine. pauli-prop is written for a different use, in which the largest
  number of terms bounds the work, not a threshold.

### cuPauliProp

No timings: it needs hardware that this comparison was not run on.
