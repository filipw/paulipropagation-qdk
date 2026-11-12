# paulipropagation-qdk

Pauli propagation for circuits compiled with the Microsoft Quantum Development Kit (QDK).

An observable is evolved backwards through a circuit as a truncated sum of Pauli strings, which
gives expectation values `<0| U† O U |0>` without holding a quantum state. It is a Rust engine
with a Python API that reads circuits from QIR, and it follows the algorithms and conventions of
[PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl).

## Example

```python
from qdk import qsharp
import paulipropagation_qdk as pp

qsharp.init(target_profile=qsharp.TargetProfile.Base)
qsharp.eval("""
operation Main() : Result[] {
    use qs = Qubit[32];
    for _ in 1..8 {
        for i in 0..30 { Rzz(0.1, qs[i], qs[i + 1]); }
        for i in 0..31 { Rx(0.1, qs[i]); }
    }
    MResetEachZ(qs)
}
""")
circuit = pp.circuit_from_qir(str(qsharp.compile("Main()")))

observable = pp.PauliSum(circuit.num_qubits)
observable.add("Z", [15])

evolved = pp.propagate(circuit, observable, max_weight=6, min_abs_coeff=1e-4)
print(len(evolved), evolved.overlap_with_zero())
```

A circuit can also be built directly with `pp.Circuit(num_qubits)` and its gate methods.

### Noise

Noise channels are part of the circuit, and the result is then the expectation value of the
noisy circuit, computed exactly rather than sampled shot by shot. A `NoiseConfig` of the QDK
can be passed when reading QIR; its Pauli errors are added after each gate, as the QDK
simulators apply them:

```python
from qdk.simulation import NoiseConfig

noise = NoiseConfig()
noise.rzz.set_depolarizing(0.01)
noise.rx.set_pauli_noise([("Z", 0.002)])
circuit = pp.circuit_from_qir(str(qsharp.compile("Main()")), noise=noise)
```

Channels can also be added directly, for example `circuit.depolarizing(0.02, q)`,
`circuit.amplitude_damping(0.01, q)` or `circuit.pauli_channel([q0, q1], [("XZ", 0.01)])`.

### States

With `heisenberg=False` the Pauli sum is a state in the Pauli basis, and the circuit acts on it
from the first gate to the last. The coefficient of a Pauli string in the result, times `2^n`,
is the expectation value of that string:

```python
state = pp.PauliSum(2)                      # |00><00| = (I + Z)(I + Z) / 4
for paulis in ("II", "IZ", "ZI", "ZZ"):
    state.add(paulis, [0, 1], 0.25)

evolved = pp.propagate(circuit, state, heisenberg=False)
print(evolved.coeff("ZZ", [0, 1]) * 4)      # <ZZ> after the circuit
```

An observable propagated backwards and a state propagated forwards meet in
`observable.overlap_with_pauli_sum(state)`, which is `Tr[state observable]`.

### Gates from matrices

A gate on up to five qubits can be given as a unitary matrix, and any linear map of Pauli
strings, such as a noise channel, as its transfer matrix:

```python
circuit.unitary([q0, q1], matrix)           # 4 x 4 complex, q0 the most significant bit
circuit.transfer_map([q], transfer_matrix)  # 4 x 4 real
```

Entry `(i, j)` of a transfer matrix is the factor of the Pauli string `i` in the image of the
Pauli string `j` of an observable, `Tr[P_i E†(P_j)] / 2^k`. A Pauli string on the `k` qubits of
the gate is numbered by its operators as base-4 digits, with I, X, Y, Z as 0 to 3 and the first
qubit as the lowest digit.

### Complex coefficients

`pp.PauliSum(num_qubits, dtype=complex)` holds complex coefficients, for observables that are
not Hermitian. `astype(float)` and `astype(complex)` convert between the two kinds.

## What is supported

- Gates: H, X, Y, Z, S, SX, T and their adjoints, CX, CY, CZ, SWAP, CCX (as Clifford + T), RX,
  RY, RZ, RXX, RYY, RZZ, `rotation(paulis, qubits, theta)` for `exp(-i theta/2 P)`, and unitary
  matrices and transfer matrices on up to five qubits.
- Noise: Pauli channels on up to six qubits, given by error probabilities; depolarizing,
  dephasing and Pauli X, Y, Z noise with the strengths of PauliPropagation.jl; amplitude damping.
  From a QDK `NoiseConfig`, the Pauli errors of every gate; qubit loss is rejected, and noise on
  the trailing measurements is dropped.
- Pauli sums with real or complex coefficients, on up to 512 qubits, propagated as observables
  (backwards) or as states (forwards).
- Truncation after every gate, with the semantics of PauliPropagation.jl:

  | Option | Drops a term when |
  | --- | --- |
  | `max_weight` | it has more non-identity operators than this |
  | `min_abs_coeff` (default `1e-10`) | its coefficient is smaller than this in absolute value |
  | `min_rel_coeff` | its coefficient is smaller than this fraction of the largest in the sum |
  | `max_freq` | more rotations than this split the Pauli string on its way |
  | `max_sins` | it took more sine branches of rotations than this |
  | `weight_damping=(gamma, threshold)` | `abs(coeff) * 10**(-gamma * weight) < threshold` |
  | `custom_truncation` | this function of the label and the coefficient returns true |

  A Clifford gate changes no coefficient but its sign, so after one only the options that read
  the Pauli string (`max_weight`, `weight_damping`, `custom_truncation`) drop anything.
- Expectation values in the all-zero state, the all-plus state, any computational basis
  state, the maximally mixed state, and a state given as a Pauli sum.
- QIR input: one unitary circuit, optionally followed by measurements. Branching on
  measurement results is rejected. OpenQASM works the same way once the QDK has compiled it
  to QIR.

Truncated results are approximations. Nothing here bounds their error; check convergence by
tightening the truncation.

### Qubit numbering in QIR

Observables refer to the qubits of the QIR program. These are the qubits of the Q# register in
allocation order as long as the program measures only at the end. For the base profile the QDK
moves a mid-circuit measurement to the end and continues on a fresh qubit, so after such a
measurement the QIR qubits no longer number the Q# register.

## Building and testing

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install maturin pytest numpy qdk
maturin develop --release
cargo test          # gate and noise rules against dense matrices
pytest              # against the QDK simulators, numpy and PauliPropagation.jl
```

`tests/reference` holds results, circuits and gate tables from PauliPropagation.jl. To
regenerate them, with a clone of that package in `~/dev/PauliPropagation.jl`:

```bash
julia --project=tools/julia -e 'using Pkg; Pkg.develop(path=joinpath(homedir(), "dev", "PauliPropagation.jl")); Pkg.add("JSON")'
julia --project=tools/julia tools/julia/generate_reference.jl
```

## Verification

Every rule is checked against a reference that shares no code with the engine:

1. **Dense matrices** (`crates/core/tests`). Every Clifford gate and every one of the 63 Pauli
   rotations on three qubits is checked on all 64 Pauli strings against `U† P U` computed by
   matrix multiplication, as are the Toffoli decomposition and random circuits. Noise channels
   and transfer maps are checked the same way against their Kraus operators, the propagation
   of states against `U rho U†`, complex coefficients against complex matrices, and overlaps
   against the matrix of the observable. Every truncation is checked against a propagation
   that applies the rules one term at a time.
2. **QDK simulators** (`tests/test_*qdk*.py`, `tests/test_truncation_options.py`). Random Q#
   circuits are compiled to QIR and propagated without truncation; the expectation values,
   real and complex, and the propagated states must match the QDK state vector for the same
   Q# source. This covers the QIR reader and the qubit order. With a `NoiseConfig`, the values
   must match the averages over a million shots of the QDK's noisy simulators.
3. **numpy** (`tests/test_matrix_gates_against_numpy.py`). Circuits of random unitaries and of
   random channels given by Kraus operators must match a state vector and a density matrix
   evolved by matrix products.
4. **PauliPropagation.jl** (`tests/test_against_julia.py`). Truncated Pauli sums must match the
   Julia package term by term on circuits it built: with every truncation, with and without
   noise, with complex coefficients, as states, with gates given as matrices, including its
   README example and a 100-qubit case.
5. **The tests of PauliPropagation.jl** (`tests/test_conformance_*.py`). The cases of its test
   files run through this package's API, with its Clifford tables and circuit builders as
   data. Where the Julia package compares its backends with each other, the engine is
   compared with a propagation on labels written in Python.

## Attribution

The algorithms, conventions and defaults follow
[PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl), which is licensed under
Apache-2.0, as is this package.
