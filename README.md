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

## What is supported

- Gates: H, X, Y, Z, S, SX, T and their adjoints, CX, CY, CZ, SWAP, CCX (as Clifford + T), RX,
  RY, RZ, RXX, RYY, RZZ, and `rotation(paulis, qubits, theta)` for `exp(-i theta/2 P)`.
- Noise: Pauli channels on up to six qubits, given by error probabilities; depolarizing,
  dephasing and Pauli X, Y, Z noise with the strengths of PauliPropagation.jl; amplitude damping.
  From a QDK `NoiseConfig`, the Pauli errors of every gate; qubit loss is rejected, and noise on
  the trailing measurements is dropped.
- Observables: sums of Pauli strings with real coefficients, on up to 512 qubits.
- Truncation after every gate by Pauli weight (`max_weight`) and by coefficient magnitude
  (`min_abs_coeff`, default `1e-10`), with the semantics of PauliPropagation.jl.
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
pytest              # against the QDK simulators and against PauliPropagation.jl
```

`tests/reference/*.json` are results from PauliPropagation.jl. To regenerate them, with a
clone of that package in `~/dev/PauliPropagation.jl`:

```bash
julia --project=tools/julia -e 'using Pkg; Pkg.develop(path=joinpath(homedir(), "dev", "PauliPropagation.jl")); Pkg.add("JSON")'
julia --project=tools/julia tools/julia/generate_reference.jl
```

## Verification

Every rule is checked against a reference that shares no code with the engine:

1. **Dense matrices** (`crates/core/tests`). Every Clifford gate and every one of the 63 Pauli
   rotations on three qubits is checked on all 64 Pauli strings against `U† P U` computed by
   matrix multiplication, as are the Toffoli decomposition and random circuits. Noise channels
   are checked the same way against their Kraus operators, and overlaps against the matrix of
   the observable.
2. **QDK simulators** (`tests/test_against_qdk.py`, `tests/test_noise_against_qdk.py`). Random
   Q# circuits are compiled to QIR and propagated without truncation; the expectation values
   must match those of the QDK state vector for the same Q# source. This covers the QIR reader
   and the qubit order. With a `NoiseConfig`, the values must match the averages over a million
   shots of the QDK's noisy simulators.
3. **PauliPropagation.jl** (`tests/test_against_julia.py`). Truncated Pauli sums must match the
   Julia package term by term on circuits it built, with and without noise, including its
   README example and a 100-qubit case.

## Attribution

The algorithms, conventions and defaults follow
[PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl), which is licensed under
Apache-2.0, as is this package.
