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

### Arithmetic

Pauli sums on the same qubits are added, subtracted and multiplied by numbers with the usual
operators. `a.product(b)` is the operator product and `a.commutator(b)` is `ab - ba`, both
with complex coefficients:

```python
hamiltonian = 0.5 * zz + 0.3 * x
derivative = 1j * hamiltonian.commutator(observable)    # dO/dt in the Heisenberg picture
```

### Symmetries

Where the circuit and the state do not change under a translation of the qubits, Pauli strings
that are translations of each other have the same expectation value, and they can be merged
into one term. `translation_merge()` does this for a ring of qubits and
`translation_merge(nx, ny)` for a torus of `nx` columns and `ny` rows, on which the qubit in
column `x` of row `y` is `y * nx + x`. The merged sum has the expectation value of the sum
it was made from, with far fewer terms; merging between the layers of a circuit keeps the
sum small all the way:

```python
for _ in range(layers):
    observable = pp.propagate(layer, observable, max_weight=6).translation_merge()
```

`symmetry_merge(representative)` merges by any other symmetry: `representative` takes the
label of a Pauli string and returns the label that stands for all its images.

### Gradients

`pp.gradient` returns an expectation value and its derivatives with respect to the angles of
all rotations. It takes one pass back through the circuit and one pass forward, where the
parameter-shift rule takes two propagations for every angle:

```python
value, derivatives = pp.gradient(circuit, observable, max_weight=6, min_abs_coeff=1e-6)
```

`derivatives[k]` belongs to the `k`-th rotation of the circuit, the one with the angle
`circuit.angles[k]`; a T gate is a rotation. `circuit.with_angles(angles)` is the circuit
with other angles. Without truncation the gradient is exact, and with truncation it is
close to the gradient of the truncated expectation value. `state` selects the state, as for
overlaps: `'zero'`, `'plus'`, `'mixed'`, the list of the qubits that are 1, or a Pauli sum.

### Surrogates

Which Pauli strings a propagation reaches does not depend on the angles of the rotations,
and neither does a truncation by `max_weight`, `max_freq` or `max_sins`. A `Surrogate`
propagates once for all angles and records how every coefficient is made of sines and
cosines. Evaluating it at given angles returns the same sum as `propagate` with those angles
and `min_abs_coeff=0`, bit for bit, without propagating again:

```python
surrogate = pp.Surrogate(circuit, observable, max_weight=5, max_freq=30)
for angles in candidates:
    value = surrogate.evaluate(angles).overlap_with_zero()
```

### Sampling

A sum that grows beyond what truncation can hold can be thinned at random instead.
`pp.resample(sum, target_size)` drops terms at random and makes the others larger so that the
sum is unchanged on average, and `pp.mcpropagate` propagates and resamples whenever a gate
leaves more than `max_size` terms. The sum then never holds more than twice `max_size` terms,
and the result is the propagated sum on average:

```python
estimate = pp.mcpropagate(circuit, observable, max_size=100_000, seed=1)
```

The resampling methods are those of PauliPropagation.jl: `'semideterministic'` (the default,
which keeps the largest terms as they are), `'systematic'` and `'multinomial'`.
`pp.mcsample(circuit, sum, samples)` follows single random paths instead, one branch at every
rotation, and returns the average of `samples` of them per term. A `seed` makes any of these
reproducible.

### The error of a truncation

`pp.estimate_mse` estimates the mean squared error that `max_weight`, `max_freq`, `max_sins`
or a rule on the Pauli string make in an expectation value, from random paths and without
propagating:

```python
mse = pp.estimate_mse(circuit, observable, 100_000, max_weight=6)            # any angles
mse = pp.estimate_mse(circuit, observable, 100_000, thetas=0.3, max_weight=6)  # angles up to 0.3
```

The average is over the angles of the rotations, each drawn uniformly from `[-thetas, thetas]`
whatever angle the circuit gives it, so it tells how a truncation does on circuits of this
shape rather than on this circuit. The observable is a single Pauli string.

### Threads

A propagation uses the threads of the machine as soon as the sum is large enough to be worth
dividing among them. `threads` sets their number:

```python
evolved = pp.propagate(circuit, observable, max_weight=6, threads=8)
```

The result does not depend on it: every coefficient comes out bit for bit the same on any
number of threads. On a processor with performance and efficiency cores, the number of
performance cores is usually the fastest choice, because the threads work through each gate
together and wait for the slowest of them.

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
- Sums, products and commutators of Pauli sums, and merging of the Pauli strings that a
  translation on a ring or a torus, or any symmetry given as a function, maps to each other.
- Gradients with respect to the angles of all rotations in one backward and one forward pass,
  for circuits without noise.
- Surrogates that are evaluated for new angles without propagating again, for circuits of
  Clifford gates and rotations.
- Resampling of a sum, propagation with resampling, sampling of single paths, and an estimate
  of the mean squared error of a truncation, with a seed for reproducible results.
- QIR input: one unitary circuit, optionally followed by measurements. Branching on
  measurement results is rejected. OpenQASM works the same way once the QDK has compiled it
  to QIR.

Truncated results are approximations. Nothing here bounds their error on a given circuit:
`estimate_mse` gives its average over the angles of the rotations, and convergence is checked
by tightening the truncation.

## How it works

The terms of the sum are kept in arrays, in the order of their Pauli strings. A rotation
changes the coefficients of the terms it acts on in place and creates one more term from
each, with some bits of the Pauli string flipped, the same bits for all of them. The created
terms are put in order by regrouping them by those bits, which is cheaper than sorting them,
and one merge of the two ordered sequences adds up the terms that have the same Pauli string
and drops the truncated ones. This is the vector backend of PauliPropagation.jl. A gate that
cannot change any term of the sum, because no term has an operator where it acts, is skipped.

For several threads the sum is divided into zones, by whether a Pauli string commutes with
each of a few fixed Pauli strings. Flipping the same bits in every string of a zone moves all
of them to one other zone, so under a rotation the zones pair up and every pair is worked on
by one thread without looking at the others. Every coefficient is computed by the same
operations in the same order however the sum is divided.

In Rust, `pauliprop::propagate_in_zones` takes the size and the largest number of the zones,
and `pauliprop::reference` is a second, simple propagation on a hash map.

A gradient is computed as in PauliPropagation.jl: the observable is propagated backwards
through the circuit, and then the circuit is undone gate by gate, on the propagated
observable and on the values that its Pauli strings have in the state. At every rotation the
two give the derivative with respect to its angle. Random numbers come from xoshiro256**,
seeded through splitmix64, so that a seed gives the same result on every platform.

### Qubit numbering in QIR

Observables refer to the qubits of the QIR program. These are the qubits of the Q# register in
allocation order as long as the program measures only at the end. For the base profile the QDK
moves a mid-circuit measurement to the end and continues on a fresh qubit, so after such a
measurement the QIR qubits no longer number the Q# register.

## Sample

`samples/pauli_propagation.ipynb` goes from a Q# program to expectation values on 100 qubits,
with truncation, noise, gradients, an error estimate and a surrogate.

## Building and testing

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install maturin pytest numpy qdk
maturin develop --release
cargo test          # against dense matrices, and the engine against the hash-map propagation
pytest              # against the QDK simulators, numpy and PauliPropagation.jl
```

Both suites cap their own memory, at a few times what they need: a propagation that fails to
truncate grows without end, and a test run that does is ended rather than left to exhaust the
machine (`crates/core/tests/common/memory.rs`, `tests/conftest.py`).

`tests/reference` holds results, circuits and gate tables from PauliPropagation.jl. To
regenerate them, with a clone of that package next to this repository:

```bash
julia --project=tools/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=tools/julia tools/julia/generate_reference.jl
julia --project=tools/julia tools/julia/generate_features.jl
```

## Benchmarks

`tools/julia/benchmark.jl` times PauliPropagation.jl on four circuits and writes them to
`benchmarks/cases`; `benchmarks/run.py` times this package on the same files and checks that
the number of terms and the expectation value agree:

```bash
julia -t 8 --project=tools/julia tools/julia/benchmark.jl
python benchmarks/run.py --threads 1,8
```

Best of three runs on an Apple M4 Pro, which has eight performance cores, against the fastest
configuration of PauliPropagation.jl 0.9.2 for the number of threads:

| Case | Terms | 1 thread | Julia, 1 thread | 8 threads | Julia, 8 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transverse-field Ising chain, 32 qubits, 32 layers | 1,671 | 1.5 ms | 1.6 ms | 1.5 ms | 1.9 ms |
| Kicked Ising on the 127-qubit heavy-hex lattice, 20 layers | 6,149 | 49 ms | 57 ms | 19 ms | 20 ms |
| Tilted-field Ising on 6 x 6 qubits, 14 layers | 1,036,142 | 0.66 s | 0.65 s | 0.11 s | 0.12 s |
| The same lattice with depolarizing noise, 12 layers | 36,935,368 | 17.5 s | 17.6 s | 3.4 s | 3.6 s |

## Verification

Every rule is checked against references that share no code with the engine, and the engine
as a whole against a second implementation of the same rules:

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
6. **A second engine** (`crates/core/tests/engines_agree.rs`, `tests/test_threads.py`).
   `pauliprop::reference` propagates on a hash map and shares only the truncation rules with
   the engine. The two must agree on every term and on every bit of every coefficient: on
   random circuits of all gates, under every truncation, as observables and as states, on
   registers of several machine words, on wide registers that the sum does not fill, however
   the engine divides the sum and on any number of threads.

7. **Free fermions** (`tests/test_free_fermions.py`). Circuits of RXX on neighbouring qubits
   and RZ keep sums of products of two Majorana operators in that form, so they are evolved
   exactly on any number of qubits by a matrix of twice that size. On 100 qubits every
   coefficient of the propagated sum must match this evolution, which is itself checked
   against dense matrices on four qubits.
8. **Arithmetic, symmetries, gradients and surrogates** (`crates/core/tests/algebra.rs`,
   `gradient.rs`, `surrogate.rs`, `tests/test_algebra.py`, `test_gradient.py`,
   `test_surrogate.py`, `test_features_against_julia.py`). Products and commutators are checked
   against matrix products, merged translations against translations written out by hand and
   against PauliPropagation.jl, gradients against the parameter-shift rule on dense matrices
   for every state and against the gradients of PauliPropagation.jl with and without
   truncation, and surrogates against the propagation, bit for bit, and against the
   surrogates of PauliPropagation.jl.
9. **Random methods** (`crates/core/tests/montecarlo.rs`, `tests/test_montecarlo.py`,
   `test_features_against_julia.py`). With a seed, resampling and sampled paths must equal the
   same methods written out in Python on the same random numbers. Without one, their averages
   must match the propagation within six standard errors. Error estimates must match an
   enumeration of all paths, the measured error of truncated propagations, and the estimates
   of PauliPropagation.jl from a million samples.

10. **A third implementation** (`tests/test_against_qiskit_pauli_prop.py`). Where the
    [pauli-prop](https://github.com/Qiskit/pauli-prop) package of Qiskit is installed, random
    circuits of Clifford gates and rotations must give the same terms without truncation, as
    observables and as states.

## Attribution

The algorithms, conventions and defaults follow
[PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl), which is licensed under
Apache-2.0, as is this package.
