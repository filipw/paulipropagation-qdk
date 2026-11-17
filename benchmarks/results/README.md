# Measured timings

Best of three runs, as quoted in [docs/comparison.md](../../docs/comparison.md).

| Files | Machine | Written by |
| --- | --- | --- |
| `julia.json` | Apple M4 Pro | `tools/julia/benchmark.jl` |
| `paulipropagation_qdk.json` | Apple M4 Pro | `benchmarks/run.py` |
| `qiskit_pauli_prop.json` | Apple M4 Pro | `benchmarks/qiskit_pauli_prop.py` |
| `standard_nc40ads_h100_v5/` | Azure `Standard_NC40ads_H100_v5`: AMD EPYC 9V84 with 40 cores, NVIDIA H100 NVL | `benchmarks/azureml/job.sh`: the Julia benchmark, `benchmarks/run.py` and `benchmarks/cupauliprop.py`, with what each printed in a `.log` |
