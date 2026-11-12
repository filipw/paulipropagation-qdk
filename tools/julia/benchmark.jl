# Times PauliPropagation.jl on the benchmark circuits and writes them out for benchmarks/run.py.
#
#   julia -t 8 --project=tools/julia tools/julia/benchmark.jl [case ...]
#
# Every case is written to benchmarks/cases/<name>.json with the circuit, the observable and the
# truncation. The timings, the number of terms and the overlap with the zero state go to
# benchmarks/results/julia.json. Qubits are 0-based in the files.

using PauliPropagation
using PauliPropagation.Performance
using JSON
using Printf

include("circuitjson.jl")

const ROOT = joinpath(@__DIR__, "..", "..", "benchmarks")
const REPETITIONS = 3

function besttime(f::F) where {F}
    best = Inf
    for _ in 1:REPETITIONS
        GC.gc()
        best = min(best, @elapsed f())
    end
    return best
end

"The ways of running PauliPropagation.jl that are timed, slowest storage first."
function configurations(pstr, capacity)
    return [
        "dict" => (circuit, params; kw...) -> propagate(circuit, PauliSum(pstr), params; kw...),
        "vector, 1 thread" => (circuit, params; kw...) ->
            propagate(circuit, VectorPauliSum(pstr), params; thread=false, kw...),
        "vector" => (circuit, params; kw...) -> propagate(circuit, VectorPauliSum(pstr), params; kw...),
        "fused, 1 thread" => (circuit, params; kw...) ->
            Performance.propagate(circuit, VectorPauliSum(pstr), params; thread=false, kw...),
        "fused" => (circuit, params; kw...) ->
            Performance.propagate(circuit, VectorPauliSum(pstr), params; kw...),
        "fused, zones" => (circuit, params; kw...) ->
            Performance.propagate(circuit, MultiPauliSum(VectorPauliSum(pstr)), params; kw...),
        "fused, zones, sized" => (circuit, params; kw...) -> begin
            cache = PropagationCache(MultiPauliSum(VectorPauliSum(pstr)))
            resize!(cache, capacity)
            Performance.propagate!(circuit, cache, params; kw...)
        end,
    ]
end

"""
Times every configuration on `nlayers` repetitions of `layer` and writes the case.
`observable` is `(symbols, qubits)` with 1-based qubits.
"""
function benchmark(name, nq, layer, layer_params, nlayers, observable; max_weight=nothing, min_abs_coeff)
    circuit = repeat(layer, nlayers)
    params = repeat(layer_params, nlayers)
    symbols, qinds = observable
    pstr = PauliString(nq, collect(symbols), collect(qinds))
    kw = (; max_weight=isnothing(max_weight) ? Inf : max_weight, min_abs_coeff)

    case = Dict(
        "name" => name,
        "num_qubits" => nq,
        "circuit" => circuittojson(circuit, params),
        "observable" => [Dict("paulis" => join(string.(symbols)), "qubits" => collect(qinds) .- 1, "coeff" => 1.0)],
        "max_weight" => max_weight,
        "min_abs_coeff" => min_abs_coeff,
    )
    mkpath(joinpath(ROOT, "cases"))
    open(io -> JSON.print(io, case), joinpath(ROOT, "cases", name * ".json"), "w")

    reference = propagate(circuit, PauliSum(pstr), params; kw...)
    nterms = length(reference)
    @printf("%s: %d qubits, %d gates, %d terms, overlap %.15g\n", name, nq, length(circuit), nterms,
        overlapwithzero(reference))

    times = Dict{String,Float64}()
    results = Dict{String,Any}()
    for (label, run) in configurations(pstr, nterms + nterms ÷ 4)
        # one layer first, so that no timed run compiles anything
        run(layer, layer_params; kw...)
        result = nothing
        times[label] = besttime(() -> (result = run(circuit, params; kw...)))
        results[label] = Dict("terms" => length(result), "overlap_with_zero" => overlapwithzero(result))
        @printf("  %-22s %8.3f s   %8d terms   overlap %.15g\n", label, times[label], length(result),
            overlapwithzero(result))
    end

    return Dict(
        "gates" => length(circuit),
        "terms" => nterms,
        "overlap_with_zero" => overlapwithzero(reference),
        "seconds" => times,
        "results" => results,
    )
end

const CASES = Dict{String,Function}()

# The circuit of the performance notebook of PauliPropagation.jl: a tilted-field Ising model on a square lattice.
CASES["tilted_ising_6x6"] = function ()
    nx, ny, dt, h, J = 6, 6, 0.05, 1.0, 0.5
    nq = nx * ny
    topology = rectangletopology(nx, ny)
    layer = PauliRotation[]
    rxlayer!(layer, nq)
    rzlayer!(layer, nq)
    rzzlayer!(layer, topology)
    params = vcat(fill(2dt * h, 2nq), fill(2dt * J, length(topology)))
    return benchmark("tilted_ising_6x6", nq, layer, params, 14, ([:Z, :Z], [21, 22]); min_abs_coeff=2.0^(-20))
end

# A kicked Ising model on the 127 qubits of the heavy-hex lattice of the IBM Eagle devices.
CASES["kicked_ising_eagle_127"] = function ()
    nq = 127
    layer = PauliRotation[]
    rzzlayer!(layer, ibmeagletopology)
    rxlayer!(layer, nq)
    params = vcat(fill(-pi / 2, length(ibmeagletopology)), fill(0.3, nq))
    return benchmark("kicked_ising_eagle_127", nq, layer, params, 20, ([:Z], [63]); max_weight=8, min_abs_coeff=1e-5)
end

# The README example of PauliPropagation.jl, where the sum stays small and the cost is per gate.
CASES["readme_tfi_32"] = function ()
    nq = 32
    layer = tfitrottercircuit(nq, 1; topology=bricklayertopology(nq; periodic=true))
    params = fill(0.1, countparameters(layer))
    return benchmark("readme_tfi_32", nq, layer, params, 32, ([:Z], [16]); max_weight=6, min_abs_coeff=1e-4)
end

# A square-lattice Ising model with depolarizing noise on every qubit after every layer.
CASES["noisy_ising_6x6"] = function ()
    nx, ny = 6, 6
    nq = nx * ny
    topology = rectangletopology(nx, ny)
    layer = Gate[]
    rzzlayer!(layer, topology)
    rxlayer!(layer, nq)
    append!(layer, (DepolarizingNoise(q) for q in 1:nq))
    params = vcat(fill(0.1, length(topology)), fill(0.2, nq), fill(0.01, nq))
    return benchmark("noisy_ising_6x6", nq, layer, params, 12, ([:Z, :Z], [21, 22]); min_abs_coeff=1e-6)
end

function main()
    names = isempty(ARGS) ? sort(collect(keys(CASES))) : ARGS
    path = joinpath(ROOT, "results", "julia.json")
    report = isfile(path) ? JSON.parsefile(path) : Dict{String,Any}()
    report["version"] = string(pkgversion(PauliPropagation))
    report["threads"] = Threads.nthreads()
    cases = get!(report, "cases", Dict{String,Any}())
    for name in names
        cases[name] = CASES[name]()
    end
    mkpath(dirname(path))
    open(io -> JSON.print(io, report, 2), path, "w")
end

main()
