# Generates reference results with PauliPropagation.jl for the tests in tests/test_features_against_julia.py:
# merged translations, gradients, error estimates and surrogates.
#
#   julia --project=tools/julia tools/julia/generate_features.jl
#
# Every case is written to tests/reference/features/<name>.json. Qubits are 0-based in the files.

using PauliPropagation
using JSON
using Random

const OUT_DIR = joinpath(@__DIR__, "..", "..", "tests", "reference", "features")

include("circuitjson.jl")

function termstojson(psum, nq)
    pairs = [(label(pstr.term, nq), pstr.coeff) for pstr in topaulistrings(psum)]
    return [[l, numbertojson(c)] for (l, c) in sort(pairs; by=first)]
end

function writejson(name, case)
    case["name"] = name
    case["reference"] = "PauliPropagation.jl $(pkgversion(PauliPropagation))"
    mkpath(OUT_DIR)
    open(joinpath(OUT_DIR, name * ".json"), "w") do io
        JSON.print(io, case)
    end
end

observabletojson(observable) = [Dict("paulis" => join(string.(symbols)), "qubits" => collect(qinds) .- 1, "coeff" => coeff)
                                for (symbols, qinds, coeff) in observable]

function sumof(nq, observable)
    psum = PauliSum(nq)
    for (symbols, qinds, coeff) in observable
        add!(psum, collect(symbols), collect(qinds), coeff)
    end
    return psum
end

"A random circuit of Clifford gates and Pauli rotations on up to three qubits, every rotation with an angle of its own."
function randomcircuit(rng, nq, ngates)
    circuit = Gate[]
    params = Float64[]
    for _ in 1:ngates
        kind = rand(rng, 1:4)
        if kind == 1
            push!(circuit, CliffordGate(rand(rng, [:H, :X, :Y, :Z, :S, :SX]), [rand(rng, 1:nq)]))
        elseif kind == 2
            push!(circuit, CliffordGate(rand(rng, [:CNOT, :CZ, :SWAP]), shuffle(rng, 1:nq)[1:2]))
        else
            k = rand(rng, 1:3)
            push!(circuit, PauliRotation(rand(rng, [:X, :Y, :Z], k), shuffle(rng, 1:nq)[1:k]))
            push!(params, 2pi * rand(rng) - pi)
        end
    end
    return circuit, params
end


## Translations

function writemerge(name, nq, psum; grid=nothing)
    merged = isnothing(grid) ? translationmerge(psum) : translationmerge(psum, grid...)
    writejson(name, Dict{String,Any}("num_qubits" => nq, "grid" => isnothing(grid) ? nothing : collect(grid),
        "terms" => termstojson(psum, nq), "merged" => termstojson(merged, nq)))
    println(rpad(name, 34), " terms=", rpad(length(psum), 8), " merged=", length(merged))
end

# The magnetization of a ring through a circuit that is the same on every bond and site.
let nq = 12
    circuit = tfitrottercircuit(nq, 4; topology=bricklayertopology(nq; periodic=true))
    psum = sumof(nq, [([:Z], [q], 1.0) for q in 1:nq])
    result = propagate(circuit, psum, fill(0.3, countparameters(circuit)); max_weight=4, min_abs_coeff=1e-6)
    writemerge("translation_merge_ring_12", nq, result)
end

# A sum without any symmetry.
let nq = 9, rng = MersenneTwister(21)
    circuit, params = randomcircuit(rng, nq, 60)
    result = propagate(circuit, sumof(nq, [([:Z], [1], 1.0), ([:X, :Y], [3, 6], -0.5)]), params; max_weight=5, min_abs_coeff=1e-5)
    writemerge("translation_merge_random_9", nq, result)
    writemerge("translation_merge_random_3x3", nq, result; grid=(3, 3))
end

let nx = 3, ny = 4, nq = 12
    topology = rectangletopology(nx, ny; periodic=true)
    circuit = tiltedtfitrottercircuit(nq, 3; topology=topology)
    psum = sumof(nq, [([:Z], [q], 1.0) for q in 1:nq])
    result = propagate(circuit, psum, fill(0.25, countparameters(circuit)); max_weight=3, min_abs_coeff=1e-6)
    writemerge("translation_merge_torus_3x4", nq, result; grid=(nx, ny))
end


## Gradients

function writegradient(name, nq, circuit, params, observable; state="zero", max_weight=nothing, min_abs_coeff)
    overlapfunc = state == "zero" ? overlapwithzero : overlapwithplus
    kwargs = Dict{Symbol,Any}(:min_abs_coeff => min_abs_coeff)
    isnothing(max_weight) || (kwargs[:max_weight] = max_weight)
    value, gradient = rewindgradient(circuit, VectorPauliSum(sumof(nq, observable)), params, overlapfunc; kwargs...)
    writejson(name, Dict{String,Any}("num_qubits" => nq, "circuit" => circuittojson(circuit, params),
        "observable" => observabletojson(observable), "state" => state, "max_weight" => max_weight,
        "min_abs_coeff" => min_abs_coeff, "value" => value, "gradient" => gradient))
    println(rpad(name, 34), " parameters=", rpad(length(params), 6), " value=", value)
end

let nq = 7, rng = MersenneTwister(22)
    circuit, params = randomcircuit(rng, nq, 70)
    observable = [([:Z], [1], 1.0), ([:X, :Y], [3, 6], -0.5), ([:Z, :Z], [2, 4], 0.25)]
    writegradient("gradient_exact_7", nq, circuit, params, observable; min_abs_coeff=0.0)
    writegradient("gradient_exact_plus_7", nq, circuit, params, observable; state="plus", min_abs_coeff=0.0)
    writegradient("gradient_truncated_random_7", nq, circuit, params, observable; max_weight=3, min_abs_coeff=1e-3)
end

let nq = 16
    circuit = tfitrottercircuit(nq, 6)
    rng = MersenneTwister(23)
    params = 0.2 .+ 0.4 .* rand(rng, countparameters(circuit))
    writegradient("gradient_truncated_tfi_16", nq, circuit, params, [([:Z], [8], 1.0)]; max_weight=4, min_abs_coeff=1e-4)
    writegradient("gradient_coefficient_tfi_16", nq, circuit, params, [([:Z], [8], 1.0), ([:Z, :Z], [3, 4], 0.5)]; min_abs_coeff=1e-3)
end


## Error estimates

function writemse(name, nq, circuit, symbols, qinds, nsamples, thetas; kwargs...)
    Random.seed!(24)
    value = estimatemse(circuit, PauliString(nq, symbols, qinds), nsamples, thetas; kwargs...)
    limits = Dict(string(key) => value for (key, value) in kwargs)
    writejson(name, Dict{String,Any}("num_qubits" => nq, "circuit" => circuittojson(circuit, zeros(countparameters(circuit))),
        "observable" => observabletojson([(symbols, qinds, 1.0)]), "samples" => nsamples, "thetas" => thetas,
        "limits" => limits, "mse" => value))
    println(rpad(name, 34), " mse=", value)
end

let nq = 10
    circuit = tfitrottercircuit(nq, 5)
    writemse("mse_weight_10", nq, circuit, [:Z], [5], 1_000_000, pi; max_weight=2)
    writemse("mse_weight_small_angles_10", nq, circuit, [:Z], [5], 1_000_000, 0.6; max_weight=2)
    writemse("mse_freq_10", nq, circuit, [:Z], [5], 1_000_000, pi; max_freq=8)
    writemse("mse_sins_10", nq, circuit, [:Z], [5], 1_000_000, 0.9; max_sins=3)
    radii = collect(range(0.2, 1.5; length=countparameters(circuit)))
    writemse("mse_all_10", nq, circuit, [:Z, :Z], [4, 5], 1_000_000, radii; max_weight=3, max_freq=14, max_sins=6)
end

let nq = 8, rng = MersenneTwister(25)
    circuit, _ = randomcircuit(rng, nq, 60)
    writemse("mse_random_8", nq, circuit, [:Z, :X], [2, 5], 1_000_000, 1.2; max_weight=3)
end


## Surrogates

let nq = 8, rng = MersenneTwister(26)
    circuit, params = randomcircuit(rng, nq, 70)
    observable = [([:Z, :X], [2, 5], 0.8)]
    wrapped = wrapcoefficients(PauliString(nq, [:Z, :X], [2, 5], 0.8), NodePathProperties)
    for (name, limits) in [("surrogate_weight_8", (max_weight=3,)), ("surrogate_counters_8", (max_freq=10, max_sins=5)),
        ("surrogate_all_8", (max_weight=4, max_freq=14, max_sins=7))]
        surrogate = propagate(circuit, wrapped; limits...)
        evaluations = []
        for round in 1:3
            thetas = round == 1 ? params : 2pi .* rand(rng, length(params)) .- pi
            evaluate!(surrogate, thetas)
            terms = [(label(pstr, nq), tonumber(path)) for (pstr, path) in zip(paulis(surrogate), coefficients(surrogate))]
            push!(evaluations, Dict("thetas" => thetas, "terms" => [[l, c] for (l, c) in sort(terms; by=first)]))
        end
        writejson(name, Dict{String,Any}("num_qubits" => nq, "circuit" => circuittojson(circuit, params),
            "observable" => observabletojson(observable), "limits" => Dict(string(k) => v for (k, v) in pairs(limits)),
            "evaluations" => evaluations))
        println(rpad(name, 34), " terms=", length(surrogate))
    end
end
