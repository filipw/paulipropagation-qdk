# Generates reference results with PauliPropagation.jl for the tests in tests/test_against_julia.py.
#
#   julia --project=tools/julia tools/julia/generate_reference.jl
#
# Every case is written to tests/reference/<name>.json with the circuit, the observable, the
# truncation, and the propagated Pauli sum. Qubits are 0-based in the files.

using PauliPropagation
using JSON
using Random

const OUT_DIR = joinpath(@__DIR__, "..", "..", "tests", "reference")

label(term, nq) = join(("IXYZ"[getpauli(term, q)+1] for q in 1:nq))

function gatetojson(gate::PauliRotation, theta)
    return Dict("gate" => "rotation", "paulis" => join(string.(gate.symbols)),
        "qubits" => gate.qinds .- 1, "theta" => theta)
end
gatetojson(gate::CliffordGate) = Dict("gate" => string(gate.symbol), "qubits" => gate.qinds .- 1)

function circuittojson(circuit, params)
    remaining = Iterators.Stateful(params)
    return [gate isa PauliRotation ? gatetojson(gate, popfirst!(remaining)) : gatetojson(gate) for gate in circuit]
end

function termstojson(psum, nq)
    pairs = [(label(pstr.term, nq), pstr.coeff) for pstr in topaulistrings(psum)]
    return [[l, c] for (l, c) in sort(pairs)]
end

"""
Propagates `observable`, a vector of `(symbols, qubits, coefficient)` with 1-based qubits, and writes the case.
`max_weight = nothing` stands for no weight truncation.
"""
function writecase(name, nq, circuit, params, observable; max_weight=nothing, min_abs_coeff)
    psum = PauliSum(nq)
    for (symbols, qinds, coeff) in observable
        add!(psum, collect(symbols), collect(qinds), coeff)
    end
    result = propagate(circuit, psum, params;
        max_weight=isnothing(max_weight) ? Inf : max_weight, min_abs_coeff)

    case = Dict(
        "name" => name,
        "reference" => "PauliPropagation.jl $(pkgversion(PauliPropagation))",
        "num_qubits" => nq,
        "circuit" => circuittojson(circuit, params),
        "observable" => [Dict("paulis" => join(string.(symbols)), "qubits" => collect(qinds) .- 1, "coeff" => coeff)
                         for (symbols, qinds, coeff) in observable],
        "max_weight" => max_weight,
        "min_abs_coeff" => min_abs_coeff,
        "overlap_with_zero" => overlapwithzero(result),
        "terms" => termstojson(result, nq),
    )
    mkpath(OUT_DIR)
    open(joinpath(OUT_DIR, name * ".json"), "w") do io
        JSON.print(io, case)
    end
    println(rpad(name, 34), " gates=", rpad(length(circuit), 6), " terms=", rpad(length(result), 8),
        " overlap=", overlapwithzero(result))
end

"A random circuit of Clifford gates and Pauli rotations on up to three qubits."
function randomcircuit(rng, nq, ngates)
    circuit = Gate[]
    params = Float64[]
    one_qubit = [:H, :X, :Y, :Z, :S, :SX, :SY]
    two_qubit = [:CNOT, :CZ, :SWAP, :ZZpihalf]
    for _ in 1:ngates
        kind = rand(rng, 1:4)
        if kind == 1
            push!(circuit, CliffordGate(rand(rng, one_qubit), [rand(rng, 1:nq)]))
        elseif kind == 2
            push!(circuit, CliffordGate(rand(rng, two_qubit), shuffle(rng, 1:nq)[1:2]))
        else
            k = rand(rng, 1:3)
            push!(circuit, PauliRotation(rand(rng, [:X, :Y, :Z], k), shuffle(rng, 1:nq)[1:k]))
            push!(params, 2pi * rand(rng) - pi)
        end
    end
    return circuit, params
end

# The example of the PauliPropagation.jl README, which states 0.154596728241.
let nq = 32
    circuit = tfitrottercircuit(nq, 32; topology=bricklayertopology(nq; periodic=true))
    params = fill(0.1, countparameters(circuit))
    writecase("readme_tfi_32", nq, circuit, params, [([:Z], [16], 1.0)]; max_weight=6, min_abs_coeff=1e-4)
end

# Weight truncation alone does not depend on rounding, so the terms have to agree one by one.
let nq = 16
    circuit = tfitrottercircuit(nq, 8)
    params = fill(0.3, countparameters(circuit))
    writecase("tfi_weight_only_16", nq, circuit, params, [([:Z], [8], 1.0)]; max_weight=4, min_abs_coeff=0.0)
end

let nq = 12
    circuit = heisenbergtrottercircuit(nq, 5)
    params = fill(0.2, countparameters(circuit))
    observable = [([:X], [6], 1.0), ([:Z, :Z], [3, 4], 0.7)]
    writecase("heisenberg_trotter_12", nq, circuit, params, observable; max_weight=5, min_abs_coeff=1e-4)
end

# Nothing truncated: every gate rule and convention has to agree.
let nq = 7, rng = MersenneTwister(1)
    circuit, params = randomcircuit(rng, nq, 80)
    observable = [([:Z], [1], 1.0), ([:X, :Y], [3, 6], -0.5), ([:Y, :Z, :X], [2, 4, 7], 0.25)]
    writecase("mixed_gates_exact_7", nq, circuit, params, observable; min_abs_coeff=0.0)
end

let nq = 20, rng = MersenneTwister(2)
    circuit, params = randomcircuit(rng, nq, 300)
    # Small angles, so that the sum stays sparse enough to truncate by coefficient.
    params .*= 0.25
    observable = [([:Z], [10], 1.0), ([:Z, :Z], [3, 4], 0.5)]
    writecase("mixed_gates_truncated_20", nq, circuit, params, observable; max_weight=6, min_abs_coeff=1e-6)
end

# More than 64 qubits, with the observable next to the boundary between two 64-bit words.
let nq = 100
    circuit = tfitrottercircuit(nq, 12; topology=bricklayertopology(nq; periodic=true))
    params = fill(0.25, countparameters(circuit))
    writecase("tfi_wide_100", nq, circuit, params, [([:Z], [64], 1.0)]; max_weight=6, min_abs_coeff=1e-5)
end
