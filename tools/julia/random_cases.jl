# Propagates random circuits with PauliPropagation.jl, for tools/compare_random_cases.py.
#
#   julia --project=tools/julia tools/julia/random_cases.jl <first seed> <cases> <cases per file> <directory> [growth]
#
# Every case is made from its seed alone: a register of 1 to 130 qubits, a circuit of every
# kind of gate, a sum of Pauli strings with real or complex coefficients, a picture, and any
# combination of truncations. A case is one line of JSON with the circuit, the propagated sum
# and its overlaps, and for some cases the sum with its translations merged and the gradient.
# The lines go to files of <cases per file> cases in <directory>, each named after its first
# seed and given the ending .jsonl once it is complete. Qubits are 0-based in the files.
#
# `growth` bounds the size of a case: on more than 5 qubits, the gates of a circuit can make
# at most 2^growth terms out of one. It is 13 unless given.

using PauliPropagation
using JSON
using LinearAlgebra
using Random

include("circuitjson.jl")

const ONE_QUBIT = [:H, :X, :Y, :Z, :S, :SX, :SY]
const TWO_QUBIT = [:CNOT, :CZ, :SWAP, :ZZpihalf]
const PAULI_NOISE = [DepolarizingNoise, PauliXNoise, PauliYNoise, PauliZNoise]
const SPECIAL_ANGLES = [0.0, pi / 2, -pi / 2, pi, pi / 4, -pi / 4, 2pi, 1e-9]

termstojson(psum, nq) = [[l, numbertojson(c)] for (l, c) in sort([(label(pstr.term, nq), pstr.coeff) for pstr in topaulistrings(psum)]; by=first)]

subset(rng, nq, k) = shuffle(rng, 1:nq)[1:k]

function randomangle(rng)
    kind = rand(rng)
    kind < 0.7 && return 2pi * rand(rng) - pi
    kind < 0.85 && return 0.6 * rand(rng) - 0.3
    return rand(rng, SPECIAL_ANGLES)
end

function randomstrength(rng)
    kind = rand(rng)
    kind < 0.8 && return 0.5 * rand(rng)
    kind < 0.9 && return rand(rng)
    return rand(rng, [0.0, 0.5, 1.0])
end

"A transfer matrix on `k` qubits that is no unitary gate: about half of its entries are zero, and no column adds up to more than 1."
function randomtransfermatrix(rng, k)
    matrix = [rand(rng) < 0.5 ? 2 * rand(rng) - 1 : 0.0 for _ in 1:4^k, _ in 1:4^k]
    largest = maximum(sum(abs, matrix; dims=1))
    return largest > 1 ? matrix ./ largest : matrix
end

randomunitary(rng, k) = Matrix(qr(randn(rng, ComplexF64, 2^k, 2^k)).Q)

"""
A random circuit and its parameters. `heisenberg` allows the gates that only act on observables, `rewindable` leaves out the gates
that `rewindgradient` does not take, and `growth` is the base-2 logarithm of the number of terms
that the gates may make out of one.
"""
function randomcircuit(rng, nq, ngates; heisenberg, rewindable, growth)
    circuit = Gate[]
    params = Float64[]
    for _ in 1:ngates
        kind = rand(rng, 1:12)
        if kind <= 2 || (kind == 3 && nq == 1)
            push!(circuit, CliffordGate(rand(rng, ONE_QUBIT), [rand(rng, 1:nq)]))
        elseif kind == 3 || kind == 4 && nq > 1
            push!(circuit, CliffordGate(rand(rng, TWO_QUBIT), subset(rng, nq, 2)))
        elseif kind <= 8
            growth < 1 && continue
            growth -= 1
            k = rand(rng, 1:min(nq, 4))
            gate = PauliRotation(rand(rng, [:X, :Y, :Z], k), subset(rng, nq, k))
            if rand(rng) < 0.15
                push!(circuit, freeze(gate, randomangle(rng)))
            else
                push!(circuit, gate)
                push!(params, randomangle(rng))
            end
        elseif kind == 9
            (heisenberg && !rewindable && growth >= 1) || continue
            growth -= 1
            push!(circuit, TGate(rand(rng, 1:nq)))
        elseif kind == 10
            rewindable && continue
            gate = rand(rng, PAULI_NOISE)(rand(rng, 1:nq))
            if rand(rng) < 0.15
                push!(circuit, freeze(gate, randomstrength(rng)))
            else
                push!(circuit, gate)
                push!(params, randomstrength(rng))
            end
        elseif kind == 11
            (heisenberg && !rewindable && growth >= 1) || continue
            growth -= 1
            push!(circuit, AmplitudeDampingNoise(rand(rng, 1:nq)))
            push!(params, randomstrength(rng))
        else
            (heisenberg && !rewindable) || continue
            k = rand(rng) < 0.7 ? 1 : min(nq, rand(rng) < 0.85 ? 2 : 3)
            growth >= 2k || continue
            growth -= 2k
            qinds = subset(rng, nq, k)
            if rand(rng) < 0.6
                push!(circuit, unitarygate(randomunitary(rng, k), qinds))
            else
                push!(circuit, TransferMapGate(randomtransfermatrix(rng, min(k, 2)), qinds[1:min(k, 2)]))
            end
        end
    end
    return circuit, params
end

"Random terms `(symbols, qubits, coefficient)` of a sum. Two of them can have the same Pauli string."
function randomterms(rng, nq, heisenberg, complex)
    terms = []
    for _ in 1:rand(rng, 1:4)
        k = rand(rng, 1:min(nq, 4))
        symbols = rand(rng, heisenberg ? [:X, :Y, :Z] : [:I, :X, :Y, :Z], k)
        coeff = rand(rng, [-1, 1]) * 10.0^(2.5 * rand(rng) - 2)
        complex && (coeff *= cis(2pi * rand(rng)))
        push!(terms, (symbols, sort(subset(rng, nq, k)), coeff))
    end
    return terms
end

function randomcase(seed, growth)
    rng = Xoshiro(seed)
    size = rand(rng)
    nq = size < 0.6 ? rand(rng, 1:8) : size < 0.85 ? rand(rng, 9:32) : rand(rng, [33, 50, 64, 65, 100, 130])
    heisenberg = rand(rng) < 0.7
    complex = rand(rng) < 0.2

    max_weight = rand(rng) < 0.35 ? rand(rng, 1:min(nq, 8)) : nothing
    min_abs_coeff = rand(rng) < 0.4 ? 0.0 : 10.0^(-1 - 6 * rand(rng))
    min_rel_coeff = rand(rng) < 0.15 ? 10.0^(-0.5 - 5.5 * rand(rng)) : nothing
    weight_damping = rand(rng) < 0.15 ? (rand(rng), 10.0^(-1 - 5 * rand(rng))) : nothing
    max_freq = rand(rng) < 0.15 ? rand(rng, 1:20) : nothing
    max_sins = rand(rng) < 0.15 ? rand(rng, 1:8) : nothing
    counted = !isnothing(max_freq) || !isnothing(max_sins)

    # A gradient needs a real observable, a circuit of Clifford gates and rotations, and neither
    # max_freq nor max_sins.
    differentiated = heisenberg && !complex && !counted && rand(rng) < 0.3
    circuit, params = randomcircuit(rng, nq, rand(rng, 1:80); heisenberg, rewindable=differentiated,
        growth=nq > 5 ? growth : typemax(Int))
    terms = randomterms(rng, nq, heisenberg, complex)
    ones = [q for q in 1:nq if rand(rng) < 0.3]

    psum = PauliSum(complex ? ComplexF64 : Float64, nq)
    for (symbols, qinds, coeff) in terms
        add!(psum, symbols, qinds, coeff)
    end

    # Every truncation is always passed, with its default where the case does not have it.
    limit(value) = isnothing(value) ? Inf : Float64(value)
    customtruncfunc = nothing
    if !isnothing(weight_damping)
        gamma, threshold = weight_damping
        customtruncfunc = (pstr, coeff) -> truncatedampingcoeff(pstr, coeff, gamma, threshold)
    end
    kwargs = (; min_abs_coeff, max_weight=limit(max_weight), max_freq=limit(max_freq), max_sins=limit(max_sins),
        min_rel_coeff, customtruncfunc, heisenberg)
    result = propagate(circuit, psum, params; kwargs...)

    case = Dict{String,Any}(
        "seed" => seed,
        "num_qubits" => nq,
        "circuit" => circuittojson(circuit, params),
        "observable" => [Dict("paulis" => join(string.(symbols)), "qubits" => qinds .- 1, "coeff" => numbertojson(coeff))
                         for (symbols, qinds, coeff) in terms],
        "max_weight" => max_weight,
        "min_abs_coeff" => min_abs_coeff,
        "max_freq" => max_freq,
        "max_sins" => max_sins,
        "min_rel_coeff" => min_rel_coeff,
        "weight_damping" => isnothing(weight_damping) ? nothing : collect(weight_damping),
        "heisenberg" => heisenberg,
        "terms" => termstojson(result, nq),
        "ones" => ones .- 1,
        "overlap_with_zero" => numbertojson(overlapwithzero(result)),
        "overlap_with_plus" => numbertojson(overlapwithplus(result)),
        "overlap_with_computational" => numbertojson(overlapwithcomputational(result, ones)),
        "overlap_with_max_mixed" => numbertojson(overlapwithmaxmixed(result)),
    )

    if nq >= 2 && rand(rng) < 0.2
        divisors = [nx for nx in 2:nq-1 if nq % nx == 0]
        grid = isempty(divisors) || rand(rng) < 0.5 ? nothing : (nx = rand(rng, divisors); (nx, nq ÷ nx))
        merged = isnothing(grid) ? translationmerge(result) : translationmerge(result, grid...)
        case["grid"] = isnothing(grid) ? nothing : collect(grid)
        case["merged"] = termstojson(merged, nq)
    end

    if differentiated && any(gate -> gate isa PauliRotation, circuit)
        state = rand(rng, ["zero", "plus"])
        overlapfunc = state == "zero" ? overlapwithzero : overlapwithplus
        # On a `PauliSum`, as the propagation above: a `VectorPauliSum` keeps the terms of the
        # starting sum that a truncation would drop for as long as the gates leave them unchanged.
        value, gradient = rewindgradient(circuit, psum, params, overlapfunc; min_abs_coeff, max_weight=limit(max_weight))
        case["gradient"] = Dict("state" => state, "value" => value, "derivatives" => gradient)
    end
    return case
end

function main(firstseed, count, perfile, directory, growth)
    mkpath(directory)
    for start in firstseed:perfile:firstseed+count-1
        path = joinpath(directory, "cases_$(start)")
        open(path * ".part", "w") do io
            for seed in start:min(start + perfile, firstseed + count)-1
                case = try
                    randomcase(seed, growth)
                catch err
                    Dict{String,Any}("seed" => seed, "error" => first(sprint(showerror, err), 300))
                end
                println(io, JSON.json(case))
                empty!(UNITARIES)
            end
        end
        mv(path * ".part", path * ".jsonl"; force=true)
    end
end

main(parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4], length(ARGS) > 4 ? parse(Int, ARGS[5]) : 13)
