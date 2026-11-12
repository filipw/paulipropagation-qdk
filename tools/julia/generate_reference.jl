# Generates reference results with PauliPropagation.jl for the tests in tests/test_against_julia.py.
#
#   julia --project=tools/julia tools/julia/generate_reference.jl
#
# Every case is written to tests/reference/<name>.json with the circuit, the observable, the
# truncation, and the propagated Pauli sum. Qubits are 0-based in the files.

using PauliPropagation
using JSON
using LinearAlgebra
using Random

const OUT_DIR = joinpath(@__DIR__, "..", "..", "tests", "reference")

include("circuitjson.jl")

function termstojson(psum, nq)
    pairs = [(label(pstr.term, nq), pstr.coeff) for pstr in topaulistrings(psum)]
    return [[l, numbertojson(c)] for (l, c) in sort(pairs; by=first)]
end

"""
Propagates `observable`, a vector of `(symbols, qubits, coefficient)` with 1-based qubits, and writes the case.
`nothing` for a truncation stands for not using it. `weight_damping` is the `(gamma, threshold)` of `truncatedampingcoeff`.
With `heisenberg=false` the sum is propagated as a state. Without `store_terms` only the number of terms and the overlap are written.
"""
function writecase(name, nq, circuit, params, observable; max_weight=nothing, min_abs_coeff, max_freq=nothing,
    max_sins=nothing, min_rel_coeff=nothing, weight_damping=nothing, heisenberg=true, store_terms=true)

    psum = PauliSum(promote_type(Float64, (typeof(coeff) for (_, _, coeff) in observable)...), nq)
    for (symbols, qinds, coeff) in observable
        add!(psum, collect(symbols), collect(qinds), coeff)
    end

    kwargs = Dict{Symbol,Any}(:min_abs_coeff => min_abs_coeff, :heisenberg => heisenberg)
    isnothing(max_weight) || (kwargs[:max_weight] = max_weight)
    isnothing(max_freq) || (kwargs[:max_freq] = max_freq)
    isnothing(max_sins) || (kwargs[:max_sins] = max_sins)
    isnothing(min_rel_coeff) || (kwargs[:min_rel_coeff] = min_rel_coeff)
    if !isnothing(weight_damping)
        gamma, threshold = weight_damping
        kwargs[:customtruncfunc] = (pstr, coeff) -> truncatedampingcoeff(pstr, coeff, gamma, threshold)
    end
    result = propagate(circuit, psum, params; kwargs...)

    case = Dict(
        "name" => name,
        "reference" => "PauliPropagation.jl $(pkgversion(PauliPropagation))",
        "num_qubits" => nq,
        "circuit" => circuittojson(circuit, params),
        "observable" => [Dict("paulis" => join(string.(symbols)), "qubits" => collect(qinds) .- 1, "coeff" => numbertojson(coeff))
                         for (symbols, qinds, coeff) in observable],
        "max_weight" => max_weight,
        "min_abs_coeff" => min_abs_coeff,
        "max_freq" => max_freq,
        "max_sins" => max_sins,
        "min_rel_coeff" => min_rel_coeff,
        "weight_damping" => isnothing(weight_damping) ? nothing : collect(weight_damping),
        "heisenberg" => heisenberg,
        "num_terms" => length(result),
        "overlap_with_zero" => numbertojson(overlapwithzero(result)),
        "terms" => store_terms ? termstojson(result, nq) : nothing,
    )
    mkpath(OUT_DIR)
    open(joinpath(OUT_DIR, name * ".json"), "w") do io
        JSON.print(io, case)
    end
    println(rpad(name, 34), " gates=", rpad(length(circuit), 6), " terms=", rpad(length(result), 8),
        " overlap=", overlapwithzero(result))
    return result
end

"A random circuit of Clifford gates and Pauli rotations on up to three qubits."
function randomcircuit(rng, nq, ngates; one_qubit=[:H, :X, :Y, :Z, :S, :SX, :SY], two_qubit=[:CNOT, :CZ, :SWAP, :ZZpihalf])
    circuit = Gate[]
    params = Float64[]
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

"A random circuit as `randomcircuit` makes, with a random noise channel after about every other gate."
function randomnoisycircuit(rng, nq, ngates; kwargs...)
    unitary_circuit, unitary_params = randomcircuit(rng, nq, ngates; kwargs...)
    remaining = Iterators.Stateful(unitary_params)
    noise_types = [DepolarizingNoise, PauliXNoise, PauliYNoise, PauliZNoise, AmplitudeDampingNoise]
    circuit = Gate[]
    params = Float64[]
    for gate in unitary_circuit
        push!(circuit, gate)
        gate isa ParametrizedGate && push!(params, popfirst!(remaining))
        if rand(rng) < 0.5
            push!(circuit, rand(rng, noise_types)(rand(rng, 1:nq)))
            push!(params, 0.3 * rand(rng))
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

# Every kind of noise channel, with nothing truncated.
let nq = 6, rng = MersenneTwister(3)
    circuit, params = randomnoisycircuit(rng, nq, 60)
    observable = [([:Z], [1], 1.0), ([:X, :Y], [3, 6], -0.5), ([:Z, :Z, :Z], [2, 4, 5], 0.25)]
    writecase("noisy_mixed_gates_exact_6", nq, circuit, params, observable; min_abs_coeff=0.0)
end

# A Trotter circuit with depolarizing noise and amplitude damping on every qubit after every layer.
let nq = 24, nlayers = 10
    topology = bricklayertopology(nq)
    circuit = Gate[]
    params = Float64[]
    for _ in 1:nlayers
        rzzlayer!(circuit, topology)
        append!(params, fill(0.3, length(topology)))
        rxlayer!(circuit, nq)
        append!(params, fill(0.3, nq))
        for q in 1:nq
            push!(circuit, DepolarizingNoise(q), AmplitudeDampingNoise(q))
            push!(params, 0.02, 0.01)
        end
    end
    writecase("noisy_tfi_24", nq, circuit, params, [([:Z], [12], 1.0)]; max_weight=6, min_abs_coeff=1e-5)
end

# The counters of the rotations a term came through, with Clifford gates and noise in between.
# The tests read :SY and :ZZpihalf as rotations, which would count, so the circuits leave them out.
let nq = 10, rng = MersenneTwister(4)
    circuit, params = randomnoisycircuit(rng, nq, 60; one_qubit=[:H, :X, :Y, :Z, :S, :SX], two_qubit=[:CNOT, :CZ, :SWAP])
    observable = [([:Z], [5], 1.0), ([:X, :X], [2, 8], 0.5)]
    writecase("frequency_truncation_10", nq, circuit, params, observable; max_freq=9, min_abs_coeff=0.0)
    writecase("sine_truncation_10", nq, circuit, params, observable; max_sins=3, min_abs_coeff=0.0)
    writecase("frequency_and_sine_truncation_10", nq, circuit, params, observable; max_freq=11, max_sins=4, max_weight=6,
        min_abs_coeff=1e-8)
end

# A threshold relative to the largest coefficient, which Clifford gates do not apply.
let nq = 12, rng = MersenneTwister(5)
    circuit, params = randomnoisycircuit(rng, nq, 120)
    params .*= 0.3
    observable = [([:Z], [6], 1.0), ([:Y, :Z], [3, 9], 0.02)]
    writecase("relative_truncation_12", nq, circuit, params, observable; min_rel_coeff=2e-3, min_abs_coeff=1e-10)
end

# Coefficients damped by the weight of their Pauli string before they are compared with a threshold.
let nq = 12
    circuit = heisenbergtrottercircuit(nq, 4)
    params = fill(0.25, countparameters(circuit))
    writecase("weight_damping_12", nq, circuit, params, [([:Z], [6], 1.0)]; weight_damping=(0.4, 1e-4), min_abs_coeff=1e-10)
end

# A rule that reads the Pauli string makes Clifford gates truncate as well. Here that is the
# damped coefficient, in a circuit where Clifford gates change the weight between rotations.
let nq = 10, rng = MersenneTwister(9)
    circuit, params = randomcircuit(rng, nq, 140; one_qubit=[:H, :X, :Y, :Z, :S, :SX], two_qubit=[:CNOT, :CZ, :SWAP])
    params .*= 0.3
    observable = [([:Z], [5], 1.0), ([:X, :X], [2, 8], 0.5)]
    writecase("weight_damping_clifford_10", nq, circuit, params, observable; weight_damping=(0.3, 1e-4), min_abs_coeff=1e-10)
end

# A Clifford gate that truncates still does not apply the relative threshold: in a circuit of
# Clifford gates alone the small coefficients stay, and only the weight cap drops terms.
let nq = 10, rng = MersenneTwister(10)
    one_qubit, two_qubit = [:H, :X, :Y, :Z, :S, :SX], [:CNOT, :CZ, :SWAP]
    circuit = Gate[rand(rng) < 0.4 ? CliffordGate(rand(rng, one_qubit), [rand(rng, 1:nq)]) :
                   CliffordGate(rand(rng, two_qubit), shuffle(rng, 1:nq)[1:2]) for _ in 1:30]
    observable = unique(term -> term[1:2], [(rand(rng, [:X, :Y, :Z], 3), sort(shuffle(rng, 1:nq)[1:3]), 10.0^(-3 * rand(rng))) for _ in 1:150])
    writecase("relative_and_weight_truncation_clifford_10", nq, circuit, Float64[], observable;
        min_rel_coeff=0.1, max_weight=7, min_abs_coeff=1e-10)
end

# Complex coefficients, exactly and with a threshold on their absolute value.
let nq = 7, rng = MersenneTwister(6)
    circuit, params = randomnoisycircuit(rng, nq, 70)
    observable = [([:Z], [1], 1.0 + 0.5im), ([:X, :Y], [3, 6], -0.5im), ([:Y, :Z, :X], [2, 4, 7], 0.25 - 0.25im)]
    writecase("complex_mixed_gates_exact_7", nq, circuit, params, observable; min_abs_coeff=0.0)
    params .*= 0.3
    writecase("complex_mixed_gates_truncated_7", nq, circuit, params, observable; min_abs_coeff=1e-3, max_weight=5)
end

"A random circuit as `randomnoisycircuit` makes, without the amplitude damping that only acts on observables."
function randompaulinoisecircuit(rng, nq, ngates)
    unitary_circuit, unitary_params = randomcircuit(rng, nq, ngates)
    remaining = Iterators.Stateful(unitary_params)
    circuit = Gate[]
    params = Float64[]
    for gate in unitary_circuit
        push!(circuit, gate)
        gate isa ParametrizedGate && push!(params, popfirst!(remaining))
        if rand(rng) < 0.3
            push!(circuit, rand(rng, [DepolarizingNoise, PauliXNoise, PauliYNoise, PauliZNoise])(rand(rng, 1:nq)))
            push!(params, 0.3 * rand(rng))
        end
    end
    return circuit, params
end

# A state propagated forwards through the circuit.
let nq = 7, rng = MersenneTwister(7)
    circuit, params = randompaulinoisecircuit(rng, nq, 80)
    state = [([:I], [1], 1.0), ([:Z], [1], 1.0), ([:X, :Y], [3, 6], -0.5), ([:Z, :Z, :X], [2, 4, 7], 0.25)]
    writecase("state_mixed_gates_exact_7", nq, circuit, params, state; heisenberg=false, min_abs_coeff=0.0)
    params .*= 0.3
    writecase("state_mixed_gates_truncated_7", nq, circuit, params, state; heisenberg=false, min_abs_coeff=1e-4, max_weight=5)
end

"A random unitary matrix on `k` qubits."
randomunitary(rng, k) = Matrix(qr(randn(rng, ComplexF64, 2^k, 2^k)).Q)

# Gates given by a unitary matrix or by a Pauli transfer matrix, between rotations and T gates.
let nq = 6, rng = MersenneTwister(8)
    damping = 0.2
    # amplitude damping as it acts on an observable: Z -> (1 - damping) Z + damping I
    dampingptm = [1 0 0 damping; 0 sqrt(1 - damping) 0 0; 0 0 sqrt(1 - damping) 0; 0 0 0 1-damping]
    circuit = Gate[]
    params = Float64[]
    for _ in 1:12
        k = rand(rng, 1:2)
        push!(circuit, unitarygate(randomunitary(rng, k), shuffle(rng, 1:nq)[1:k]))
        push!(circuit, PauliRotation(rand(rng, [:X, :Y, :Z], 2), shuffle(rng, 1:nq)[1:2]))
        push!(params, 2pi * rand(rng) - pi)
        push!(circuit, TGate(rand(rng, 1:nq)))
        push!(circuit, TransferMapGate(dampingptm, rand(rng, 1:nq)))
    end
    observable = [([:Z], [3], 1.0), ([:X, :Y], [1, 5], -0.5)]
    writecase("matrix_gates_6", nq, circuit, params, observable; min_abs_coeff=1e-12)
end

# The circuit of the tests of PauliPropagation.jl whose expectation value they state.
let nq = 8, nl = 4
    circuit = hardwareefficientcircuit(nq, nl; topology=bricklayertopology(nq; periodic=false))
    Random.seed!(42)
    params = randn(countparameters(circuit))
    observable = [([:Z], [4], 1.0)]
    exact = writecase("upstream_hardware_efficient_8", nq, circuit, params, observable; min_abs_coeff=0.0, store_terms=false)
    @assert overlapwithzero(exact) ≈ 0.21720058439757214
    writecase("upstream_hardware_efficient_damped_8", nq, circuit, params, observable;
        weight_damping=(0.01, 1e-5), min_abs_coeff=1e-5, store_terms=false)
    writecase("upstream_hardware_efficient_relative_8", nq, circuit, params, observable;
        min_rel_coeff=1e-2, min_abs_coeff=1e-8)
end

# The Clifford gates of PauliPropagation.jl as tables: for every Pauli string on the qubits of the
# gate, numbered in base 4 with the first qubit as the lowest digit, its image and the sign.
let
    tables = Dict(string(symbol) => [[Int(image), sign] for (image, sign) in map] for (symbol, map) in clifford_map)
    mkpath(joinpath(OUT_DIR, "tables"))
    open(io -> JSON.print(io, tables), joinpath(OUT_DIR, "tables", "clifford_maps.json"), "w")
    println("clifford_maps: ", join(sort(collect(keys(tables))), " "))
end

# The topologies and circuits that PauliPropagation.jl builds, with 0-based qubits, for the tests
# that build the same ones. A circuit is listed without its angles.
let
    pairs(topology) = [[a - 1, b - 1] for (a, b) in topology]
    gatejson(gate::PauliRotation) = [join(string.(gate.symbols)), collect(gate.qinds .- 1)]
    gatejson(gate::CliffordGate) = [string(gate.symbol), collect(gate.qinds .- 1)]

    topologies = []
    for nq in 1:9, periodic in (false, true)
        push!(topologies, Dict("kind" => "bricklayer", "size" => [nq], "periodic" => periodic,
            "pairs" => pairs(bricklayertopology(nq; periodic))))
        push!(topologies, Dict("kind" => "staircase", "size" => [nq], "periodic" => periodic,
            "pairs" => pairs(staircasetopology(nq; periodic))))
    end
    for (nx, ny) in ((1, 1), (1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 3)), periodic in (false, true)
        push!(topologies, Dict("kind" => "rectangle", "size" => [nx, ny], "periodic" => periodic,
            "pairs" => pairs(rectangletopology(nx, ny; periodic))))
    end

    builders = Dict("hardware_efficient" => hardwareefficientcircuit, "efficient_su2" => efficientsu2circuit,
        "tfi_trotter" => tfitrottercircuit, "tilted_tfi_trotter" => tiltedtfitrottercircuit)
    circuits = []
    for (kind, builder) in builders, nl in (0, 1, 3)
        push!(circuits, Dict("kind" => kind, "num_qubits" => 5, "layers" => nl, "topology" => nothing,
            "gates" => gatejson.(builder(5, nl))))
        topology = rectangletopology(2, 3; periodic=true)
        push!(circuits, Dict("kind" => kind, "num_qubits" => 6, "layers" => nl, "topology" => pairs(topology),
            "gates" => gatejson.(builder(6, nl; topology))))
    end

    open(joinpath(OUT_DIR, "tables", "circuits.json"), "w") do io
        JSON.print(io, Dict("topologies" => topologies, "circuits" => circuits))
    end
    println("circuits: ", length(topologies), " topologies, ", length(circuits), " circuits")
end
