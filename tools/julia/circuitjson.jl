# Circuits of PauliPropagation.jl as JSON, with 0-based qubits.

label(term, nq) = join(("IXYZ"[getpauli(term, q)+1] for q in 1:nq))

"A real number as it is, a complex number as its real and imaginary part."
numbertojson(x::Real) = x
numbertojson(x::Complex) = [real(x), imag(x)]

"The unitary matrices that transfer map gates were made from, for `gatetojson`."
const UNITARIES = IdDict{Any,Matrix{ComplexF64}}()

"A `TransferMapGate` of the unitary `matrix`, which `gatetojson` writes along with the gate."
function unitarygate(matrix, qinds)
    gate = TransferMapGate(matrix, qinds)
    UNITARIES[gate] = matrix
    return gate
end

function gatetojson(gate::PauliRotation, theta)
    return Dict("gate" => "rotation", "paulis" => join(string.(gate.symbols)),
        "qubits" => gate.qinds .- 1, "theta" => theta)
end
gatetojson(gate::CliffordGate) = Dict("gate" => string(gate.symbol), "qubits" => gate.qinds .- 1)
gatetojson(gate::TGate) = Dict("gate" => "T", "qubits" => [gate.qind - 1])
# A gate with its parameter fixed is written as the gate, marked as frozen.
gatetojson(gate::FrozenGate) = merge(gatetojson(gate.gate, gate.parameter), Dict("frozen" => true))
function gatetojson(gate::ParametrizedNoiseChannel, strength)
    return Dict("gate" => string(nameof(typeof(gate))), "qubits" => [gate.qind - 1], "strength" => strength)
end

# The entry in row i and column j is the factor of the Pauli string i in the image of the string j.
function gatetojson(gate::TransferMapGate)
    columns = collect(gate.transfer_map)
    matrix = [zeros(length(columns)) for _ in columns]
    for (j, column) in enumerate(columns), (image, factor) in column
        matrix[Int(image)+1][j] = factor
    end
    json = Dict{String,Any}("gate" => "TransferMap", "qubits" => gate.qinds .- 1, "matrix" => matrix)
    if haskey(UNITARIES, gate)
        unitary = UNITARIES[gate]
        json["unitary"] = [[numbertojson(complex(unitary[i, j])) for j in axes(unitary, 2)] for i in axes(unitary, 1)]
    end
    return json
end

function circuittojson(circuit, params)
    remaining = Iterators.Stateful(params)
    return [gate isa ParametrizedGate ? gatetojson(gate, popfirst!(remaining)) : gatetojson(gate) for gate in circuit]
end
