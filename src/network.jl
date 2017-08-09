"""
TODO: handle non-atomic generators, and sub-traces.
"""
mutable struct GeneratorNetwork{T} <: Generator{ProgramTrace{T}}
    generators::Dict{T, AtomicGenerator}
    preludes::Dict{T, Function}
    dag::DAG
    frozen::Bool
end


function add_node!(network::GeneratorNetwork{T}, generator::AtomicGenerator,
                   addr::T, inputs::Vector{T}, prelude::Function) where {T}
    if network.frozen
        error("Network is frozen")
    end
    add_node!(network.dag, addr, inputs)
    network.generators[addr] = generator
    network.preludes[addr] = prelude
end

function freeze!(network::GeneratorNetwork)
    if network.frozen
        error("Network is frozen")
    end
    network.frozen = true
end


# TODO in both, check that conditions do not cause forward simulation to no longer be the conditional distribution.
# TODO in both, check that all of the conditions and none of the outputs are given in the trace
function regenerate!(network::GeneratorNetwork{T}, args::Tuple, outputs, conditions, trace::ProgramTrace{T})
    order = execution_order(network.dag, outputs, conditions)
    score = 0.
    for addr in order
        extra_args = map((parent_addr) -> trace[parent_addr], network.dag.inputs)
        subgenerator_args = network.preludes[addr](args, extra_args...)
        # TODO which outputs and which conditions?
        score += regenerate!(network.generators[addr], subgenerator_args, Set([()]), Set())
    end
    (score, nothing)
end

function simulate!(network::GeneratorNetwork, outputs, conditions, trace::ProgramTrace)
    order = execution_order(network.dag, outputs, conditions)
    score = 0.
    for addr in order
        extra_args = map((parent_addr) -> trace[parent_addr], network.dag.inputs)
        subgenerator_args = network.preludes[addr](args, extra_args...)
        # TODO which outputs and which conditions?
        score += regenerate!(network.generators[addr], subgenerator_args, Set([()]), Set())
    end
    (score, nothing)
end
