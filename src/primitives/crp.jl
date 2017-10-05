####################################
# Chinese restaurant process (CRP) #
####################################

import DataStructures.Queue
import DataStructures.OrderedDict

"""
Data structure for storing sufficient statistics for CRP
"""
mutable struct CRPState

    # map from table index to size of table
    counts::OrderedDict{Int, Int}

    # number of elements
    total_count::Int

    # reuse ids for the new tables by pushing them into a set
    # and only increment next_table when the set is empty
	# this is necessary because we may do billions of incorporates and
	# unincorporates
    free::Set{Int}

    # the next table id to allocate, after free stack is empty
    next_table::Int

    # create an empty CRP state
    function CRPState()
        free = Set{Int}()
        push!(free, 1)
        new(OrderedDict{Int, Int}(), 0, free, 2)
    end
end

function new_table(state::CRPState)
    table = next(state.free, start(state.free))[1]
    @assert !haskey(state.counts, table)
    table
end

function draw(state::CRPState, alpha::Real)
    tables = collect(keys(state.counts))
    probs = Array{Float64,1}(length(tables) + 1)
    for (j, table) in enumerate(tables)
        probs[j] = state.counts[table]
    end
    probs[end] = alpha
    probs = probs / sum(probs)
    j = rand(Distributions.Categorical(probs))
    if (j == length(tables) + 1)
        return new_table(state)
    else
        return tables[j]
    end
end

has_table(state::CRPState, table::Int) = haskey(state.counts, table)
counts(state::CRPState, table::Int) = state.counts[table]
get_tables(state::CRPState) = keys(state.counts)

function log_joint_probability(state::CRPState, alpha)
    N = state.total_count
    ll = length(state.counts) * log(alpha)
    ll += sum(lgamma.(collect(values(state.counts))))
    ll += lgamma(alpha) - lgamma(N + alpha)
    ll
end

function incorporate!(state::CRPState, table::Int)
    if !haskey(state.counts, table)
        @assert table >= state.next_table || table in state.free
        if table in state.free
            Base.delete!(state.free, table)
        end
        # allocate a new table
        state.counts[table] = 0
        if isempty(state.free)
            # add a new table to the free set
            while haskey(state.counts, state.next_table)
                state.next_table += 1
            end
            push!(state.free, state.next_table)
            state.next_table += 1
        end
    else
        @assert state.counts[table] > 0
    end
    state.total_count += 1
    state.counts[table] += 1
    return table
end

function unincorporate!(state::CRPState, table::Int)
    @assert haskey(state.counts, table)
    @assert state.counts[table] > 0
    state.counts[table] -= 1
    if state.counts[table] == 0
        # free the empty table
        Base.delete!(state.counts, table)
        push!(state.free, table)
    end
    state.total_count -= 1
end

export CRPState
export new_table
export has_table
export counts
export get_tables
export log_joint_probability
export incorporate!
export unincorporate!
export draw

#######################
# CRP joint generator #
#######################

mutable struct CRPTrace{T} <: Trace
    assignments::Dict{T,Int}
    state::CRPState
end

function CRPTrace(::Type{T}) where {T}
    CRPTrace(Dict{T,Int}(), CRPState())
end

function Base.haskey(t::CRPTrace{T}, addr::Tuple{T}) where {T}
    haskey(t.assignments, addr[1])
end

function Base.delete!(t::CRPTrace{T}, addr::Tuple{T}) where {T}
    unincorporate!(t.state, t.assignments[addr[1]])
    delete!(t.assignments, addr[1])
end

function Base.getindex(t::CRPTrace{T}, addr::Tuple{T}) where {T}
    t.assignments[addr[1]]
end

function Base.setindex!(t::CRPTrace{T}, value::Int, addr::Tuple{T}) where {T}
    if haskey(t.assignments, addr[1])
        delete!(t, addr)
    end
    table = incorporate!(t.state, value)
    t.assignments[addr[1]] = table
end

new_table(t::CRPTrace) = new_table(t.state)

struct CRPGenerator{T} <: Generator{CRPTrace{T}} end

function CRPGenerator(::Type{T}) where {T}
    CRPGenerator{T}()
end

function empty_trace(::CRPGenerator{T}) where {T}
    return CRPTrace(T)
end

function regenerate!(::CRPGenerator{T}, args::Tuple{Any,Any}, outputs, conditions, trace::CRPTrace{T}) where {T}
    (addresses, alpha) = args

    # no assignments other than for outputs and conditions may be in the trace
    # every assignment in the trace must be registered in the argument 'addresses'
    for addr in keys(trace.assignments)
        if !(addr in outputs || addr in conditions)
            error("address $addr was in trace but not in outputs or conditions")
        end
        if !(addr in addresses)
            error("address $addr was in trace but not in argument addresses")
        end
    end

    # check that output and condition addresses are in the argument 'addresses'
    for addr in (outputs..., conditions...)
        if !(length(addr) == 1 && addr[1] in addresses)
            error("address $addr was in outputs or conditions but is not in argument addresses")
        end
    end
    
    # joint probability of the conditions and outputs
    score_combined = log_joint_probability(trace.state, alpha)
    for addr in outputs
        unincorporate!(trace.state, trace.assignments[addr[1]])
    end

    # joint probability of just the conditions
    score_condition_only = log_joint_probability(trace.state, alpha)

    # restore the trace to its original state
    for addr in outputs
        incorporate!(trace.state, trace.assignments[addr[1]])
    end

    return (score_combined - score_condition_only, copy(trace.assignments))
end

function simulate!(::CRPGenerator{T}, args::Tuple{Any,Any}, outputs, conditions, trace::CRPTrace{T}) where {T}
    (addresses, alpha) = args

    # no assignments other than for conditions may be in the trace
    # every assignment in the trace must be registered in the argument 'addresses'
    for addr in keys(trace.assignments)
        if !(addr in conditions)
            error("address $addr was in trace but not in outputs or conditions")
        end
        if !(addr in addresses)
            error("address $addr was in trace but not in argument addresses")
        end
    end

    # check that output and condition addresses are in the argument 'addresses'
    for addr in (outputs..., conditions...)
        if !(length(addr) == 1 && addr[1] in addresses)
            error("address $addr was in outputs or conditions but is not in argument addresses")
        end
    end

    # if outputs and conditions are both empty, then simulate all addresses in argument 'addresses'
    if isempty(outputs) && isempty(conditions)
        for addr in addresses
            table = draw(trace.state, alpha)
            trace.assignments[addr] = table
            incorporate!(trace.state, table)
        end
        return (0., copy(trace.assignments))
    else

        # joint probability of just the conditions
        score_condition_only = log_joint_probability(trace.state, alpha)
    
        for addr in outputs
            table = draw(trace.state, alpha)
            trace.assignments[addr[1]] = table
            incorporate!(trace.state, table)
        end
    
        # joint probability of conditions and outputs
        score_combined = log_joint_probability(trace.state, alpha)

        return (score_combined - score_condition_only, copy(trace.assignments))
    end
end

export CRPTrace
export CRPGenerator
