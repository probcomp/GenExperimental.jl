############################################
# CRP joint Generator with custom subtrace #
############################################

# a generator over values [a1, a2, a3, ..., an] where n is the argument
# can re-use traces for different n
# the sub-trace can be consrained at arbitrary subset of indices

# the score returned by generate! is the log joint probability of the constrained assignments
# the unconstrained assignments are sampled from the conditoinal distribution
# NOTE: this specification of generates follows the 'marginal likelihood estimate' semantics
# and not the importance-weighted semantics

# TODO handle 'propose!

mutable struct CRPJointTrace
    # NOTE: the same struct can be used with multiple N

    # the state that has incorporated all constrained draws
    constrained_state::CRPState
    
    # the maximum constrained index
    max_constrained::Int

    # the set of constrained indices (in [1, max_constrained])
    constrained::Set{Int} 

    # the set of unconstrained indices in [1, max_constrained)
    unconstrained::Set{Int} 

    # assignments for constrained or recorded assignments
    assignments::Dict{Int,Int} 
end

CRPJointTrace() = CRPJointTrace(CRPState(), 0, Set{Int}(), Set{Int}(), Dict{Int,Int}())

# next new cluster relative to the currently constrained assignments
next_new_cluster(trace::CRPJointTrace) = next_new_cluster(trace.constrained_state)

function constrain!(trace::CRPJointTrace, i::Int, cluster::Int)
    # cluster is either the next new cluster or an existing cluster
    # indices are in the range 1, 2, 3, ...
    if i < 1
        error("i=$i < 1")
    end
    if i in trace.constrained
        error("cannot constrain $i, it is already constrained")
    end
    trace.assignments[i] = cluster 
    incorporate!(trace.constrained_state, cluster)
    push!(trace.constrained, i)
    if i > trace.max_constrained
        for j=trace.max_constrained+1:i-1
            push!(trace.unconstrained, j)
        end
        trace.max_constrained = i
    end
end

function unconstrain!(trace::CRPJointTrace, i::Int)
    if !(i in trace.constrained)
        error("cannot unconstrain $i, it is not constrained")
    end
    unincorporate!(trace.constrained_state, assignments[i])
    delete!(trace.constrained, i)
    push!(trace.unconstrained, i)
end

hasvalue(trace::CRPJointTrace, i::Int) = haskey(trace.assignments, i)
value(trace::CRPJointTrace, i::Int) = trace.assignments[i]

type CRPJointGenerator <: Generator{CRPJointTrace} end

# samples new draws from the conditional distribution
# score is the marginal probability of the constrained choices
function draw_and_incorporate!(trace::CRPJointTrace, i::Integer, alpha)
    cluster = simulate(CRPDraw(), trace.constrained_state, alpha)
    incorporate!(trace.constrained_state, cluster)
    trace.assignments[i] = cluster
end

function generate!(::CRPJointGenerator, args::Tuple{Int,T}, trace::CRPJointTrace) where {T}
    max_index = args[1]
    alpha = args[2]

    # can't constrain addresses that aren't in the address space
    if trace.max_constrained > max_index
        error("max_constrained=$(trace.max_constrained) is greater than max_index=$max_index")
    end

    # the score is the log marginal probability of the constrained assignments
    score = joint_log_probability(trace.constrained_state, alpha)

    # sample unconstrained assignments from the conditional distributio given
    # constrained assignments
    new_unconstrained = Set{Int}()
    for i in trace.unconstrained
        if i <= max_index
            draw_and_incorporate!(trace, i, alpha)
            push!(new_unconstrained, i)
        else
            delete!(trace.unconstrained, i)
            if haskey(tace.assignments, i)
                delete!(trace.assignments, i)
            end
        end
    end
    for i=trace.max_constrained+1:max_index
        draw_and_incorporate!(trace, i, alpha)
        push!(new_unconstrained, i)
        push!(trace.unconstrained, i)
    end
    
    # remove sufficient statistics for unconstrained 
    for i in new_unconstrained
        @assert i in trace.unconstrained
        unincorporate!(trace.constrained_state, trace.assignments[i])
    end
    
    score
end

export CRPJointTrace
export CRPJointGenerator
export generate!
