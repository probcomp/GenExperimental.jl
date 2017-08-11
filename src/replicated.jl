###################################
# Generator replicator combinator #
###################################

"""
A generator that wraps another generator with more accurate scores.
"""
struct ReplicatedGenerator{T} <: Generator{T}
    inner_generator::Generator{T}
    num::Int
end

replicated(g::Generator, num::Int) = ReplicatedGenerator(g, num)

empty_trace(g::ReplicatedGenerator) = empty_trace(g.inner_generator)

function simulate!(g::ReplicatedGenerator{T}, args::Tuple, outputs, conditions, trace::T) where {T}
    # run one simulate! and num-1 regenerate!s
    scores = Vector{Float64}(g.num)
    (scores[1], retval) = simulate!(g.inner_generator, args, outputs, conditions, trace)
    for i=2:g.num
        # NOTE: opportunity for performance optimization?
        supplementary_trace = empty_trace(g.inner_generator)
        for addr in conditions
            supplementary_trace[addr] = trace[addr]
        end
        for addr in outputs
            supplementary_trace[addr] = trace[addr]
        end
        (scores[i], _) = regenerate!(g.inner_generator, args, outputs, conditions, supplementary_trace)
    end
    score = logsumexp(scores) - log(g.num)

    # return the return value from the simulate
    (score, retval)
end

function regenerate!(g::ReplicatedGenerator{T}, args::Tuple, outputs, conditions, trace::T) where {T}
    # run num regenerate!s
    scores = Vector{Float64}(g.num)
    local retval
    for i=1:g.num
        (scores[i], retval) = regenerate!(g.inner_generator, args, outputs, conditions, trace)
    end
    score = logsumexp(scores) - log(g.num)

    # return one of the return values
    (score, retval)
end

export replicated
