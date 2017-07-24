######################################################
# Sufficient stastics for Chinese restaurant process #
######################################################

import DataStructures.Queue
import DataStructures.OrderedDict

# data structure for storing sufficient statistics for CRP

type CRPState

    # map from cluster index size of cluser
    counts::OrderedDict{Int, Int}

    # sum of values(counts)
    total_count::Int

    # reuse ids for the new clusters by pushing them into a set
    # and only increment next_cluster when the set is empty
    # this is necessary because we may do billions of Gibbs sweeps
    free::Set{Int}

    # the next cluster id to allocate, after free stack is empty
    next_cluster::Int

    # create an empty CRP state
    function CRPState()
        free = Set{Int}()
        push!(free, 1)
        new(OrderedDict{Int, Int}(), 0, free, 2)
    end
end

function new_cluster(state::CRPState)
    cluster = next(state.free, start(state.free))[1]
    @assert !haskey(state.counts, cluster)
    cluster
end
has_cluster(state::CRPState, cluster::Int) = haskey(state.counts, cluster)
counts(state::CRPState, cluster::Int) = state.counts[cluster]
get_clusters(state::CRPState) = keys(state.counts)

# TODO unecessary type parameter
function logpdf(state::CRPState, alpha::T) where {T}
    # alpha may be concrete or differentaible value
    N = state.total_count
    ll = length(state.counts) * log(alpha)
    ll += sum(lgamma.(collect(values(state.counts))))
    ll += lgamma(alpha) - lgamma(N + alpha)
    ll
end

const CRP_NEW_CLUSTER = -1

function incorporate!(state::CRPState, cluster::Int)
    if cluster == CRP_NEW_CLUSTER
        next = new_cluster(state)
        return incorporate!(state, next)
    end
    if !haskey(state.counts, cluster)
        @assert cluster >= state.next_cluster || cluster in state.free
        if cluster in state.free
            Base.delete!(state.free, cluster)
        end
        # allocate a new cluster
        state.counts[cluster] = 0
        if isempty(state.free)
            # add a new cluster to the free set
            while haskey(state.counts, state.next_cluster)
                state.next_cluster += 1
            end
            push!(state.free, state.next_cluster)
            state.next_cluster += 1
        end
    else
        @assert state.counts[cluster] > 0
    end
    state.total_count += 1
    state.counts[cluster] += 1
    return cluster
end

function unincorporate!(state::CRPState, cluster::Int)
    @assert haskey(state.counts, cluster)
    @assert state.counts[cluster] > 0
    state.counts[cluster] -= 1
    if state.counts[cluster] == 0
        # free the empty cluster
        Base.delete!(state.counts, cluster)
        push!(state.free, cluster)
    end
    state.total_count -= 1
end


##################################
# Generator for drawing from CRP #
##################################

# NOTE: does not mutate the CRP state

import Distributions

struct CRPDraw <: AssessableAtomicGenerator{Int} end

function logpdf(::CRPDraw, cluster::Int, state::CRPState, alpha)
    if cluster == new_cluster(state)
        log(alpha) - log(state.total_count + alpha)
    else
        log(state.counts[cluster]) - log(state.total_count + alpha)
    end
end

function simulate(::CRPDraw, state::CRPState, alpha)
    clusters = collect(keys(state.counts))
    probs = Array{Float64,1}(length(clusters) + 1)
    for (j, cluster) in enumerate(clusters)
        probs[j] = state.counts[cluster]
    end
    probs[end] = alpha
    probs = probs / sum(probs)
    j = rand(Distributions.Categorical(probs))
    
    # return the drawn cluster
    if (j == length(clusters) + 1)
        # new cluster
        new_cluster(state)
    else
        clusters[j]
    end
end

register_primitive(:draw_crp, CRPDraw)


############################################
# CRP joint Generator with custom subtrace #
############################################

# TODO the fact that its not a set is not enforced in the type

make_exchangeable_generator(:CRPJointTrace, :CRPJointGenerator, :crp_joint,
    Tuple{Set,Float64}, CRPState, CRPDraw, Int)

# next new cluster relative to the currently constrained assignments
new_cluster(trace::CRPJointTrace) = new_cluster(trace.trace.state)
has_cluster(trace::CRPJointTrace, cluster::Int) = haskey(trace.trace.state.counts, cluster)
get_clusters(trace::CRPJointTrace) = keys(trace.trace.state.counts)

export CRPState
export new_cluster
export has_cluster
export counts
export get_clusters
export logpdf
export CRP_NEW_CLUSTER
