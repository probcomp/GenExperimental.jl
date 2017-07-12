import DataStructures.Stack
import DataStructures.OrderedDict
import Gen.simulate
import Gen.regenerate

# data structure for storing sufficient statistics for CRP

type CRPState

    # map from cluster index size of cluser
    counts::OrderedDict{Int, Int}

    # sum of values(counts)
    total_count::Int

    # reuse ids for the new clusters by pushing them onto a stack
    # this is necessary because we may do billions of Gibbs sweeps
    free::Stack{Int}

    # the next cluster id to allocate, after free stack is empty
    next_cluster::Int

    # create an empty CRP state
    function CRPState()
        free = Stack(Int)
        push!(free, 1)
        new(Dict{Int, Int}(), 0, free, 2)
    end
end

next_new_cluster(state::CRPState) = DataStructures.top(state.free)
has_cluster(state::CRPState, cluster::Int) = haskey(state.counts, cluster)
counts(state::CRPState, cluster::Int) = state.counts[cluster]
clusters(state::CRPState) = keys(state.counts)

function joint_log_probability(state::CRPState, alpha::T) where {T}
    N = state.total_count
    ll = length(state.counts) * log(alpha)
    ll += sum(lgamma.(collect(values(state.counts))))
    ll += lgamma(alpha) - lgamma(N + alpha)
    ll
end

function incorporate!(state::CRPState, cluster::Int)
    @assert !haskey(state.counts, next_new_cluster(state))
    @assert (cluster == next_new_cluster(state)) || haskey(state.counts, cluster)
    if cluster == next_new_cluster(state)
        # allocate a new cluster
        state.counts[cluster] = 0
        pop!(state.free)
        if isempty(state.free)
            @assert !haskey(state.counts, state.next_cluster)
            push!(state.free, state.next_cluster)
            state.next_cluster += 1
        end
    else
        @assert state.counts[cluster] > 0
    end
    state.total_count += 1
    state.counts[cluster] += 1
end

function unincorporate!(state::CRPState, cluster::Int)
    @assert !haskey(state.counts, next_new_cluster(state))
    state.counts[cluster] -= 1
    if state.counts[cluster] == 0
        # free the empyt cluster
        delete!(state.counts, cluster)
        push!(state.free, cluster)
    end
    state.total_count -= 1
end

# draw_crp -------------------------------------------------------

# NOTE: this module does not mutate the CRP state

struct CRPDraw <: Gen.Module{Int} end

function simulate(::CRPDraw, state::CRPState, alpha::T) where {T}
    # NOTE: does not incorporate the cluster draw into the CRP state
    clusters = collect(keys(state.counts))
    probs = Array{Float64,1}(length(clusters) + 1)
    for (j, cluster) in enumerate(clusters)
        probs[j] = state.counts[cluster]
    end
    probs[end] = alpha
    probs = probs / sum(probs)
    j = rand(Categorical(probs))
    if (j == length(clusters) + 1)
        # new cluster
        cluster = next_new_cluster(state)
    else
        cluster = clusters[j]
    end
    (cluster, log(probs[j]))
end

function regenerate(::CRPDraw, cluster::Int, state::CRPState, alpha::T) where {T}
    new_cluster = next_new_cluster(state)
    if cluster == new_cluster
        log(alpha) - log(state.total_count + alpha)
    else
        log(state.counts[cluster]) - log(state.total_count + alpha)
    end
end

register_module(:draw_crp, CRPDraw())
function draw_crp(state::CRPDraw, alpha::T) where {T}
    simulate(CRPDraw(), state, alpha)[1]
end

export CRPState
export next_new_cluster
export has_cluster
export counts
export clusters
export joint_log_probability
export incorporate!
export unincorporate!
export CRPDraw
export draw_crp
