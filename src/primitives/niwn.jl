########################################
# Normal Inverse Wishart Normal (NIWN) #
########################################

import Distributions

"""
Parameters for a normal inverse Wishart normal (NIW) distribution.
"""
struct NIWParams

    # mean of normal distribution on mean
    # corresponds to 'mu' in NIGNParams
    mu::Vector{Float64}

    # strength of prior on mean, equivalently relative precision of mean vs
    # precision of data. the precision matrix of the mean is k times the
    # precision matrix of the data.
    # corresponds to 'r' in NIGNParams
    k::Float64

    # degrees of freedom for the inverse Wishart distribution on covariance
    # corresponds to 'mu' in NIGNParams
	m::Float64

    # scale parameter for the inverse Wishart distribution on covariance
    # corresponds to 's' in NIGNParams
    Psi::Matrix{Float64}

    # number of dimensions
    d::Int
end

function NIWParams(mu::Vector{Float64}, k::Real, m::Real, Psi::Matrix{Float64})
    d = length(mu)
    if !isapprox(Psi, Psi')
        error("Scale matrix Psi is not symmetric")
    end
    Psi = (Psi + Psi')/2.
    @assert ishermitian(Psi)
    NIWParams(mu, k, m, Psi, d)
end

"""
Stores sufficient statistics for collapsed normal inverse Wishart normal (NIWN) model
"""
mutable struct NIWNState
    n::Int
    x_total::Vector{Float64}
    S_total::Matrix{Float64}
    function NIWNState(dimension::Int)
        new(0, zeros(dimension), zeros(dimension, dimension))
    end
end
function incorporate!(state::NIWNState, x::Vector{Float64})
    state.n += 1
    state.x_total += x
    state.S_total += (x * x')
end

function unincorporate!(state::NIWNState, x::Vector{Float64})
    @assert state.n > 0
    state.n -= 1
    state.x_total -= x
    state.S_total -= (x * x')
end

function posterior(prior::NIWParams, state::NIWNState)
    m = prior.m + state.n
    k = prior.k + state.n
    mu = (prior.k * prior.mu + state.x_total)/k
    Psi = prior.Psi
    if state.n > 0
        Psi += state.S_total - (state.x_total * state.x_total')/state.n
        xdiff = (state.x_total/state.n) - prior.mu
        Psi += ((prior.k * state.n) / k) * xdiff * xdiff'
    end
    return NIWParams(mu, k, m, Psi)
end

function multivariate_lgamma(dimension::Int, x::Float64)
    result = dimension * (dimension - 1) * log(pi) / 4
    for i=1:dimension
        result += lgamma(x + (1 - i)/2.)
    end
    return result
end

function log_z(params::NIWParams)
    lz = -0.5 * params.m * logdet(params.Psi)
    lz += 0.5 * params.m * params.d * log(2.)
    lz += multivariate_lgamma(params.d, params.m/2)
    lz += 0.5 * params.d * (log(2*pi) - log(params.k))
    return lz
end

function log_marginal_likelihood(state::NIWNState, prior_params::NIWParams)
    posterior_params = posterior(prior_params, state)
    result = -0.5 * prior_params.d * state.n * log(2*pi)
    result += log_z(posterior_params)
    result -= log_z(prior_params)
    return result
end

# TODO check that I'm the same as log_marginal_likelihood()
function log_marginal_likelihood_faster(state::NIWNState, prior_params::NIWParams)
    posterior_params = posterior(prior_params, state)
    result = -0.5 * prior_params.d * state.n * log(pi)
    result += 0.5 * prior_params.m * logdet(prior_params.Psi)
    result -= 0.5 * posterior_params.m * logdet(posterior_params.Psi)
    result += multivariate_lgamma(prior_params.d, posterior_params.m/2)
    result -= multivariate_lgamma(prior_params.d, prior_params.m/2)
    result += 0.5 * prior_params.d * (log(prior_params.k) - log(posterior_params.k))
    return result
end

function predictive_logp(x::Vector{Float64}, state::NIWNState, prior::NIWParams)
    log_marginal_likelihood_before = log_marginal_likelihood(state, prior)
    incorporate!(state, x)
    log_marginal_likelihood_after = log_marginal_likelihood(state, prior)
    unincorporate!(state, x)
    return log_marginal_likelihood_after - log_marginal_likelihood_before
end

function predictive_sample(state::NIWNState, prior_params::NIWParams)
    # TODO: consider a more efficient implementation based on multivariate student T:
    # e.g. https://journal.r-project.org/archive/2013-2/hofert.pdf

    posterior_params = posterior(prior_params, state)

    # sample the data covariance from inverse wishart 
    data_covariance = rand(Distributions.InverseWishart(posterior_params.m, posterior_params.Psi))

    # sample the mean from the multivariate normal
    mu_covariance = data_covariance / posterior_params.k
    mu = rand(Distributions.MvNormal(posterior_params.mu, mu_covariance))

    # sample the data
    x = rand(Distributions.MvNormal(mu, data_covariance))
end

# TODO test against the single-dimension code.

export NIWParams
export NIWNState
export posterior
export log_marginal_likelihood
export log_marginal_likelihood_faster
export multivariate_lgamma # TODO move to a math page
export predictive_logp
export predictive_sample

########################
# NIWN Joint generator #
########################

"""
Custom trace type for normal inverse Wishart normal generator.

Parametrized by the type of the addresses.
"""
mutable struct NIWNTrace{A} <: Trace
    data::Dict{A,Vector{Float64}}
    state::NIWNState
end

Base.keys(t::NIWNTrace) = keys(t.data)


function NIWNTrace(::Type{A}, dim::Int) where {A}
    NIWNTrace(Dict{A,Vector{Float64}}(), NIWNState(dim))
end

function Base.haskey(t::NIWNTrace{A}, addr::Tuple{A}) where {A}
    haskey(t.data, addr[1])
end

function Base.delete!(t::NIWNTrace{A}, addr::Tuple{A}) where {A}
    unincorporate!(t.state, t.data[addr[1]])
    delete!(t.data, addr[1])
end

function Base.getindex(t::NIWNTrace{A}, addr::Tuple{A}) where {A}
    t.data[addr[1]]
end

function Base.getindex(t::NIWNTrace, addr::Tuple{})
    copy(t.data)
end

function Base.setindex!(t::NIWNTrace{A}, value::Vector{Float64}, addr::Tuple{A}) where {A}
    if haskey(t.data, addr[1])
        delete!(t, addr)
    end
    incorporate!(t.state, value)
    t.data[addr[1]] = value
end

Base.haskey(t::NIWNTrace{A}, addr_element::A) where {A} = haskey(t, (addr_element,))

Base.delete!(t::NIWNTrace{A}, addr_element::A) where {A} = delete!(t, (addr_element,))

Base.getindex(t::NIWNTrace{A}, addr_element::A) where {A} = t[(addr_element,)]

Base.setindex!(t::NIWNTrace{A}, value::Vector{Float64}, addr_element::A) where {A} = begin t[(addr_element,)] = value end


struct NIWNGenerator{A} <: Generator{NIWNTrace{A}}
    dim::Int
end

function NIWNGenerator(::Type{A}, dim::Int) where {A}
    NIWNGenerator{A}(dim)
end

function empty_trace(generator::NIWNGenerator{A}) where {A}
    return NIWNTrace(A, generator.dim)
end

function check_trace_addresses(trace::NIWNTrace, outputs, conditions, addresses)
    # no data other than for outputs and conditions may be in the trace
    # every datum in the trace must be registered in the argument 'addresses'
    for addr in keys(trace.data)
        if !(addr in outputs || addr in conditions)
            error("address $addr was in trace but not in outputs or conditions")
        end
        if !(addr in addresses)
            error("address $addr was in trace but not in argument addresses")
        end
    end
end

function regenerate!(::NIWNGenerator{A}, args::Tuple{Any,NIWParams,Bool}, outputs, conditions, trace::NIWNTrace{A}) where {A}
    # addresses is the full address space. all outputs and conditions must be a
    # subset of that address space.  the trace must also not contain any other
    # addresses besides those in outputs and conditions.  these preconditions
    # are only checked if safe=true
    # the implementation is optimized for small outputs and large conditions.
    (addresses, params, safe) = args

    if safe
        check_trace_addresses(trace, outputs, conditions, addresses)

        # check that output and condition addresses are in the argument 'addresses'
        for addr in (outputs..., conditions...)
            if !(length(addr) == 1 && addr[1] in addresses)
                error("address $addr was in outputs or conditions but is not in argument addresses")
            end
        end
    end

    # joint probability of the conditions and outputs
    score_combined = log_marginal_likelihood(trace.state, params)
    for addr in outputs
        unincorporate!(trace.state, trace.data[addr[1]])
    end

    # joint probability of just the conditions
    score_condition_only = log_marginal_likelihood(trace.state, params)

    # restore the trace to its original state
    for addr in outputs
        incorporate!(trace.state, trace.data[addr[1]])
    end

    return (score_combined - score_condition_only, copy(trace.data))
end

function simulate!(::NIWNGenerator{A}, args::Tuple{Any,NIWParams,Bool}, outputs, conditions, trace::NIWNTrace{A}) where {A}
    (addresses, params, safe) = args
    if safe
        check_trace_addresses(trace, outputs, conditions, addresses)

        # check that output and condition addresses are in the argument 'addresses'
        for addr in (outputs..., conditions...)
            if !(length(addr) == 1 && addr[1] in addresses)
                error("address $addr was in outputs or conditions but is not in argument addresses")
            end
        end
    end

    # if outputs and conditions are both empty, then simulate all addresses in argument 'addresses'
    if isempty(outputs) && isempty(conditions)
        for addr in addresses
            value = predictive_sample(trace.state, params)
            trace.data[addr] = value
            incorporate!(trace.state, value)
        end
        return (0., copy(trace.data))
    else

        # joint probability of just the conditions
        score_condition_only = log_marginal_likelihood(trace.state, params)

        for addr in outputs
            value = predictive_sample(trace.state, params)
            trace.data[addr[1]] = value
            incorporate!(trace.state, value)
        end

        # joint probability of conditions and outputs
        score_combined = log_marginal_likelihood(trace.state, params)

        return (score_combined - score_condition_only, copy(trace.data))
    end
end

export NIWNTrace
export NIWNGenerator
