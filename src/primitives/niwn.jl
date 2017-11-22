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
