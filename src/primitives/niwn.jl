########################################
# Normal Inverse Wishart Normal (NIWN) #
########################################

import Distributions

"""
Parameters for a collapsed normal inverse Wishart normal (NIWN) model, which is:

    $\Lambda \sim \mathcal{W}(\Lambda_0, n_0)$ precision matrix has Wishart dist.

    $\mu | \Lambda \sim \mathcal{N}(\mu_0, (k_0 \Lambda)^{-1})$

    $x_i \sim \mathcal{N}(\mu, \Lambda^{-1})$ for $i=1\dots N$

Named according to:

    http://thaines.com/content/misc/gaussian_conjugate_prior_cheat_sheet.pdf
"""
struct NIWNParams
    mu::Vector{Float64}
    k::Float64
	n::Float64
    T::Matrix{Float64} # inverse of \Lambda_0
end

"""
Stores sufficient statistics for collapsed normal inverse Wishart normal (NIWN) model
"""
mutable struct NIWNState
    n::Int
    x_total::Vector{Float64}
    S_total::Matrix{Float64}
    function NIGNState(dimension::Int)
        new(0, zeros(dimension), zeros(dimension, dimension))
    end
end
function incorporate!(state::NIWNState, x::Vector{Float64})
    state.n += 1
    state.x_total += x
    state.S_total += (x * x')
end

function unincorporate!(state::NIGNState, x::Vector{Float64})
    @assert state.n > 0
    state.n -= 1
    state.x_total -= x
    state.S_total -= (x * x')
end

function posterior(prior:NIWNParams, state::NIWNState)
    n = prior.n + state.n
    k = prior.k + state.n
    mu = (prior.k * prior.mu + state.x_total)/k
    T = prior.T
    T += state.S_total - (state.x_total * state.x_total')/state.n
    xdiff = (state.x_total/state.n) - prior.mu
    T += ((prior.k * state.n) / k) * diff * diff'
    return NIWNParams(mu, k, n, T)
end

struct MultivariateStudentTParams
    v::Float64
    mu::Vector{Float64}
    Sigma_inverse::Matrix{Float64}
end

function multivariate_student_t_logpdf(x::Vector{Float64}, params::MultivariateStudentTParams)
    d = length(x)
    lpdf = lgamma(0.5*(params.v+d))
    lpdf -= lgamma(0.5*params.v)
    lpdf -= (0.5*d) * log(params.v * pi)
    lpdf += 0.5*logdet(params.Sigma_inverse) # determinant of inverse is 1/determinant
    diff = x - params.mu
    lpdf -= 0.5*(params.v+d)*log1p((diff'*params.Sigma_inverse * diff)/params.v)
    return lpdf
end

function predictive_logp(x::Vector{Float64}, state::NIWNState, prior::NIWNParams)
    posterior_params = posterior(prior, state)
    dim = length(x)
    n_minus_dim_plus_1 = posterior_params.n - length(dim) + 1
    v = n_minus_dim_plus_1
    Sigma_inverse = (posterior_params.k*n_minus_dim_plus_1 / (posterior_params.k + 1))* posterior_params.T
    student_t_params = MultivariateStudentTParams(v, posterior_params.mu, Sigma_inverse)
    return multivariate_student_t_logpdf(x, student_t_params)
end

function predictive_sample(state::NIWNState, prior::NIWNParams)
    # sample the covariance from inverse wishart 
    # TODO
    # sample the mean from multivariate normal given scaled covariance with mean prior.mu
    # TODO
    # sample the prediction from multivariate normal with given mean and covariance
    # TODO
end
