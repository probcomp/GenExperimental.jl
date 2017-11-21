########################################
# Normal Inverse Wishart Normal (NIWN) #
########################################

import Distributions

"""
Parameters for a collapsed normal inverse Wishart normal (NIWN) model, which is:

    $\Lambda \sim \mathcal{W}(\Lambda_0, n_0)$

    $\mu | \Lambda \sim \mathcal{N}(\mu_0, (k_0 \Lambda)^{-1})$

    $x_i \sim \mathcal{N}(\mu, \Lambda^{-1})$ for $i=1\dots N$

Named according to:

    http://thaines.com/content/misc/gaussian_conjugate_prior_cheat_sheet.pdf
"""
struct NIWNParams
    mu::Vector{Float64}
    k::Float64
	n::Float64
    S::Matrix{Float64} # inverse of \Lambda_0
end

"""
Stores sufficient statistics for collapsed normal inverse Wishart normal (NIWN) model
"""
mutable struct NIWNState
    N::Int
    x_total::Vector{Float64}
    S_total::Matrix{Float64}
    function NIGNState(dimension::Int)
        new(0, zeros(dimension), zeros(dimension, dimension))
    end
end
function incorporate!(state::NIWNState, x::Vector{Float64})
    state.N += 1
    state.x_total += x
    state.S_total += (x * x')
end

function unincorporate!(state::NIGNState, x::Vector{Float64})
    @assert state.N > 0
    state.N -= 1
    state.x_total -= x
    state.S_total -= (x * x')
end

function posterior(prior:NIWNParams, state::NIWNState)
    n = prior.n + state.N
    k = prior.k + state.N
    mu = prior.mu + (prior.k * prior.mu + state.x_total)/k
    # S_m is the inverse of Lambda_m; we never actually store Lambda_m
    Sm = prior.S + state.S - (state.x_total * state.x_total')/(state.N)
    xdiff = (state.x_total/state.N) - prior.mu
    Sm += ((prior.k * state.N) / (prior.k + state.N)) * (xdiff * xdiff')
    return NIWNParams(mu, k, n, S)
end


