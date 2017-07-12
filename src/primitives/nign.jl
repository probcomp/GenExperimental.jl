# parametric to allow for AD with respect to parameters
struct NIGNParams{T,U,V,W}
	m::T
	r::U
    nu::V
    s::W
end

# sufficient statistic for NIGN
mutable struct NIGNState <: Gen.Module{Float64}
    N::Int
    sum_x::Float64
    sum_x_sq::Float64
end

NIGNState() = NIGNState(0, 0., 0.)

function incorporate!(state::NIGNState, x::Float64)
    state.N += 1
    state.sum_x += x
    state.sum_x_sq += x*x
end

function unincorporate!(state::NIGNState, x::Float64)
    @assert state.N > 0
    state.N -= 1
    state.sum_x -= x
    state.sum_x_sq -= x*x
end

function log_z(r::T, s::U, nu::V) where {T,U,V}
    lz = ((nu + 1.) / 2.) * log(2)
    lz += .5 * log(pi)
    lz -= .5 * log(r)
    lz -= (nu/2.) * log(s)
	lz += lgamma(nu/2.0)
    lz
end

function predictive_logp(x::Float64, state::NIGNState, params::NIGNParams)
	posterior_without_x = posterior_params(state, params)
	state_with_x = NIGNState(state.N+1, state.sum_x+x, state.sum_x_sq+(x*x))
	posterior_with_x = posterior_params(state_with_x, params)
	ZN = log_z(posterior_without_x.r, posterior_without_x.s, posterior_without_x.nu)
	ZM = log_z(posterior_with_x.r, posterior_with_x.s, posterior_with_x.nu)
	-.5 * log(2*pi) + ZM - ZN
end

function posterior_params(state::NIGNState, params::NIGNParams)
	rn = params.r + Float64(state.N)
	nun = params.nu + Float64(state.N)
	mn = (params.r*params.m + state.sum_x)/rn
	sn = params.s + state.sum_x_sq + params.r*params.m*params.m - rn*mn*mn
	if concrete(sn) == Float64(0)
		sn = params.s
	end
    @assert rn >= 0.
    @assert nun >= 0.
    @assert sn >= 0.
	NIGNParams(mn, rn, nun, sn)
end

function sample_parameters(params::NIGNParams)
	shape = params.nu / 2.
	scale = 2. / params.s
	# TODO sample from collapsed representation?
    rho = rand(Distributions.Gamma(shape, scale))
	mu = rand(Distributions.Normal(params.m, 1./sqrt(rho*params.r)))
	(mu, rho)
end

function joint_log_density(state::NIGNState, params::NIGNParams)
    post_params = posterior_params(state, params)
    Z0 = log_z(params.r, params.s, params.nu)
    ZN = log_z(post_params.r, post_params.s, post_params.nu)
	-(state.N/2.) * log(2 * pi) + ZN - Z0
end

# draw_nign -------------------------------------------------------

# NOTE: this module does not mutate the NIGN state


struct NIGNDraw <: Gen.Module{Float64} end

function regenerate(::NIGNDraw, x::Float64, state::NIGNState, params::NIGNParams)
	predictive_logp(x, state, params)
end

function simulate(normal::NIGNDraw, state::NIGNState, params::NIGNParams)
	post_params = posterior_params(state, params)
	(mu, rho) = sample_parameters(post_params)
	x = rand(Distributions.Normal(mu, 1./sqrt(rho)))
	(x, predictive_logp(x, state, params))
end

register_module(:draw_nign, NIGNDraw())

draw_nign(state::NIGNState, params::NIGNParams) = simulate(NIGNDraw(), state, params)[1]

export NIGNParams
export NIGNState
export NIGNDraw
export incorporate!
export unincorporate!
export joint_log_density
export posterior_params
export draw_nign
