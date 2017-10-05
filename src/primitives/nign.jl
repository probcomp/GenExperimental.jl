######################################
# Normal Inverse Gamma Normal (NIGN) #
######################################

import Distributions

"""
Parameters for a collapsed normal inverse gamma normal (NIGN) model
"""
struct NIGNParams{T,U,V,W}
	m::T
	r::U
    nu::V
    s::W
end

"""
Stores sufficient statistics for collapsed normal inverse gamma normal (NIGN) model
"""
mutable struct NIGNState
    N::Int
    sum_x::Float64
    sum_x_sq::Float64
end

NIGNState() = NIGNState(0, 0., 0.)

function incorporate!(state::NIGNState, x::Float64)
    state.N += 1
    state.sum_x += x
    state.sum_x_sq += x*x
    x
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

function log_joint_density(state::NIGNState, params::NIGNParams)
    post_params = posterior_params(state, params)
    Z0 = log_z(params.r, params.s, params.nu)
    ZN = log_z(post_params.r, post_params.s, post_params.nu)
	-(state.N/2.) * log(2 * pi) + ZN - Z0
end

function draw(state::NIGNState, params::NIGNParams)
	post_params = posterior_params(state, params)
	(mu, rho) = sample_parameters(post_params)
    rand(Distributions.Normal(mu, 1./sqrt(rho)))
end

export NIGNState
export NIGNParams
export posterior_params
export log_joint_density 
export incorporate!
export unincorporate!
export draw


########################
# NIGN Joint generator #
########################

"""
Custom trace type for normal inverse gamma normal generator.

Parametrized by the type of the addresses.
"""
mutable struct NIGNTrace{A} <: Trace
    assignments::Dict{A,Float64}
    state::NIGNState
end

function NIGNTrace(::Type{A}) where {A}
    NIGNTrace(Dict{A,Float64}(), NIGNState())
end

function Base.haskey(t::NIGNTrace{A}, addr::Tuple{A}) where {A}
    haskey(t.assignments, addr[1])
end

function Base.delete!(t::NIGNTrace{A}, addr::Tuple{A}) where {A}
    unincorporate!(t.state, t.assignments[addr[1]])
    delete!(t.assignments, addr[1])
end

function Base.getindex(t::NIGNTrace{A}, addr::Tuple{A}) where {A}
    t.assignments[addr[1]]
end

function Base.setindex!(t::NIGNTrace{A}, value::Float64, addr::Tuple{A}) where {A}
    if haskey(t.assignments, addr[1])
        delete!(t, addr)
    end
    incorporate!(t.state, value)
    t.assignments[addr[1]] = value
end

Base.haskey(t::NIGNTrace{A}, addr_element::A) where {A} = haskey(t, (addr_element,))

Base.delete!(t::NIGNTrace{A}, addr_element::A) where {A} = delete!(t, (addr_element,))

Base.getindex(t::NIGNTrace{A}, addr_element::A) where {A} = t[(addr_element,)]

Base.setindex!(t::NIGNTrace{A}, value::Float64, addr_element::A) where {A} = begin t[(addr_element,)] = value end


struct NIGNGenerator{A} <: Generator{NIGNTrace{A}} end

function NIGNGenerator(::Type{A}) where {A}
    NIGNGenerator{A}()
end

function empty_trace(::NIGNGenerator{A}) where {A}
    return NIGNTrace(A)
end

function check_trace_addresses(trace::NIGNTrace, outputs, conditions, addresses)
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
end

function regenerate!(::NIGNGenerator{A}, args::Tuple{Any,NIGNParams}, outputs, conditions, trace::NIGNTrace{A}) where {A}
    (addresses, params) = args
    check_trace_addresses(trace, outputs, conditions, addresses)

    # check that output and condition addresses are in the argument 'addresses'
    for addr in (outputs..., conditions...)
        if !(length(addr) == 1 && addr[1] in addresses)
            error("address $addr was in outputs or conditions but is not in argument addresses")
        end
    end
    
    # joint probability of the conditions and outputs
    score_combined = log_joint_density(trace.state, params)
    for addr in outputs
        unincorporate!(trace.state, trace.assignments[addr[1]])
    end

    # joint probability of just the conditions
    score_condition_only = log_joint_density(trace.state, params)

    # restore the trace to its original state
    for addr in outputs
        incorporate!(trace.state, trace.assignments[addr[1]])
    end

    return (score_combined - score_condition_only, copy(trace.assignments))
end

function simulate!(::NIGNGenerator{A}, args::Tuple{Any,NIGNParams}, outputs, conditions, trace::NIGNTrace{A}) where {A}
    (addresses, params) = args
    check_trace_addresses(trace, outputs, conditions, addresses)

    # check that output and condition addresses are in the argument 'addresses'
    for addr in (outputs..., conditions...)
        if !(length(addr) == 1 && addr[1] in addresses)
            error("address $addr was in outputs or conditions but is not in argument addresses")
        end
    end

    # if outputs and conditions are both empty, then simulate all addresses in argument 'addresses'
    if isempty(outputs) && isempty(conditions)
        for addr in addresses
            value = draw(trace.state, params)
            trace.assignments[addr] = value
            incorporate!(trace.state, value)
        end
        return (0., copy(trace.assignments))
    else

        # joint probability of just the conditions
        score_condition_only = log_joint_density(trace.state, params)
    
        for addr in outputs
            value = draw(trace.state, params)
            trace.assignments[addr[1]] = value
            incorporate!(trace.state, value)
        end
    
        # joint probability of conditions and outputs
        score_combined = log_joint_density(trace.state, params)

        return (score_combined - score_condition_only, copy(trace.assignments))
    end
end

export NIGNTrace
export NIGNGenerator
