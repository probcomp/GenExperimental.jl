######################################
# Normal Inverse Gamma Normal (NIGN) #
######################################

import Distributions

"""
Parameters for a collapsed normal inverse gamma normal (NIGN) model
"""
struct NIGNParams{T,U,V,W}

    # mean of normal distribution on mean
	m::T

    # Relative precision of mu vs data. The precision of mu is r times the
    # precision of the data.
	r::U

    # degrees of freedom for inverse gamma distribution on variance
    nu::V

    # scale parameter for inverse gamma distribution on variance
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

function NIGNState()
    NIGNState(0, 0., 0.)
end

Base.isempty(state::NIGNState) = (state.N == 0)
    
function NIGNState(values::Vector{W}) where {W <: Real}
    state = NIGNState()
    for value in values
        incorporate!(state, value)
    end
    return state
end

function incorporate!(state::NIGNState, x::Real)
    state.N += 1
    state.sum_x += x
    state.sum_x_sq += x*x
    x
end

function unincorporate!(state::NIGNState, x::Real)
    if state.N == 0
        error("Cannot unincorporate, there are no data.")
    end
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
    before = log_joint_density(state, params)
    incorporate!(state, x)
    after = log_joint_density(state, params)
    unincorporate!(state, x)
    return after - before
end

function posterior_params(state::NIGNState, params::NIGNParams)
	rn = params.r + Float64(state.N)
	nun = params.nu + Float64(state.N)
	mn = (params.r*params.m + state.sum_x)/rn
    if state.N == 0
        sn = params.s
    else
	    sn = params.s + state.sum_x_sq + params.r*params.m*params.m - rn*mn*mn
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

Base.keys(t::NIGNTrace) = keys(t.assignments)

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

function Base.getindex(t::NIGNTrace, addr::Tuple{})
    copy(t.assignments)
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

get_state(trace::NIGNTrace) = trace.state

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

function regenerate!(::NIGNGenerator{A}, args::Tuple{Any,NIGNParams,Bool}, outputs, conditions, trace::NIGNTrace{A}) where {A}
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

function simulate!(::NIGNGenerator{A}, args::Tuple{Any,NIGNParams,Bool}, outputs, conditions, trace::NIGNTrace{A}) where {A}
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
export get_state
export NIGNGenerator
