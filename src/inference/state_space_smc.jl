"""
StateSpaceSMCScheme{H} is an abstract type, that should implement the following methods:

    init(scheme::StateSpaceSMCScheme)

Returns a (state::H, score::Float64) pair

    init_score(scheme::StateSpaceSMCScheme{H}, state::H)

Returns a score (only needed for conditional SMC)

    forward(scheme::StateSpaceSMCScheme{H}, prev_state::H, t::Integer)

Returns a (state::H, score::Float64) pair

    forward(scheme::StateSpaceSMCScheme{H}, prev_state::H, state::H, t::Integer)

Returns a score (only needed for conditional SMC)

    resample(scheme::StateSpaceSMCScheme{H}, weights::Vector{float64}, t::Integer)

Returns a tuple (parents::Vector{Int}, did_resample::Bool)
Returns a Vector of integers, of length equal to the wlength of weights, drawn from the range 1:num_particles.
Note: the implementer can calculate the effective sample size inside `resample` and choose not to resapmle

    get_num_steps(scheme::StateSpaceSMCScheme{H})

Returns an integer representing the total number of steps.

    get_num_particles(scheme::StateSpaceSMCScheme{H})

Returns an integer representing the total number of states.
"""
abstract type StateSpaceSMCScheme{H} end

function init end
function init_score end
function forward end
function forward_score end
function resample end
function get_num_steps end
function get_num_particles end

struct StateSpaceSMCResult{H}
    states::Matrix{H}
    parents::Matrix{Int}
    log_weights::Vector{Float64}
    log_ml_estimate::Float64
end

function smc(scheme::StateSpaceSMCScheme{H}) where {H}
    N = get_num_particles(scheme)
    T = get_num_steps(scheme)
    states = Matrix{H}(N, T)
    parents = Matrix{Int}(N, T-1)
    log_weights = Vector{Float64}(N)
    log_ml_estimate = 0.
    for i=1:N
        (states[i, 1], log_weights[i]) = init(scheme)
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate += (log_total_weight - log(N))
    log_weights = log_weights - log_total_weight
    for t=2:T
        (parents[:, t-1], did_resample) = resample(scheme, log_weights, t)
        for i=1:N
            parent = parents[i, t-1]
            (states[i, t], log_weight) = forward(scheme, states[parent, t-1], t)
            if did_resample
                log_weights[i] = log_weight
            else
                log_weights[i] += log_weight
            end
        end
        log_total_weight = logsumexp(log_weights)
        log_ml_estimate += (log_total_weight - log(N))
        log_weights = log_weights - log_total_weight
    end
    
    StateSpaceSMCResult(states, parents, log_weights, log_ml_estimate)
end

function get_particle(result::StateSpaceSMCResult{H}, final_index::Integer) where {H}
    (N, T) = size(result.states)
    particle = Vector{H}(T)
    particle[T] = result.state[final_index, T]
    current_index = final_index
    for t=T-1:1
        current_index = result.parents[current_index, t]
        particle[t] = result.states[current_index, t]
    end
    return particle
end

function effective_sample_size(log_weights::Vector{Float64})
    # assumes weights are normalized
    log_ess = -logsumexp(2. * log_weights)
    exp(log_ess)
end

const DISTINGUISHED = 1

function conditional_smc(scheme::StateSpaceSMCScheme{H}, distinguished_particle::Vector{H}) where {H}
    N = get_num_particles(scheme)
    T = get_num_steps(scheme)
    if length(distinguished_particle) != T
        error("Expected particle length $T, actual length was $(length(distinguished_particle))")
    end
    states = Matrix{H}(N, T)
    parents = Matrix{Int}(N, T-1)
    log_weights = Vector{Float64}(N)
    log_ml_estimate = 0.

    # distinguished particle
    states[DISTINGUISHED, 1] = distinguished_particle[1]
    log_weights[DISTINGUISHED] = init_score(scheme)

    for i=2:N
        (states[i, 1], log_weights[i]) = init(scheme)
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate += (log_total_weight - log(N))
    log_weights = log_weights - log_total_weight
    for t=2:T
        (parents[:, t-1], did_resample) = resample(scheme, log_weights, t)
        
        # distinguished particle
        parents[DISTINGUISHED, t-1] = DISTINGUISHED
        states[DISTINGUISHED, t] = distinguished_particle[t]
        distinguished_log_weight = forward_score(scheme, states[DISTINGUISHED, t-1],
                                                 states[DISTINGUISHED, t], t)
        if did_resample
            log_weights[DISTINGUISHED] = distinguished_log_weight
        else
            log_weights[DISTINGUISHED] += distinguished_log_weight
        end

        for i=2:N
            parent = parents[i, t-1]
            (states[i, t], log_weight) = forward(scheme, states[parent, t-1], t)
            if did_resample
                log_weights[i] = log_weight
            else
                log_weights[i] += log_weight
            end
        end
        log_total_weight = logsumexp(log_weights)
        log_ml_estimate += (log_total_weight - log(N))
        log_weights = log_weights - log_total_weight
    end
    StateSpaceSMCResult(states, parents, log_weights, log_ml_estimate)
end

export StateSpaceSMCScheme
export StateSpaceSMCResult
export smc
export conditional_smc
export effective_sample_size
export get_particle
export init
export init_score
export forward
export forward_score
export resample
export get_num_steps
export get_num_particles
