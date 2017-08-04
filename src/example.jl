using Gen

# custom trace type

struct MotionParams
    speed::Float64
    slack::Float64
    noise::Float64
end

MotionParams() = MotionParams(NaN, NaN, NaN)

# the hidden state is the 'dist-noise'

struct MotionModelTrace{U}
    params::MotionParams
    hiddens::Vector{Float64}
    observations::Matrix{Float64}
end

function MotionModelTrace(n::Int)
    params = MotionParams()
    hiddens = Vector{Float64}(n)
    observations = Matrix{Float64}(n, 3)
    MotionModelTrace(params, hiddens, observations)
end

# custom address types
struct ParamsAddress
end

struct HiddenAddress
    i::Int
end
    
struct ObserveAddress
    i::Int
end

### queries ###

# a full sample from the prior
# NOTE: this is the query that will be used for forwrad simulation as part of a probabilistc program
# NOTE: this is a general convention---if you don't give any conditions or output, the
# the generator sample from the 'full' generative process.
struct EmptyQuery
end

struct ParamsQuery
end

struct InitialHiddenStateQuery
end

struct DynamicsQuery
    prev_i::Int
end

struct ObserveQuery
    i::Int
end


function check_query(g::HMMGenerator, output::AddressTrie, condition::AddressTrie,
                     trace::MotionModelTrace, n::Int)
    if !disjoint(output, condition)
        error("output and condition are not disjoint")
    end

    # supported queries:
    #   P() -- empty; results in full auxiliary joint forward simulation
    #   P(params)
    #   P(Z_1 | params)
    #   P(Z_{i+1} | Z_i, params)
    #   P(X_i | Z_i, params)

    output_params = false
    condition_params = false

    output_hiddens = Vector{Int}()
    output_observes =Vector{Int}()
    condition_hiddens = Vector{Int}()
    condition_observes = Vector{Int}()

    for addr in output
        if isa(addr, ParamsAddress)
            output_params = true
        elseif isa(addr, HiddenAddress)
            push!(output_hiddens, addr.i)
        elseif isa(addr, ObserveAddress)
            push!(output_observes, addr.i)
        else
            error("Invalid address: $addr")
        end
    end

    for addr in condition
        if isa(addr, ParamsAddress)
            condition_params = true
        elseif isa(addr, HiddenAddress)
            push!(condition_hiddens, addr.i)
        elseif isa(addr, ObserveAddress)
            push!(condition_observes, addr.i)
        else
            error("Invalid address: $addr")
        end
    end

    if (length(condition_hiddens) > n ||
        length(output_hiddens) > n ||
        length(condition_observes) > n ||
        length(output_observes) > n)
        error("Invalid query")
    end

    if (output_params && 
        length(output_hiddens) == 0 &&
        length(output_observes) == 0 &&
        length(condition_hiddens) == 0 &&
        length(condition_observes) == 0)

        # P(params)
        return ParamsQuery()
    end

    if condition_params

        if (length(output_hiddens) == 1 &&
            output_hiddens[1] == 1 &&
            length(output_observes) == 0 &&
            length(condition_hiddens) == 0 &&
            length(condition_observes) == 0)

            # P(Z1 | params)
            return InitialHiddenStateQuery()
        end

        if (length(output_hiddens) == 1 &&
            length(output_observes) == 0 &&
            length(condition_hiddens) == 1 &&
            length(condition_observes) == 0 &&
            output_hiddens[1] == condition_hiddens[1] + 1)

            # P(Z_{i+1} | Z_i, params)
            return DynamicsQuery(output_hiddens[1])
        end

        if (length(output_hiddens) == 0 &&
            length(output_observes) == 1 &&
            length(condition_hiddens) == 1 &&
            length(condition_observes) == 0 &&
            output_observes[1] == condition_hiddens[1])

            # P(X_i | Z_i, params)
            return ObserveQuery(output_observes[1])
        end
    end

    if (!output_params &&
        !condition_params &&
        length(output_hiddens) == 0 &&
        length(output_observes) == 0 &&
        length(condition_hiddens) == 0 &&
        length(condition_observes) == 0)
        return EmptyQuery()
    end
    
    error("Invalid query")
end

# custom generator
struct MotionModelGenerator 
end

ArgsType = Tuple{MotionHypers,Int,Vector{Float64}}

empty_trace(g::HMMGenerator, args::ArgsType) = MotionModelTrace(args[2])

# the return value is a deterministic function of the trace as generated.
# in our case, it is a the vector of observations.
# we copy so that the enclosing generator cannot mutate the trace
return_value(trace::MotionModelTrace) = copy(trace.observations)

struct MotionHypers
    speed_mu::Float64
    speed_std::Float64
    slack_alpha::Float64
    slack_beta::Float64
    noise_alpha::Float64
    noise_beta::Float64
end

function _simulate!(query::ParamsQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    # TODO sample and return the probability of the parameters in the trace
end

function _simulate!(query::InitialHiddenStateQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    # TODO sample and return the probability of the initial state in the trace given the params
end

function _simulate!(query::DynamicsQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    # TODO sample and return the probability of the given state transition in the trace
end

function _simulate!(query::ObserveQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    # TODO sample and return the probability of a given observation
end

function _simulate!(query::EmptyQuery, args::ArgsType, trace::MotionModelTrace)
    # sample the entire process and populate the trace
    # NOTE: everything we are sampling is an auxiliary variable.
    # a generator is free to sample all the auxiliary variables it wants, for a given query.

    # sample parameters
    _simulate!(ParamsQuery(), args, trace)

    # sample initial hidden and observed states
    _simulate!(InitialHiddenStateQuery(), args, trace)
    _simulate!(ObserveQuery(1), args, trace)

    # sample remaining hidden and observed states
    for i=2:n
        _simulate!(DynamicsQuery(i-1), args, trace)
        _simulate!(ObserveQuery(i), args, trace)
    end

    # score is zero because there were no outputs
    0.
end

# TODO need to know the entire history of hidden states, not just the most recent one...

function simulate!(g::HMMGenerator{T,U}, args::ArgsType, output::AddressTrie, condition::AddressTrie, trace::MotionModelTrace)
    (hypers, n, times) = args
    query = check_query(output, condition, trace, n)
    score = _simulate!(query, hypers, n, trace)
    (score, return_value(trace))
end

function _regenerate(query::ParamsQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    l = logpdf(normal, trace.params.speed, hypers.speed_mu, hypers.speed_std)
    l += logpdf(Gen.gamma, trace.params.slack, hypers.slack_alpha, hypers.slack_beta)
    l += logpdf(Gen.gamma, trace.params.noise, hypers.noise_alpha, hypers.noise_beta)
    l
end

function _regenerate(query::InitialHiddenStateQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    logpdf(normal, trace.hiddens[1], 0., trace.params.slack)
end

function _regenerate(query::DynamicsQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    prev_state = trace.hiddens[query.i]
    next_state = trace.hiddens[query.i+1]
    dt = times[query.i+1] - times[query.i]
    logpdf(normal, next_state - prev_state, dt * trace.params.speed, trace.params.slack)
end

function _regenerate(query::ObserveQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    logpdf(mvnormal, trace.observations, noise * eye(3))
end

function _regenerate(query::EmptyQuery, args::ArgsType, trace::MotionModelTrace)
    (hypers, n, times) = args
    # for the EmptyQuery, simulate and regenreate are identical
    # both use rsimulation for the auxiliary variables, and return a score of 0.
    simulate!(query, hypers, n, trace)
end



function regenerate!(g::HMMGenerator{T,U}, args::ArgsType, output::AddressTrie, condition::AddressTrie, trace::MotionModelTrace)
    query = check_query(output, condition, trace, n)
    score = regenerate(query, args, trace)
    (score, return_value(trace))
end








