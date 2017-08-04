using Gen

# custom generator for HMM

struct MotionParams
    speed_mu::Float64
    speed_std::Float64
    slack_alpha::Float64
    slack_beta::Float64
    noise_alpha::Float64
    noise_beta::Float64
end

MotionParams() = MotionParams(NaN, NaN, NaN, NaN, NaN, NaN)

struct MotionModelTrace{T,U}
    params::MotionParams
    hiddens::Vector{T}
    observations::Vector{U}
end

function MotionModelTrace(T::Type, U::Type)
    params = MotionParams()
    hiddens = Vector{T}(0)
    observations = Vector{U}(0)
    MotionModelTrace(params, hiddens, observations)
end

struct HMMGenerator{T,U}
    hidden_type::Type{T}
    observation_type::Type{U}
end

empty_trace(g::HMMGenerator{T,U}) where {T,U} = MotionModelTrace(T, U)

struct ParamsAddress
end

struct HiddenAddress
    i::Int
end
    
struct ObserveAddress
    i::Int
end

# types of queries
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
                     trace::MotionModelTrace)
    if !disjoint(output, condition)
        error("output and condition are not disjoint")
    end

    # supported queries:
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

    if length(output_hiddens) > 1
        error("Invalid query")
    end

    if length(condition_hiddens) > 1
        error("Invalid query")
    end

    # cannot condition on the observations
    if length(condition_observes) > 0
        error("Invalid query")
    end

    if length(condition_hiddens) > 1
        error("Invalid query")
    end

    if output_params
        if length(output_hiddens) != 0
            error("Invalid query")
        end
        if length(output_observes) != 0
            error("Invalid query")
        end
        if length(condition_hiddens) != 0
            error("Invalid query")
        end
        if length(condition_observes) != 0
            error("Invalid query")
        end

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
    
    error("Invalid query")
end

function simulate!(g::HMMGenerator{T,U}, output::AddressTrie, condition::AddressTrie, trace::MotionModelTrace{T,U})
    query = check_query(output, condition, trace)
end

function regenerate!(g::HMMGenerator{T,U}, output::AddressTrie, condition::AddressTrie, trace::MotionModelTrace{T,U})
    query = check_query(output, condition, trace)
end
