#############################################################
# Generic exchangeable joint generator with custom subtrace #
#############################################################

# a generator over values [a1, a2, a3, ..., an] where n is the argument
# can re-use traces for different n
# the sub-trace can be consrained at arbitrary subset of indices

# the score returned by generate! is the log joint probability of the constrained values
# the unconstrained values are sampled from the conditoinal distribution
# NOTE: this specification of generates follows the 'marginal likelihood estimate' semantics
# and not the importance-weighted semantics

# TODO handle 'propose!

mutable struct ExchangeableJointTrace{StateType,DrawType,ValueType}
    # the generator that will be used for drawing new values
    # must be compataible with the StateType
    draw_generator::DrawType

    # the state that has incorporated all constrained draws
    constrained_state::StateType
    
    # the maximum constrained index
    max_constrained::Int

    # the set of constrained indices (in [1, max_constrained])
    constrained::Set{Int} 

    # the set of unconstrained indices in [1, max_constrained)
    unconstrained::Set{Int} 

    # constrained or recorded values
    values::Dict{Int,ValueType} 
end

function ExchangeableJointTrace{S,D,V}(::Type{S}, ::Type{D}, ::Type{V})
    ExchangeableJointTrace{S,D,V}(D(), S(), 0, Set{Int}(), Set{Int}(), Dict{Int,V}())
end

function constrain!(trace::ExchangeableJointTrace, i::Int, value)
    # indices are in the range 1, 2, 3, ...
    if i < 1
        error("i=$i < 1")
    end
    if i in trace.constrained
        error("cannot constrain $i, it is already constrained")
    end
    trace.values[i] = value
    incorporate!(trace.constrained_state, value)
    push!(trace.constrained, i)
    if i > trace.max_constrained
        for j=trace.max_constrained+1:i-1
            push!(trace.unconstrained, j)
        end
        trace.max_constrained = i
    end
end

function unconstrain!(trace::ExchangeableJointTrace, i::Int)
    if !(i in trace.constrained)
        error("cannot unconstrain $i, it is not constrained")
    end
    unincorporate!(trace.constrained_state, values[i])
    delete!(trace.constrained, i)
    push!(trace.unconstrained, i)
end

hasvalue(trace::ExchangeableJointTrace, i::Int) = haskey(trace.values, i)
value(trace::ExchangeableJointTrace, i::Int) = trace.values[i]

type ExchangeableJointGenerator{T} <: Generator{T <: ExchangeableJointTrace} end

# samples new draws from the conditional distribution
# score is the marginal probability of the constrained choices
function draw_and_incorporate!(trace::ExchangeableJointTrace, i::Integer, params::Tuple)
    value = simulate(trace.draw_generator, trace.constrained_state, params...)
    incorporate!(trace.constrained_state, value)
    trace.values[i] = value
end

function generate!(::ExchangeableJointGenerator, args::Tuple{Int,Tuple}, trace::ExchangeableJointTrace)
    max_index = args[1]
    params = args[2]

    # can't constrain addresses that aren't in the address space
    if trace.max_constrained > max_index
        error("max_constrained=$(trace.max_constrained) is greater than max_index=$max_index")
    end

    # the score is the log marginal probability of the constrained values
    score = logpdf(trace.constrained_state, params...)

    # sample unconstrained values from the conditional distributio given
    # constrained values
    new_unconstrained = Set{Int}()
    for i in trace.unconstrained
        if i <= max_index
            draw_and_incorporate!(trace, i, params)
            push!(new_unconstrained, i)
        else
            delete!(trace.unconstrained, i)
            if haskey(tace.values, i)
                delete!(trace.values, i)
            end
        end
    end
    for i=trace.max_constrained+1:max_index
        draw_and_incorporate!(trace, i, params)
        push!(new_unconstrained, i)
        push!(trace.unconstrained, i)
    end
    
    # remove sufficient statistics for unconstrained 
    for i in new_unconstrained
        @assert i in trace.unconstrained
        unincorporate!(trace.constrained_state, trace.values[i])
    end
    
    score
end

export ExchangeableJointTrace
export ExchangeableJointGenerator

function make_exchangeable_generator(trace_type_name::Symbol, generator_type_name::Symbol,
    generator_args_type::Type, state_type::Type, draw_type::Type, value_type::Type)
    eval(quote
        $trace_type_name() = ExchangeableJointTrace($state_type, $draw_type, $value_type)

        struct $generator_type_name <: Generator{ExchangeableJointTrace{$state_type, $draw_type, $value_type}}
        end

        function generate!(::$generator_type_name, args::Tuple{Int,$generator_args_type},
                           trace::ExchangeableJointTrace{$state_type, $draw_type, $value_type})
            generator = ExchangeableJointGenerator{ExchangeableJointTrace{$state_type, $draw_type, $value_type}}()
            generate!(generator, (args[1], (args[2],)), trace)
        end

        export $trace_type_name
        export $generator_type_name
    end)
end
