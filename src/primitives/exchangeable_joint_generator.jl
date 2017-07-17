#############################################################
# Generic exchangeable joint generator with custom subtrace #
#############################################################

# a generator over arbitrary addresses, and a reserved 
# address 'next'

# takes no arguments
# the sub-trace can be consrained at arbitrary subset of indices

# the score returned by generate! is the log joint probability of the constrained values
# the unconstrained values are sampled from the conditoinal distribution
# NOTE: this specification of generates follows the 'marginal likelihood estimate' semantics
# and not the importance-weighted semantics

# TODO handle 'propose!

mutable struct ExchangeableJointTrace{StateType,DrawType,ValueType}
    # the generator that will be used for drawing new values
    # must be compatible with the StateType
    draw_generator::DrawType

    # the state that has incorporated all draws
    state::StateType

    # the state that has incorporated only constrained draws
    constrained_state::StateType
    
    # values for all constrained or recorded addresses
    values::Dict{Any,ValueType}

    # which addressed are constrained
    constrained::Set{Any}
end

function Base.print(io::IO, trace::ExchangeableJointTrace{S,D,V}) where {S,D,V}
    println(io, "ExchangeableJointTrace{$S,$D,$V}(")
    indent = "  "
    for (name, value) in trace.values
        if name in trace.constrained
            println(io, "$indent*$name = $value")
        else 
            println(io, "$indent $name = $value")
        end
    end
    println("state")
    println(trace.state)
    println("constrained state")
    println(trace.constrained_state)
    println(io, ")")
end


function ExchangeableJointTrace{S,D,V}(::Type{S}, ::Type{D}, ::Type{V})
    ExchangeableJointTrace{S,D,V}(D(), S(), S(), Dict{Any,V}(), Set{Any}())
end

constrained_state(trace::ExchangeableJointTrace) = trace.constrained_state

function constrain!(trace::ExchangeableJointTrace, name, value)
    println("trying to constrain to value = $value")
    if name in trace.constrained
        error("cannot constrain $name, it is already constrained")
    end
    if haskey(trace.values, name)
        # it was recorded
        unincorporate!(trace.state, value)
    end
    trace.values[name] = value
    incorporate!(trace.constrained_state, value)
    incorporate!(trace.state, value)
    push!(trace.constrained, name)
end

function delete!(trace::ExchangeableJointTrace, name)
    if !(name in trace.constrained)
        error("cannot unconstrain $name, it is not constrained")
    end
    value = trace.values[name]
    Base.delete!(trace.values, name)
    unincorporate!(trace.constrained_state, value)
    unincorporate!(trace.state, value)
    Base.delete!(trace.constrained, name)
end

Base.length(trace::ExchangeableJointTrace) = length(trace.values)
hasvalue(trace::ExchangeableJointTrace, name) = haskey(trace.values, name)
value(trace::ExchangeableJointTrace, name) = trace.values[name]

type ExchangeableJointGenerator{T} <: Generator{T <: ExchangeableJointTrace} end

# samples new draws from the conditional distribution
# score is the marginal probability of the constrained choices
function draw_and_incorporate!(trace::ExchangeableJointTrace, name, params::Tuple)
    value = simulate(trace.draw_generator, trace.state, params...)
    incorporate!(trace.state, value)
    incorporate!(trace.constrained_state, value)
    trace.values[name] = value
    push!(trace.constrained, name)
end

function generate!(::ExchangeableJointGenerator, args::Tuple{Set, Tuple}, trace::ExchangeableJointTrace{S,D,V}) where {S,D,V}

    # the set of new addresses to generate at
    # NOTE: this could be the total address space, but then every call to generate! would be O(N)
    # so that we could check that every constrained address is in the space.
    names = args[1]
    
    # the parameters
    params = args[2]

    # the score is the log marginal probability of the constrained values
    score = logpdf(trace.constrained_state, params...)

    # TODO can we combine the constrained and non-constrained state into one and just roll-back
    # the unconstraints after regenerate?

    # generate new values
    new_values = Dict{Any,V}()
    for name in names
        if !(name in trace.constrained)
            if haskey(trace.values, name)
                Base.delete!(trace.values, name)
                # NOTE these are not persiste din the sufficient staitics (but they are persisted in trace.values)
                #unincorporate!(trace.state, trace.values[name])
            end
            value = simulate(trace.draw_generator, trace.state, params...)
            incorporate!(trace.state, value)
            trace.values[name] = value
        end
        new_values[name] = trace.values[name]
    end

    # unincorporate all new values
    for name in names
        if !(name in trace.constrained)
            unincorporate!(trace.state, trace.values[name])
        end
    end

    # return only the new values
    (score, new_values)
end


export ExchangeableJointTrace
export ExchangeableJointGenerator

function make_exchangeable_generator(trace_type_name::Symbol, generator_type_name::Symbol,
    generator_args_type::Type, state_type::Type, draw_type::Type, value_type::Type)
    eval(quote
    
        # we define a custom trace type so we can get a zero-argument constructor
        # and so that its separately extensible
        struct $trace_type_name
            trace::ExchangeableJointTrace{$state_type, $draw_type, $value_type}
        end
        constrain!(trace::$trace_type_name, name, value) = constrain!(trace.trace, name, value)
        delete!(trace::$trace_type_name, name) = delete!(trace.trace, name)
        Base.length(trace::$trace_type_name) = length(trace.trace)
        hasvalue(trace::$trace_type_name, name) = hasvalue(trace.trace)
        value(trace::$trace_type_name, name) = value(trace.trace, name)
        Base.print(io::IO, trace::$trace_type_name) = print(io, trace.trace)
        $trace_type_name() = $trace_type_name(
            ExchangeableJointTrace($state_type, $draw_type, $value_type))

        struct $generator_type_name <: Generator{$trace_type_name} end

        function generate!(::$generator_type_name, args::$generator_args_type, trace::$trace_type_name)
            generator = ExchangeableJointGenerator{ExchangeableJointTrace{$state_type, $draw_type, $value_type}}()
            generate!(generator, (args[1], (args[2],)), trace.trace)
        end

        (::Type{$generator_type_name})(args...) = ($generator_type_name(), args)



        export $trace_type_name
        export $generator_type_name
    end)
end

