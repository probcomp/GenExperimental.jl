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

mutable struct ExchangeableJointTrace{StateType,DrawType,ValueType} <: Trace
    # the generator that will be used for drawing new values
    # must be compatible with the StateType
    draw_generator::DrawType

    # the state that contains all constrained draws between calls to generate!
    # it is also temporarily to generate unconstrained draws inte
    state::StateType

    # values for all constrained or recorded addresses
    values::Dict{Any,ValueType}

    # which addressed are constrained
    constrained::Set{Any}
end

function Base.print(io::IO, trace::ExchangeableJointTrace{S,D,V}) where {S,D,V}
    println(io, "ExchangeableJointTrace{$S,$D,$V}(")
    indent = "  "
    for (addr, value) in trace.values
        if addr in trace.constrained
            println(io, "$indent*$addr = $value")
        else 
            println(io, "$indent $addr = $value")
        end
    end
    println("state")
    println(trace.state)
    println(io, ")")
end


function ExchangeableJointTrace{S,D,V}(::Type{S}, ::Type{D}, ::Type{V})
    ExchangeableJointTrace{S,D,V}(D(), S(), Dict{Any,V}(), Set{Any}())
end

state(trace::ExchangeableJointTrace) = trace.state

# TODO propose!, intervene!

function constrain!(trace::ExchangeableJointTrace, addr, value)
    if addr in trace.constrained 
        # it's okay to reconstrain (to a potentially different value)
        # TODO is there numerical instability introduced by repeatedly 
        # unincorporating and incorporating?
        unincorporate!(trace.state, value)
    else
        push!(trace.constrained, addr)
    end
    trace.values[addr] = value
    incorporate!(trace.state, value)
end

function delete!(trace::ExchangeableJointTrace, addr)
    if !(addr in trace.constrained)
        error("cannot unconstrain $addr, it is not constrained")
    end
    value = trace.values[addr]
    Base.delete!(trace.values, addr)
    unincorporate!(trace.state, value)
    Base.delete!(trace.constrained, addr)
end

num_constrained(trace::ExchangeableJointTrace) = length(trace.constrained)
Base.haskey(trace::ExchangeableJointTrace, addr) = haskey(trace.values, addr)
value(trace::ExchangeableJointTrace, addr) = trace.values[addr]

type ExchangeableJointGenerator{T} <: Generator{T} end

# samples new draws from the conditional distribution
# score is the marginal probability of the constrained choices
function generate!(::ExchangeableJointGenerator, args::Tuple{Set, Tuple}, trace::ExchangeableJointTrace{S,D,V}) where {S,D,V}

    # the set of new addresses to generate at
    # NOTE: the order of the set is undefined?
    # we sort it 
    addrs = args[1]
    
    # the parameters
    params = args[2]

    # the score is the log marginal probability of the constrained values
    score = logpdf(trace.state, params...)

    # TODO can we combine the constrained and non-constrained state into one and just roll-back
    # the unconstraints after regenerate?

    # generate requested values
    new_values = Dict{Any,V}()
    for addr in addrs
        if !(addr in trace.constrained)
            # NOTE unconstrained draws are not persisted in the state between
            # calls to generate!  but they are persisted in trace.values
            if haskey(trace.values, addr)
                Base.delete!(trace.values, addr)
            end
            value = simulate(trace.draw_generator, trace.state, params...)
            incorporate!(trace.state, value)
            trace.values[addr] = value
        end
        new_values[addr] = trace.values[addr]
    end

    # unincorporate all unconstrained requested values
    # TODO: constantly adding and removing values in each call to generate!
    # may introduce extra numerical drift
    for addr in addrs
        if !(addr in trace.constrained)
            unincorporate!(trace.state, trace.values[addr])
        end
    end
    @assert isapprox(logpdf(trace.state, params...), score, atol=1e-10)

    # return only the new values
    (score, new_values)
end

export ExchangeableJointTrace
export ExchangeableJointGenerator
export num_constrained

function make_exchangeable_generator(trace_type_name::Symbol, generator_type_name::Symbol,
    generator_args_type::Type, state_type::Type, draw_type::Type, value_type::Type)
    eval(quote
    
        # we define a custom trace type so we can get a zero-argument constructor
        # and so that its separately extensible
        struct $trace_type_name <: Trace
            trace::ExchangeableJointTrace{$state_type, $draw_type, $value_type}
        end
        constrain!(trace::$trace_type_name, addr, value) = constrain!(trace.trace, addr, value)
        Base.delete!(trace::$trace_type_name, addr) = delete!(trace.trace, addr)
        num_constrained(trace::$trace_type_name) = num_constrained(trace.trace)
        Base.haskey(trace::$trace_type_name, addr) = haskey(trace.trace, addr)
        value(trace::$trace_type_name, addr) = value(trace.trace, addr)
        Base.print(io::IO, trace::$trace_type_name) = print(io, "$($trace_type_name)(\n$(trace.trace)\n)")
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

