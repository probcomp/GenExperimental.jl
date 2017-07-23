##################################
# Generic generators  and traces #
##################################

abstract type Trace end

function value end
function constrain! end
function intervene! end
function propose! end
function release! end
function mode end

Base.delete!(t::Trace, addr) = delete!(t, (addr,))
Base.haskey(t::Trace, addr) = haskey(t, (addr,))
value(t::Trace, addr) = value(t, (addr,))
release!(t::Trace, addr) = release!(t, (addr,))
mode(t::Trace, addr) = mode(t, (addr,))
Base.getindex(t::Trace, addr::Tuple) = value(t, addr)
Base.getindex(t::Trace, addr...) = t[addr]
# NOTE: defining generic mappings from addr to (addr,) for caused infinite looping
# when Type{val} does not match the method signature of constrain! implemented by the actual generator
propose!(t::Trace, addr, valtype::Type) = propose!(t, (addr,), valtype)

abstract type Generator{T <: Trace} end

"""
    (score, value) = generate!(generator::Geneerator{T}, trace::T)
"""
function generate! end

# subtraces can be in one of several modes:
@enum SubtraceMode record=1 propose=2 constrain=3 intervene=4

macro generate!(generator_and_args, trace)
    if generator_and_args.head == :call 
        generator = generator_and_args.args[1]
        generator_args = generator_and_args.args[2:end]
        Expr(:call, generate!,
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(trace))
    else
        error("invalid use of @generate!")
    end
end
# some generators overload generator(args) into (generator, args)
# this function allows the syntax generate!(generator(args), trace)
#function generate!(generator_and_args::Tuple{Generator,Tuple}, trace)
    #generate!(generator_and_args[1], generator_and_args[2], trace)
#end

export Generator
export generate!
export @generate!
export Trace
export SubtraceMode
export value
export mode
export constrain!
export intervene!
export propose!
export release!


#####################
# Atomic generators #
#####################

# These are generators with traces that are an atomic value (i.e. there is only
# one 'address' in the trace) these correspond to 'probabilistic modules' of
# https://arxiv.org/abs/1612.04759

mutable struct AtomicTrace{T} <: Trace
    value::Nullable{T}
    mode::SubtraceMode
end

value_type(::AtomicTrace{T}) where {T} = T

"Construct an atomic trace with a missing value"
AtomicTrace(::Type{T}) where {T} = AtomicTrace(Nullable{T}(), record)

"Construct an atomic trace with a value"
AtomicTrace(value) = AtomicTrace(Nullable(value), record)

"Get the value from an atomic trace, or an error if there is none"
Base.get(trace::AtomicTrace) = get(trace.value)

atomic_addr_err(addr::Tuple) = error("address not found: $addr; the only valid address is ()")

function Base.getindex(trace::AtomicTrace, addr::Tuple)
    addr == () ? get(trace) : atomic_addr_err(addr)
end

function _constrain!(trace::AtomicTrace{T}, value::T) where {T}
    trace.mode = constrain
    trace.value = Nullable{T}(value)
end

function constrain!(trace::AtomicTrace{T}, addr::Tuple, value::T) where {T}
    addr == () ? _constrain!(trace, value) : atomic_addr_err(addr)
end

function _intervene!(trace::AtomicTrace{T}, value::T) where {T}
    trace.mode = intervene
    trace.value = Nullable{T}(value)
end

function intervene!(trace::AtomicTrace{T}, addr::Tuple, value::T) where {T}
    addr == () ? _intervene!(trace, value) : atomic_addr_err(addr)
end

function _propose!(trace::AtomicTrace)
    trace.mode = propose
end

function propose!(trace::AtomicTrace{T}, addr::Tuple, valtype::Type{T}) where {T}
    addr == () ? _propose!(trace) : atomic_addr_err(addr)
end

function propose!(trace::AtomicTrace{T}, addr::Tuple, valtype::Type) where {T}
    error("type $valtype does match trace type $T")
end

function _release!(t::AtomicTrace)
    t.mode = record
end

function release!(t::AtomicTrace, addr::Tuple)
    addr == () ? _release!(t) : atomic_addr_err(addr)
end

function _delete!(t::AtomicTrace{T}) where {T}
    t.mode = record
    t.value = Nullable{T}()
end

function Base.delete!(t::AtomicTrace, addr)
    addr == () ? _delete!(t) : atomic_addr_err(addr)
end

value(t::AtomicTrace) = get(t.value)

function value(t::AtomicTrace, addr)
    addr == () ? get(t.value) : atomic_addr_err(addr)
end

mode(t::AtomicTrace) = t.mode

function mode(t::AtomicTrace, addr)
    addr == () ? mode(t) : atomic_addr_err(addr)
end

function Base.haskey(t::AtomicTrace, addr)
    addr == () ? !isnull(t.value) : atomic_addr_err(addr)
end

AtomicGenerator{T} = Generator{AtomicTrace{T}}

empty_trace(::AtomicGenerator{T}) where {T} = AtomicTrace(T)

function Base.print(io::IO, trace::AtomicTrace)
    prefix = if trace.mode == constrain 
        "*"
    elseif trace.mode == intervene
        "!"
    elseif trace.mode == propose
        "+"
    else
        " "
    end
    valstring = isnull(trace.value) ? "" : "$(get(trace.value))"
    print(io, "[$prefix]$(valstring)")
end

export AtomicTrace
export AtomicGenerator
export value_type


################################
# Assessable atomic generators #
################################

# These are stochastic computations whose log density can be computed:
# They should implement two methods:
#
# simulate(args...)::T
# logpdf(value::T, args...)::Any

abstract type AssessableAtomicGenerator{T} <: AtomicGenerator{T} end

function simulate end
function logpdf end

function generate!(g::AssessableAtomicGenerator{T}, args::Tuple, trace::AtomicTrace{T}) where {T}
    local value::T
    if trace.mode == intervene || trace.mode == constrain
        value = get(trace.value)
    else
        value = simulate(g, args...)
        trace.value = Nullable(value)
    end
    if trace.mode == constrain || trace.mode == propose
        score = logpdf(g, value, args...)
    else
        score = 0.
    end
    (score, value)
end

export AssessableAtomicGenerator
export simulate
export logpdf


#########################################
# Generator nested inference combinator #
#########################################

# TODO: Alias = Tuple

type PairedGenerator{T} <: Generator{T}
    p::Generator{T}
    q::Generator

    # mapping from q_address to (p_address, type)
    mapping::Dict
end

function compose(p::Generator, q::Generator, mapping::Dict)
    PairedGenerator(p, q, mapping)
end

# TODO what are the semantics of the score?
# the trace is the same type as the trace of p
# the distribution is different from that of p
# the semantics of the score are that it is an unbiased estimate of the 
function generate!(g::PairedGenerator{U}, args::Tuple, p_trace::U) where {U}
    (p_args, q_args) = args

    # populate q's trace with proposal or constraint directives
    proposed_from_q = Set()
    q_trace = empty_trace(g.q)
    for (q_addr, (p_addr, value_type)) in g.mapping
        m = haskey(p_trace, p_addr) ? mode(p_trace, p_addr) : record
        if m == constrain
            constrain!(q_trace, q_addr, p_trace[p_addr])
        elseif m == propose
            error("cannot handle proposal at address $p_addr")
        elseif m == intervene
            error("cannot handle intervention at address $p_addr")
        else
            # m == record, propose from q
            push!(proposed_from_q, q_addr)
            propose!(q_trace, q_addr, value_type)
        end
    end

    # generate q
    (q_score, _) = generate!(g.q, q_args, q_trace)

    # populate p's trace with new constraints generated from q
    for q_addr in proposed_from_q
        (p_addr, _) = g.mapping[q_addr]
        constrain!(p_trace, p_addr, q_trace[q_addr])
    end

    # generate p
    (p_score, p_retval) = generate!(g.p, p_args, p_trace)

    # release constraints on p's trace for values proposed from q
    for q_addr in proposed_from_q
        (p_addr, _) = g.mapping[q_addr]
        release!(p_trace, p_addr)
    end

    score = p_score - q_score
    (score, p_retval)
end

# shorthand combinatorifor when the proposal is an AtomicGenerator
function compose(p::Generator, q::AtomicGenerator{T}, p_addr::Tuple) where {T}
    mapping = Dict([() => (p_addr, T)])
    PairedGenerator(p, q, mapping)
end

export compose

###################################
# Generator replicator combinator #
###################################

# NOTE: this is an atomic genreator
struct ReplicatedAtomicGenerator{T} <: AtomicGenerator{T}
    inner_generator::AtomicGenerator{T}
    num::Int
end

replicate(generator::AtomicGenerator, num::Int) = ReplicatedAtomicGenerator(generator, num)

# TODO permit AD on the score?
function generate!(g::ReplicatedAtomicGenerator{T}, args::Tuple, trace::AtomicTrace{T}) where {T}
    if trace.mode == record
        (score, val) = generate!(g.inner_generator, args, trace)
        @assert score == 0.
        @assert val == value(trace)
        return (score, val)
    elseif trace.mode == intervene
        return generate!(g.inner_generator, args, trace)
    end
    scores = Vector{Float64}(g.num)
    (scores[1], _) = generate!(g.inner_generator, args, trace)
    for i=2:g.num
        supplementary_trace = AtomicTrace(T)
        constrain!(supplementary_trace, (), value(trace))
        (scores[i], _) = generate!(g.inner_generator, args, supplementary_trace)
    end
    score = logsumexp(scores) - log(g.num)
    (score, value(trace))
end

export replicate
