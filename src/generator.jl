##################################
# Generic generators  and traces #
##################################

abstract type Trace end

function value end
function constrain! end
function intervene! end
function propose! end

Base.delete!(t::Trace, addr) = delete!(t, (addr,))
Base.haskey(t::Trace, addr) = haskey(t, (addr,))
value(t::Trace, addr) = value(t, (addr,))
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
export constrain!
export intervene!
export propose!


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


function _delete!(t::AtomicTrace{T}) where {T}
    trace.mode = record
    trace.value = Nullable{T}()
end

function Base.delete!(t::AtomicTrace, addr)
    addr == () ? _delete!(trace, value) : atomic_addr_err(addr)
end

function value(t::AtomicTrace, addr)
    addr == () ? get(t.value) : atomic_addr_err(addr)
end

Base.haskey(t::AtomicTrace, addr) = (addr == ())

AtomicGenerator{T} = Generator{AtomicTrace{T}}

empty_trace(::AtomicGenerator{T}) where {T} = AtomicTrace(T)

export AtomicTrace
export AtomicGenerator


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
