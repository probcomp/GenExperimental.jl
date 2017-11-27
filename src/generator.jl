#######################
# Abstract trace type #
#######################

const FlatAddress = String

"""
Mutable associative collection indexed by addresses.

Implements the following methods:


Check if a value exists in the trace at the given address.

    Base.haskey(t::Trace, addr::FlatAddress)


Delete a value from the trace at the given address, if one exists.

    Base.delete!(t::Trace, addr::FlatAddress)


Retrieve a value from the trace at the given address, if one exists.

    Base.getindex(t::Trace, addr::FlatAddress)


Set a value at in the trace at the given address.

    Base.setindex!(t::Trace, value, addr::FlatAddress)
"""
abstract type Trace end


###########################
# Abstract generator type #
###########################


"""
Generative process that can record values into a `Trace`

Each `Generator` type can record values into a paticular `Trace` type `T`.
"""
abstract type Generator{T <: Trace} end

"""
TODO: Explain

outputs is a data structure that has method `in(outputs, addr::FlatAddress)` and length(outputs)

    (score, trace::T) = propose(generator::Generator{T}, outputs, args::Tuple)
"""
function propose! end

"""
TODO: Explain

constraints is a data structure that has method `in(outputs, addr::FlatAddress)` and length(outputs)

    score = assess!(generator::Generator{T}, constraints, args::Tuple, trace::T)

The return value is the value at the empty address `()` in the trace.
"""
function assess! end


"""
Return an empty trace that is compatible with the given generator.

    trace::T = empty_trace(generator::Generator{T})
"""
function empty_trace end



######################################
# Atomic (single address) trace type #
######################################

const ADDR_OUTPUT = ""

"""
A trace that can contains a single value, of type `T`, at address `ADDR_OUTPUT'

    AtomicTrace{T} <: Trace
"""
mutable struct AtomicTrace{T} <: Trace
    value::Nullable{T}
    AtomicTrace(value::T) where {T} = new{T}(Nullable{T}(value))
    AtomicTrace(::Type{T}) where {T} = new{T}(Nullable{T}())
end

value_type(::AtomicTrace{T}) where {T} = T

"""
Get the value from an atomic trace, or an error if there is none
"""
Base.get(trace::AtomicTrace) = get(trace.value)

atomic_addr_err(addr::FlatAddress) = error("Address not found: $addr; the only valid address is $ADDR_OUTPUT")

Base.haskey(t::AtomicTrace, addr::FlatAddress) = (ADDR_OUTPUT == addr)

function Base.delete!(t::AtomicTrace{T}, addr::FlatAddress) where {T}
    if ADDR_OUTPUT == addr
        t.value = Nullable{T}()
    else
        atomic_addr_err(addr)
    end
end

function Base.getindex(t::AtomicTrace, addr::FlatAddress)
    (ADDR_OUTPUT == addr) ? get(t) : atomic_addr_err(addr)
end

function Base.setindex!(t::AtomicTrace{T}, value::T, addr::FlatAddress) where {T}
    if ADDR_OUTPUT == addr
        t.value = Nullable(value)
    else
        atomic_addr_err(addr)
    end
end

function Base.print(io::IO, trace::AtomicTrace)
    valstring = isnull(trace.value) ? "" : "$(get(trace.value))"
    print(io, valstring)
end

###############################
# Generator for atomic traces #
###############################

"""
A generator that generates values for a single address.
"""
AtomicGenerator{T} = Generator{AtomicTrace{T}}

empty_trace(::AtomicGenerator{T}) where {T} = AtomicTrace(T)


"""
A generator defined in terms of its sampler function (`simulate`) and its exact log density evaulator function (`logpdf`).

    rand(g::AssessableAtomicGenerator{T}, args...)
    logpdf(g::AssessableAtomicGenerator{T}, value::T, args...)

The `propose!` method accepts the following queries:

    outputs: []

Returns zero as the score, and populates the trace with a value for address ADDR_OUTPUT

    outputs: [ADDR_OUTPUT]

Returns logpdf as the score (like simulate followed by assess in Venture)

The `assess!` method accepts the following queries:

    constraints: []

Returns zero as the score, and does not modify the trace (a no-op).

    constraints: [ADDR_OUTPUT]

Returns logpdf as the score (like assess in Venture)
"""
abstract type AssessableAtomicGenerator{T} <: AtomicGenerator{T} end

function logpdf end

function propose!(g::AssessableAtomicGenerator{T}, args::Tuple, outputs) where {T}
    local value::T
    has_output = (ADDR_OUTPUT in outputs)
    num_outputs = length(outputs)
    if has_output && num_outputs == 1
        value = rand(g, args...)
        score = logpdf(g, value, args...)
        trace = AtomicTrace(value)
    elseif !has_output && num_outputs == 0
        value = rand(g, args...)
        score = 0.
        trace = AtomicTrace(T)
    else
        error("Invalid query; outputs may only contain $ADDR_OUTPUT")
    end
    (score, trace)
end

function assess!(g::AssessableAtomicGenerator{T}, args::Tuple, constraints, trace::AtomicTrace{T}) where {T}
    local value::T
    has_output = (ADDR_OUTPUT in constraints)
    num_constraints= length(constraints)
    if has_output && num_constraints == 1
        value = get(trace)
        score = logpdf(g, value, args...)
    elseif !has_output && num_constraints == 0
        score = 0.
    else
        error("Invalid query; constraints may only contain $ADDR_OUTPUT")
    end
    score
end

export Trace
export Generator
export propose!
export assess!
export empty_trace
export AtomicTrace
export AtomicGenerator
export value_type
export AssessableAtomicGenerator
export logpdf
