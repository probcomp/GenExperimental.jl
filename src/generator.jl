#######################
# Abstract trace type #
#######################

"""
Mutable associative collection indexed by addresses.

Implements the following methods:


Check if a value exists in the trace at the given address.

    Base.haskey(t::Trace, addr::Tuple)


Delete a value from the trace at the given address, if one exists.

    Base.delete!(t::Trace, addr::Tuple)


Retrieve a value from the trace at the given address, if one exists.

    Base.getindex(t::Trace, addr::Tuple)


Set a value at in the trace at the given address.

    Base.setindex!(t::Trace, value, addr::Tuple)
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

    (score, value) = simulate!(generator::Generator{T}, outputs, conditions, args::Tuple, trace::T)

`outputs` and `conditions` must implement `in`, `haskey`, `getindex`, and `isempty` methods that match the behavior of `AddressTrie`.

The return value is the value at the empty address `()` in the trace.
"""
function simulate! end

"""
TODO: Explain

    (score, value) = regenerate!(generator::Generator{T}, outputs, conditions, args::Tuple, trace::T)

`outputs` and `conditions` must implement `in`, `haskey`, `getindex`, and `isempty` methods that match the behavior of `AddressTrie`.

The return value is the value at the empty address `()` in the trace.
"""
function regenerate! end


"""
Return an empty trace that is compatible with the given generator.

    trace::T = empty_trace(generator::Generator{T})
"""
function empty_trace end



######################################
# Atomic (single address) trace type #
######################################

"""
A trace that can contains a single value, of type `T`, at address `()`.

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

atomic_addr_err(addr::Tuple) = error("address not found: $addr; the only valid address is ()")

function Base.haskey(t::AtomicTrace, addr::Tuple)
    addr == () ? !isnull(t.value) : atomic_addr_err(addr)
end

function Base.delete!(t::AtomicTrace{T}, addr::Tuple) where {T}
    if addr == ()
        t.value = Nullable{T}()
    else
        atomic_addr_err(addr)
    end
end

function Base.getindex(t::AtomicTrace, addr::Tuple)
    addr == () ? get(t) : atomic_addr_err(addr)
end

function Base.setindex!(t::AtomicTrace{T}, value::T, addr::Tuple) where {T}
    if addr == ()
        t.value = Nullable(value)
    else
        atomic_addr_err(addr)
    end
end

function Base.setindex!(t::AtomicTrace{T}, value, addr::Tuple) where {T}
    error("Wrong type for value, got type $(typeof(value)), expected type $T")
end

Base.haskey(t::AtomicTrace, addr_element) = haskey(t, (addr_element,))

Base.delete!(t::AtomicTrace, addr_element) = delete!(t, (addr_element,))

Base.getindex(t::AtomicTrace, addr_element) = t[(addr_element,)]

Base.setindex!(t::AtomicTrace{T}, value::T, addr_element) where {T} = begin t[(addr_element,)] = value end

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


The `simulate!` method accepts the following queries:


Sample a value for the output, and return its log density as the score, and return the sampled value.
The trace is populated with the sampled value.

    OUTPUT_QUERY
    outputs: ()
    conditions: None


Return a score of zero and the return the value given in the trace.
the value given in the trace is unmodified.

    CONDITION_QUERY
    outputs: None
    conditions: ()


Sample a value for the output, and return zero as the score, and return the sampled value.
The trace is populated with the sampled value, which is considered an auxiliary variable.

    EMPTY_QUERY
    outputs: None
    conditions: None


The `regenerate!` method accepts the following queries:


Return the log density of the value given in the input trace as the score.
The value in the trace is not modified.

    OUTPUT_QUERY
    outputs: ()
    conditions: None


Return a score of zero and the return the value given in the trace.
the value given in the trace is unmodified.
Equivalent to `simulate!` with the same arguments.

    CONDITION_QUERY
    outputs: None
    conditions: ()


Sample a value for the output, and return zero as the score, and return the sampled value.
The trace is populated with the sampled value, which is considered an auxiliary variable.
Equivalent to `simulate!` with the same arguments.

    EMPTY_QUERY
    outputs: None
    conditions: None
"""
abstract type AssessableAtomicGenerator{T} <: AtomicGenerator{T} end

function logpdf end

# outputs: ()
# conditions: None
const OUTPUT_QUERY = 1

# outputs: None
# conditions: ()
const CONDITION_QUERY = 2

# outputs: None
# conditions: None
const EMPTY_QUERY = 3

function parse_query(outputs, conditions)
    # TODO check for other addresses and error if there are any
    # Any query not one of the above forms is technically an error.
    if () in outputs && !(() in conditions)
        return OUTPUT_QUERY 
    elseif () in conditions && !(() in outputs)
        return CONDITION_QUERY
    elseif !(() in conditions) && !(() in outputs)
        return EMPTY_QUERY
    else
        error("Invalid query. The address () was contained in both outputs and conditions")
    end
end

function simulate!(g::AssessableAtomicGenerator{T}, args::Tuple, outputs, conditions, trace::AtomicTrace{T}) where {T}
    local value::T
    query_type = parse_query(outputs, conditions)
    if query_type == OUTPUT_QUERY
        value = rand(g, args...)
        trace[()] = value
        score = logpdf(g, value, args...)
    elseif query_type == CONDITION_QUERY
        value = trace[()]
        score = 0.
    elseif query_type == EMPTY_QUERY
        value = rand(g, args...)
        trace[()] = value
        score = 0.
    else
        error("Invalid query")
    end
    @assert trace[()] == value
    (score, value)
end

function regenerate!(g::AssessableAtomicGenerator{T}, args::Tuple, outputs, conditions, trace::AtomicTrace{T}) where {T}
    local value::T
    query_type = parse_query(outputs, conditions)
    if query_type == OUTPUT_QUERY
        value = trace[()]
        score = logpdf(g, value, args...)
    elseif query_type == CONDITION_QUERY
        value = trace[()]
        score = 0.
    elseif query_type == EMPTY_QUERY
        value = rand(g, args...)
        trace[()] = value
        score = 0.
    else
        error("Invalid query")
    end
    @assert trace[()] == value
    (score, value)
end



export Trace
export Generator
export simulate!
export regenerate!
export empty_trace
export AtomicTrace
export AtomicGenerator
export value_type
export AssessableAtomicGenerator
export logpdf
