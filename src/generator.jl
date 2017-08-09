#################################
# Generic generators and traces #
#################################

"""
Mutable associative record of a generative process, which must implement the following methods:

Check if a value exists in the trace at the given address. Note that this does not check if a sub-trace exists at a given address, use `

    Base.haskey(t::Trace, addr::Tuple)


Delete a value from the trace at the given address, if one exists.

    Base.delete!(t::Trace, addr::Tuple)


Retrieve a value from the trace at the given address, if one exists.

    Base.getindex(t::Trace, addr::Tuple)


Set a value at in the trace at the given address.

    Base.setindex!(t::Trace, addr::Tuple, value)

"""
abstract type Trace end

Base.haskey(t::Trace, addr) = haskey(t, (addr,))
Base.delete!(t::Trace, addr) = delete!(t, (addr,))
Base.getindex(t::Trace, addr) = t[(addr,)]
Base.setindex!(t::Trace, addr, value) = begin t[(addr,)] = value end

"""
Generative process that can record values into a `Trace`

Each `Generator` type can record values into a paticular `Trace` type `T`.
"""
abstract type Generator{T <: Trace} end

"""
    (score, value) = simulate!(generator::Generator{T}, outputs, conditions, args::Tuple, trace::T)
"""
function simulate! end

"""
    (score, value) = regenerate!(generator::Generator{T}, outputs, conditions, args::Tuple, trace::T)
"""
function simulate! end

"""
    trace::T = empty_trace(generator::Generator{T})
"""
function empty_trace end

export Generator
export simulate!
export regenerate!
export Trace
export empty_trace


#####################
# Atomic generators #
#####################

"""
A trace that can contains a single value, of type `T`, at address `()`.

    AtomicTrace{T} <: Trace
"""
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

function _delete!(t::AtomicTrace{T}) where {T}
    t.mode = record
    t.value = Nullable{T}()
end

function Base.delete!(t::AtomicTrace, addr)
    addr == () ? _delete!(t) : atomic_addr_err(addr)
end

function Base.haskey(t::AtomicTrace, addr)
    addr == () ? !isnull(t.value) : atomic_addr_err(addr)
end

"""
A generator that generates values for a single address.
"""
AtomicGenerator{T} = Generator{AtomicTrace{T}}

empty_trace(::AtomicGenerator{T}) where {T} = AtomicTrace(T)

function Base.print(io::IO, trace::AtomicTrace)
    valstring = isnull(trace.value) ? "" : "$(get(trace.value))"
    print(io, valstring)
end

export AtomicTrace
export AtomicGenerator
export value_type


"""
A generator defined in terms of its sampler function (`simulate`) and its exact log density evaulator function (`logpdf`).

    simulate(g::AssessableAtomicGenerator{T}, args...)

    logpdf(g::AssessableAtomicGenerator{T}, value::T, args...)
"""
abstract type AssessableAtomicGenerator{T} <: AtomicGenerator{T} end

function simulate end
function logpdf end

const CONDITION_QUERY = 0
const OUTPUT_QUERY = 1
const EMPTY_QUERY = 2

function parse_query(outputs, conditions)
    output_list = [addr for addr in outputs]
    condition_list = [addr for addr in conditions]
    if length(condition_list) 1 == && condition_list[1] == ()
        return CONDITION_QUERY
    if length(output_list) 1 == && output_list[1] == ()
        return OUTPUT_QUERY
    elseif length(output_list) == 0 && length(condition_list) == 0
        return EMPTY_QUERY
    else
        error("Invalid query")
    end
end

function regenerate!(g::AssessableAtomicGenerator{T}, args::Tuple, outputs, conditions, trace::AtomicTrace{T}) where {T}
    local value::T
    query_type = parse_query(outputs, conditions)
    if query_type == CONDITION_QUERY
        # return P(nothing | value) = 1.
        value = trace[()]
        score = 0.
    elseif query_type == OUTPUT_QUERY
        value = trace[()]
        score = logpdf(g, value, args...)
    elseif query_type == EMPTY_QUERY
        value = simulate(g, args...)
        score = 0.
    end
    (score, value)
end

function simulate!(g::AssessableAtomicGenerator{T}, args::Tuple, outputs, conditions, trace::AtomicTrace{T}) where {T}
    local value::T
    query_type = parse_query(outputs, conditions)
    if query_type == CONDITION_QUERY
        # nothing ~ P(nothing | value), and return P(nothing | value) = 1.
        value = trace[()]
        score = 0.
    elseif query_type == OUTPUT_QUERY
        value = simulate(g, args...)
        score = logpdf(g, value, args...)
    elseif query_type == EMPTY_QUERY
        value = simulate(g, args...)
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

"""
A generator that returns log importance weights for its score.
"""
struct PairedGenerator{T} <: Generator{T}
    p::Generator{T}
    q::Generator

    # mapping from q_address to (p_address, type)
    mapping::Dict
end

"""
Construct a `PairedGenerator` from two generators.

The `mapping` maps addresses of `q` to tuples `(p_addr, value_type)` where `p_addr` is an address of `p` and `value_type` is the type of that address (which must be consistent between `p` and `q`).

The resulting generator inherits the trace type of `p` and address space of `p`.

The score is a log importance weight in which the target density is the unnormalized joint density of `p`, which may be constrained, and the importance distribution is `q`.
"""
function compose(p::Generator, q::Generator, mapping::Dict)
    PairedGenerator(p, q, mapping)
end

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

export PairedGenerator
export compose

###################################
# Generator replicator combinator #
###################################

"""
A generator that wraps another generator with more accurate scores.
"""
struct ReplicatedAtomicGenerator{T} <: AtomicGenerator{T}
    inner_generator::AtomicGenerator{T}
    num::Int
end

"""
Construct a `ReplicatedAtomicGenerator`.

The resulting generator inherits the trace type, address space, and sampling distribution of the `generator` argument.

The generator has the same forward sampling distribution as the innter generator, but the scores are more accurate estimates of the log density of constraints and proposals.

As `num` increases, the scores become more accurate estimates.

Score is not (yet) differentaible [#65](https://github.com/probcomp/Gen.jl/issues/65)
"""
replicate(generator::AtomicGenerator, num::Int) = ReplicatedAtomicGenerator(generator, num)

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

export ReplicatedAtomicGenerator
export replicate
