#################################
# Generic generators and traces #
#################################

"""
Record of a generative process.
"""
abstract type Trace end

"""
    value(trace::Trace, addr::Tuple)

Retrieve the value at a particular address of a trace.
"""
function value end

"""
    constrain!(trace::Trace, addr::Tuple, value)

Constrain an address of a trace to a particular value.
"""
function constrain! end

"""
    intervene!(trace::Trace, addr::Tuple, value)

Modify the behavior of a `Generator` recording to this trace by fixing the value at the given address.
"""
function intervene! end

"""
    propose!(trace::Trace, addr:Tuple, t::Type)
"""
function propose! end

"""
    release!(trace::Trace, addr::Tuple)
"""
function release! end

"""
    mode(trace::Trace, addr::Tuple)
"""
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

"""
Generative process that can record values into a `Trace`.

Each `Generator` type can record values into a paticular `Trace` type `T`.
"""
abstract type Generator{T <: Trace} end

"""
    (score, value) = generate!(generator::Generator{T}, args::Tuple, trace::T)

Record a generative process, which takes arguments, in a trace.
Return a score describing how this realization of the generative process interacted with the constraints and proposals in the trace, and the return value of the process.

There is also a macro that allows for a function-call syntax to be used with generators:

    (score, value) = @generate!(generator::Generator{T}(args::Tuple), trace::T)
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

"""
A generator that generates values for a single address.
"""
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


"""
A generator defined in terms of its sampler function (`simulate`) and its exact log density evaulator function (`logpdf`).

    simulate(g::AssessableAtomicGenerator{T}, args...)

    logpdf(g::AssessableAtomicGenerator{T}, value::T, args...)
"""
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
