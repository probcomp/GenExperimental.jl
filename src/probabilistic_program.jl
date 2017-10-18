##########################
# Probabilistic programs #
##########################

mutable struct Score
    value::Float64
    gen_value::Nullable{GenScalar}
end

Score() = Score(0., Nullable())

function increment!(score::Score, increment::Real)
    score.value += increment
    if !isnull(score.gen_value)
        score.gen_value
    end
end

function increment!(score::Score, increment::GenScalar)
    score.value += concrete(increment)
    if isnull(score.gen_value)
        score.gen_value = score.value + increment
    else
        score.gen_value = get(score.gen_value) + increment
    end
end

function Base.get(score::Score)
    if !isnull(score.gen_value)
        return get(score.gen_value)
    else
        return score.value
    end
end


"""
A generative process represented by a Julia function and constructed with `@program`.

    ProbabilisticProgram <: Generator{DictTrace}

The output address `()` can never be included in the outputs in a query.

The output address `()` may be included in the conditions, provided that the outputs is empty, and that there are no other conditions.
In this case, the auxiliary structure is empty, the score is zero, and the returned value is the value given in the trace.
"""
struct ProbabilisticProgram <: Generator{DictTrace}
    program::Function
end

# TOOD pass args in
empty_trace(::ProbabilisticProgram) = DictTrace()

# these symbols are passed as the first arguments to every probabilistic program
# TODO put these into one data structure, called the RuntimeState?
const trace_symbol = gensym()
const output_symbol = gensym()
const condition_symbol = gensym()
const method_symbol = gensym()
const score_symbol = gensym()
const aliases_symbol = gensym()

"""
Annotate an invocation of a `Generator` within a `@program` with an address.

Example:

    result = @g(generator(generator_args...), addr_first)

The address element `addr_first` should uniquely identify this point in the dynamic execution of the program.
"""
macro g(expr, addr_first)
    # NOTE: This macro is separate from @e because it uses the special function call syntax for generators:
    if expr.head == :call
        generator = expr.args[1]
        generator_args = expr.args[2:end]
        Expr(:call, 
            tagged!,
            esc(trace_symbol),
            esc(output_symbol),
            esc(condition_symbol),
            esc(method_symbol),
            esc(score_symbol),
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(addr_first))
    else
        error("Invalid application of @g, it is only used to address generator invocations")
    end
end

"""
Annotate an arbitrary expression within a `@program` with an address.

The program can process `intervene!` requests for this address that are present in the trace passed to `generate!`.
"""
macro e(expr, addr_first)
    Expr(:call, tagged!, esc(trace_symbol), esc(expr), esc(addr_first), esc(condition_symbol))
end

"""
BEGIN ALIASING
# TODO
"""

function add_alias!(aliases::Dict, exposed_addr, inner_addr::Tuple)
    if len(inner_addr) != 2
        error("inner address of alias must be a two-element tuple")
    end
    if !haskey(aliases, inner_addr[1])
        aliases[inner_addr[1]] = Dict()
    end
    aliases[inner_addr[1]][inner_addr[2]] = exposed_addr
end

macro alias(exposed_addr, inner_addr)
    Expr(:call, add_alias!, esc(aliases_symbol), exposed_addr, inner_addr)
end

"""
END ALIASING
"""

const METHOD_SIMULATE = 1
const METHOD_REGENERATE = 2

# process a tagged generator invocation
function tagged!(t::DictTrace, outputs, conditions,
                 method::Int, score, generator::Generator{T}, args::Tuple, addr_first) where {T}
    local subtrace::T
    if haskey(t.subtraces, addr_first)
        subtrace = t.subtraces[addr_first]
    else
        subtrace = empty_trace(generator) # TODO pass args in
    end

    # TODO: if the address is in the 'condition' set, then we need to verify
    # that it actually can be conditioned upon without modifying the
    # conditional distribution of the auxiliary structure and outputs to be
    # different from forward simulation. For now, we do not verify this.

    # retrieve all outputs and conditions with the prefix addr_first
    sub_outputs = outputs[addr_first]
    sub_conditions = conditions[addr_first]

    # recursively query the tagged generator
    if method == METHOD_SIMULATE
        (increment, value) = simulate!(generator, args, sub_outputs, sub_conditions, subtrace)
    elseif method == METHOD_REGENERATE
        (increment, value) = regenerate!(generator, args, sub_outputs, sub_conditions, subtrace)
    else
        # there are no other methods
        @assert false
    end
    increment!(score, increment)
    t.subtraces[addr_first] = subtrace

    # According to the generator specification, the return value is the value at address `()`.
    @assert subtrace[()] == value

    value
end

# process a tagged expression that is not a generator invocation
function tagged!(trace::DictTrace, value, addr_first, conditions)
    local subtrace::AtomicTrace
    if haskey(trace.subtraces, addr_first)
        subtrace = trace.subtraces[addr_first]

        # if the addr_first is in conditions, use the given value
        # TODO: this is not really correct, since there may have been
        # no distribution on the expression for us to condition
        # this capability should be handled by a separate intervention feature, which should
        # be implemented as a mutation of the generator, not a condition
        if !(addr_first in conditions)
            subtrace.value = value
        end
    else
        subtrace = AtomicTrace(value)
    end
    trace.subtraces[addr_first] = subtrace
    subtrace[()]
end


"""
Define a probabilisic program.

The body of the program is just the body of a regular Julia function, except that the annotation macros [`@g`](@ref) and [`@e`](@ref) can be used.
"""
macro program(args, body)

    # first argument is the trace
    new_args = Any[
        :($trace_symbol::DictTrace),
        :($output_symbol),
        :($condition_symbol),
        :($method_symbol::Int),
        :($score_symbol),
        :($aliases_symbol::Dict) # map from subtrace address to map from address to top-level addr
    ]

    # remaining arguments are the original arguments
    local name = Nullable{Symbol}()
    if isa(args, Symbol)

        # single untyped argument
        push!(new_args, args)
    elseif args.head == :call
    
        # @program name(args...)
        name = Nullable{Symbol}(args.args[1])
        for arg in args.args[2:end]
            push!(new_args, arg)
        end
    elseif args.head == :(::)

        # single typed argument
        push!(new_args, args)
    elseif args.head == :tuple

        # multiple arguments
        for arg in args.args
            push!(new_args, arg)
        end
    else
        error("invalid @program")
    end
    arg_tuple = Expr(:tuple, new_args...)

    generator_expr = Expr(:call, ProbabilisticProgram, 
                        Expr(:function, esc(arg_tuple), esc(body)))
    if isnull(name)
        generator_expr
    else
        generator_symbol = Base.get(name)
        Expr(Symbol("="), esc(generator_symbol), generator_expr)
    end
end

function _simulate_or_regenerate!(p::ProbabilisticProgram, args::Tuple, outputs, conditions,
                                  trace::DictTrace, method::Int)
    if () in outputs
        error("Invalid query. () cannot be an output")
    end
    if () in conditions
        if !isempty(outputs)
            # TODO also check that there are no other conditions
            error("Invalid query. Can only condition on () if there are no outputs.")
        else
            return (0., trace[()])
        end
    else
        score = Score()
        aliases = Dict()
        value = p.program(trace, outputs, conditions, method, score, aliases, args...)
        trace[()] = value
        return (get(score), value)
    end
end

function check_addresses_present(addresses, trace::DictTrace)
    for addr in addresses
        if !haskey(trace, addr)
            error("Trace does not contain value for address $addr")
        end
    end
end

function simulate!(p::ProbabilisticProgram, args::Tuple, outputs, conditions, trace::DictTrace)
    check_addresses_present(conditions, trace)
    _simulate_or_regenerate!(p, args, outputs, conditions, trace, METHOD_SIMULATE)
end

function regenerate!(p::ProbabilisticProgram, args::Tuple, outputs, conditions, trace::DictTrace)
    check_addresses_present(outputs, trace)
    check_addresses_present(conditions, trace)
    _simulate_or_regenerate!(p, args, outputs, conditions, trace, METHOD_REGENERATE)
end


# NOTE: this is not generic for all Generators, because Generator is an abstract type
# and abstract types and https://github.com/JuliaLang/julia/issues/14919
(p::ProbabilisticProgram)(args...) = simulate!(p, args, AddressTrie(), AddressTrie(), DictTrace())[2]

export ProbabilisticProgram
export @program
export @g
export @e
