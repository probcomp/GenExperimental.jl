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

    ProbabilisticProgram <: Generator{FlatDictTrace}
"""
struct ProbabilisticProgram <: Generator{FlatDictTrace}
    program::Function
end

# TOOD pass args in??
empty_trace(::ProbabilisticProgram) = FlatDictTrace()

struct ProbabilisticProgramRuntimeState
    trace::FlatDictTrace
    outputs
    score::Score
    aliases::Dict{FlatAddress,Dict{FlatAddress, FlatAddress}}
end

function ProbabilisticProgramRuntimeState(trace::DictTrace, outputs, conditions)
    aliases = Dict{FlatAddress,Dict{FlatAddress, FlatAddress}}()
    ProbabilisticProgramRuntimeState(trace, outputs, conditions, Score(), aliases)
end

function add_alias!(state::ProbabilisticProgramRuntimeState, alias::FlatAddress, generator_addr::FlatAddress, sub_addr::FlatAddress)
    if !haskey(state.aliases, generator_addr)
        state.aliases[generator_addr] = Dict{FlatAddress, FlatAddress}()
    end
    aliases_for_subtrace = state.aliases[generator_addr]
    if haskey(aliases_for_subtrace, sub_addr)
        error("Multiple aliases given for address $((generator_addr, sub_addr))")
    end
    aliases_for_subtrace[sub_addr] = alias
end

function has_aliases(state::ProbabilisticProgramRuntimeState, generator_addr::FlatAddress)
    return haskey(state.aliases, generator_addr)
end

function get_aliases(state::ProbabilisticProgramRuntimeState, generator_addr:FlatAddress)
    return state.aliases[generator_addr]
end


# these symbols are passed as the first arguments to every probabilistic program
const runtime_state_symbol = gensym()
const method_symbol = gensym()

"""
Dynamically add an address alias.

The first argument is an address tuple.
The second argument is a single-element address.
"""

macro alias(addr, alias)
    Expr(:call, add_alias!, esc(runtime_state_symbol), esc(alias), esc(addr))
end

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
            esc(runtime_state_symbol),
            esc(method_symbol),
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
    Expr(:call, tagged!, esc(runtime_state_symbol), esc(expr), esc(addr_first))
end

# TODO use different methods..
const METHOD_SIMULATE = 1
const METHOD_REGENERATE = 2

function tagged_propose!(runtime_state::ProbabilisticProgramRuntimeState, generator::AtomicGenerator{T},
                          args::Tuple, addr::FlatAddress) where {T}
    if has_aliases(runtime_state, addr)
        error("Atomic sub-generator cannot have its addresses aliased")
    end
    (increment, subtrace) = propose!(generator, args, (ADDR_OUTPUT,))
    if addr in runtime_state.outputs
        increment!(runtime_state.score, increment)
    end
    runtime_state.trace[addr] = get(subtrace)
end

function tagged_regenerate!(runtime_state::ProbabilisticProgramRuntimeState, generator::AtomicGenerator{T},
                          args::Tuple, addr::FlatAddress) where {T}
    if has_aliases(runtime_state, addr)
        error("Atomic sub-generator cannot have its addresses aliased")
    end
    if addr in runtime_state.outputs
        increment = regenerate!(generator, args, (ADDR_OUTPUT,), AtomicTrace(runtime_state.trace[addr]))
        increment!(runtime_state.score, increment)
    else
        (_, subtrace) = propose!(generator, args, (ADDR_OUTPUT,))
        runtime_state.trace[addr] = get(subtrace)
    end
end

function tagged_propose!(runtime_state::ProbabilisticProgramRuntimeState, generator::Generator{T},
                          args::Tuple, addr::FlatAddress) where {T}
    sub_outputs = Array{FlatAddress}()
    # TODO what is the notion of return value? for a generator... is there one?
    for (sub_addr, alias) in get_aliases(runtime_state, addr)
        if alias in runtime_state.outputs
            push!(sub_outputs, sub_addr)
        end
    end
    (score, subtrace) = propose!(generator, args, sub_outputs)
    for (sub_addr, alias) in get_aliases(runtime_state, addr)
        if alias in runtime_state.outputs
            runtime_state.trace[alias] = subtrace[sub_addr]
        end
    end
    # TODO what to return????? need a notion of return value....
end

# process a tagged generator invocation
function tagged!(runtime_state::ProbabilisticProgramRuntimeState,
                 method::Int, generator::Generator{T}, args::Tuple, addr_first) where {T}

    local subtrace::T
    if has_subtrace(runtime_state.trace, addr_first)
        subtrace = get_subtrace(runtime_state.trace, addr_first)
    else
        subtrace = empty_trace(generator) # TODO pass args in??
    end

    # TODO: if the address is in the 'condition' set, then we need to verify
    # that it actually can be conditioned upon without modifying the
    # conditional distribution of the auxiliary structure and outputs to be
    # different from forward simulation. For now, we do not verify this.

    # retrieve all outputs and conditions with the prefix addr_first
    sub_outputs = runtime_state.outputs[addr_first]

    # handle any aliases for addresses under addr_first
    for (addr_rest, alias) in get_aliases(runtime_state, addr_first)
        if alias in runtime_state.outputs
            if addr_rest in sub_conditions
                error("An address and its alias $alias disagree on its role in the query")
            end
            # NOTE: this mutates the input query
            push!(sub_outputs, addr_rest)
        elseif alias in runtime_state.conditions
            if addr_rest in sub_outputs
                error("An address and its alias $alias disagree on its role in the query")
            end
            # NOTE: this mutates the input query
            push!(sub_conditions, addr_rest)
        end
        if haskey(runtime_state.trace, alias)
            subtrace[addr_rest] = runtime_state.trace[alias]
        end
    end

    # recursively query the tagged generator
    if method == METHOD_SIMULATE
        (increment, value) = simulate!(generator, args, sub_outputs, sub_conditions, subtrace)
    elseif method == METHOD_REGENERATE
        (increment, value) = regenerate!(generator, args, sub_outputs, sub_conditions, subtrace)
    else
        # there are no other methods
        @assert false
    end
    increment!(runtime_state.score, increment)
    set_subtrace!(runtime_state.trace, addr_first, subtrace)

    # copy values from subtrace to aliases
    for (addr_rest, alias) in get_aliases(runtime_state, addr_first)
        runtime_state.trace[alias] = subtrace[addr_rest]
    end

    # According to the generator specification, the return value is the value at address `()`.
    @assert subtrace[()] == value

    value
end

# process a tagged expression that is not a generator invocation
function tagged!(runtime_state::ProbabilisticProgramRuntimeState, value, addr_first)
    local subtrace::AtomicTrace
    if has_subtrace(runtime_state.trace, addr_first)
        subtrace = get_subtrace(runtime_state.trace, addr_first)

        # if the addr_first is in conditions, use the given value
        # TODO: this is not really correct, since there may have been
        # no distribution on the expression for us to condition
        # this capability should be handled by a separate intervention feature, which should
        # be implemented as a mutation of the generator, not a condition
        if !(addr_first in runtime_state.conditions)
            subtrace.value = value
        end
    else
        subtrace = AtomicTrace(value)
    end
    set_subtrace!(runtime_state.trace, addr_first, subtrace)
    subtrace[()]
end


"""
Define a probabilisic program.

The body of the program is just the body of a regular Julia function, except that the annotation macros [`@g`](@ref) and [`@e`](@ref) can be used.
"""
macro program(args, body)

    # first argument is the trace
    new_args = Any[
        :($runtime_state_symbol::ProbabilisticProgramRuntimeState),
        :($method_symbol::Int)
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
        runtime_state = ProbabilisticProgramRuntimeState(trace, outputs, conditions)
        value = p.program(runtime_state, method, args...)
        trace[()] = value
        return (get(runtime_state.score), value)
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
export ProbabilisticProgramRuntimeState
export @program
export @g
export @e
export @alias
