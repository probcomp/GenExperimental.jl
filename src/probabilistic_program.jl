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


struct ProbabilisticProgramRuntimeState
    trace::DictTrace
    outputs
    conditions
    score::Score
    aliases_by_subtrace_address::Dict
    all_aliases::Set
end

function ProbabilisticProgramRuntimeState(trace::DictTrace, outputs, conditions)
    ProbabilisticProgramRuntimeState(trace, outputs, conditions, Score(), Dict(), Set())
end

function add_alias!(state::ProbabilisticProgramRuntimeState, alias, addr::Tuple)
    addr_first = addr[1]
    addr_rest = addr[2:end]
    if !haskey(state.aliases_by_subtrace_address, addr_first)
        state.aliases_by_subtrace_address[addr_first] = Dict()
    end
    aliases_for_subtrace = state.aliases_by_subtrace_address[addr_first]
    if haskey(aliases_for_subtrace, addr_rest)
        error("Multiple aliases given for address $addr")
    end
    aliases_for_subtrace[addr_rest] = alias
    push!(state.all_aliases, alias)
end

function get_aliases(state::ProbabilisticProgramRuntimeState, addr_first)
    if haskey(state.aliases_by_subtrace_address, addr_first)
        return state.aliases_by_subtrace_address[addr_first]
    else
        return ()
    end
end

function has_alias(state::ProbabilisticProgramRuntimeState, alias)
    return alias in state.all_aliases
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

const METHOD_SIMULATE = 1
const METHOD_REGENERATE = 2

# process a tagged generator invocation
function tagged!(runtime_state::ProbabilisticProgramRuntimeState,
                 method::Int, generator::Generator{T}, args::Tuple, addr_first) where {T}

    if has_alias(runtime_state, addr_first)
        error("Address $addr_first was already set as an alias of another address")
    end

    local subtrace::T
    if has_subtrace(runtime_state.trace, addr_first)
        subtrace = get_subtrace(runtime_state.trace, addr_first)
    else
        subtrace = empty_trace(generator) # TODO pass args in
    end

    # TODO: if the address is in the 'condition' set, then we need to verify
    # that it actually can be conditioned upon without modifying the
    # conditional distribution of the auxiliary structure and outputs to be
    # different from forward simulation. For now, we do not verify this.

    # retrieve all outputs and conditions with the prefix addr_first
    sub_outputs = runtime_state.outputs[addr_first]
    sub_conditions = runtime_state.conditions[addr_first]

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

    # score every atomic assessable generator
    # NOTE: HACK because it only applies to assessable atomic generators
    if isa(generator, AssessableAtomicGenerator)
        set_score!(runtime_state.trace, addr_first, logpdf(generator, subtrace[()], args...))
    end

    # copy values from subtrace to aliases
    for (addr_rest, alias) in get_aliases(runtime_state, addr_first)
        runtime_state.trace[alias] = subtrace[addr_rest]
    end

    # According to the generator specification, the return value is the value at address `()`.
    @assert isequal(subtrace[()], value)

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
