
#################################################
# Generic flat trace type based on a dictionary #
#################################################

struct FlatDictTrace <: Trace
    dict::Dict{FlatAddress,Any}
end

function FlatDictTrace()
    FlatDictTrace(Dict{FlatAddress,Any}())
end

Base.haskey(t::FlatDictTrace, addr::FlatAddress) = haskey(t.dict, addr)

Base.delete!(t::FlatDictTrace, addr::FlatAddress) = delete!(t.dict, addr)

Base.getindex(t::FlatDictTrace, addr::FlatAddress) = t.dict[addr]

function Base.setindex!(t::FlatDictTrace, value, addr::FlatAddress)
    t.dict[addr] = value
end

export FlatDictTrace

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

    ProbabilisticProgram <: Generator{FlatDictTrace,Any}
"""
struct ProbabilisticProgram <: Generator{FlatDictTrace,Any} # TODO give them a return type?
    program::Function
end

empty_trace(::ProbabilisticProgram) = FlatDictTrace()

abstract type ProbabilisticProgramRuntimeState end

struct SimulateProgramRuntimeState <: ProbabilisticProgramRuntimeState
    trace::FlatDictTrace
    outputs
    score::Score
    aliases::Dict{FlatAddress,Dict{FlatAddress, FlatAddress}}
end

struct AssessProgramRuntimeState <: ProbabilisticProgramRuntimeState
    trace::FlatDictTrace
    outputs
    score::Score
    aliases::Dict{FlatAddress,Dict{FlatAddress, FlatAddress}}
end

function SimulateProgramRuntimeState(trace::FlatDictTrace, outputs) 
    aliases = Dict{FlatAddress,Dict{FlatAddress, FlatAddress}}()
    SimulateProgramRuntimeState(trace, outputs, Score(), aliases)
end

function AssessProgramRuntimeState(trace::FlatDictTrace, outputs)
    aliases = Dict{FlatAddress,Dict{FlatAddress, FlatAddress}}()
    AssessProgramRuntimeState(trace, outputs, Score(), aliases)
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

function get_aliases(state::ProbabilisticProgramRuntimeState, generator_addr::FlatAddress)
    return state.aliases[generator_addr]
end

const runtime_state_symbol = gensym()

"""
Dynamically add an address alias.

The first argument is an address tuple.
The second argument is a single-element address.
"""

macro alias(addr, sub_addr, alias)
    Expr(:call, add_alias!, esc(runtime_state_symbol), esc(alias), esc(addr), esc(sub_addr))
end

"""
Annotate an invocation of a `Generator` within a `@program` with an address.

Example:

    result = @g(generator(generator_args...), addr)

The address `addr` should uniquely identify this point in the dynamic execution of the program.
"""
macro g(expr, addr)
    if expr.head == :call
        generator = expr.args[1]
        generator_args = expr.args[2:end]
        Expr(:call, 
            process_tagged,
            esc(runtime_state_symbol),
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(addr))
    else
        error("Invalid use of @g, it is only used to tag a generator invocation")
    end
end

function process_tagged(runtime_state::SimulateProgramRuntimeState, generator::AtomicGenerator{T}, args::Tuple, addr::FlatAddress) where {T}
    if has_aliases(runtime_state, addr)
        error("Atomic sub-generator cannot have its addresses aliased")
    end
    (increment, value, _) = simulate(generator, args, (ADDR_OUTPUT,))
    if addr in runtime_state.outputs
        increment!(runtime_state.score, increment)
    end
    runtime_state.trace[addr] = value
    return value
end

function process_tagged(runtime_state::AssessProgramRuntimeState, generator::AtomicGenerator{T}, args::Tuple, addr::FlatAddress) where {T}
    if has_aliases(runtime_state, addr)
        error("Atomic sub-generator cannot have its addresses aliased")
    end
    local value::T
    if addr in runtime_state.outputs
        value = runtime_state.trace[addr]
        (increment, _) = assess!(generator, args, (ADDR_OUTPUT,), AtomicTrace(value))
        increment!(runtime_state.score, increment)
    else
        (increment, value) = assess!(generator, args, (), AtomicTrace(T))
        @assert increment == 0.
        runtime_state.trace[addr] = value
    end
    return value
end

function get_outputs(runtime_state::ProbabilisticProgramRuntimeState, addr::FlatAddress)
    sub_outputs = Vector{FlatAddress}()
    if has_aliases(runtime_state, addr)
        for (sub_addr, alias) in get_aliases(runtime_state, addr)
            if alias in runtime_state.outputs
                push!(sub_outputs, sub_addr)
            end
        end
    end
    return sub_outputs
end

function copy_subtrace_to_trace!(runtime_state::ProbabilisticProgramRuntimeState, addr::FlatAddress, subtrace)
    if has_aliases(runtime_state, addr)
        for (sub_addr, alias) in get_aliases(runtime_state, addr)
            runtime_state.trace[alias] = subtrace[sub_addr]
        end
    end
end

function process_tagged(runtime_state::SimulateProgramRuntimeState, generator::Generator{T,V}, args::Tuple, addr::FlatAddress) where {T,V}
    local value::V
    sub_outputs = get_outputs(runtime_state, addr)
    (increment, value, subtrace) = simulate(generator, args, sub_outputs)
    copy_subtrace_to_trace!(runtime_state, addr, subtrace)
    increment!(runtime_state.score, increment)
    return value
end

function initialize_subtrace(runtime_state::AssessProgramRuntimeState, addr::FlatAddress, generator::Generator)
    subtrace = empty_trace(generator)
    if has_aliases(runtime_state, addr)
        for (sub_addr, alias) in get_aliases(runtime_state, addr)
            if haskey(runtime_state.trace, alias)
                subtrace[sub_addr] = runtime_state.trace[alias]
            end
        end
    end
    return subtrace
end

function process_tagged(runtime_state::AssessProgramRuntimeState, generator::Generator{T,V}, args::Tuple, addr::FlatAddress) where {T,V}
    local value::V
    sub_outputs = get_outputs(runtime_state, addr)
    subtrace = initialize_subtrace(runtime_state, addr, generator)
    (increment, value) = assess!(generator, args, sub_outputs, subtrace)
    copy_subtrace_to_trace!(runtime_state, addr, subtrace)
    increment!(runtime_state.score, increment)
    return value
end


"""
Define a probabilisic program.

The body of the program is just the body of a regular Julia function, except that the annotation macros [`@g`](@ref) and [`@e`](@ref) can be used.
"""
macro probabilistic(expr)
    head = expr.head
    if head != :function
        error("Invalid @probabilistic function definition syntax")
    end
    args = expr.args[1]
    body = expr.args[2]
    
    # first argument is the trace
    new_args = Any[
        :($runtime_state_symbol::ProbabilisticProgramRuntimeState),
    ]

    # remaining arguments are the original arguments
    local name = Nullable{Symbol}()
    if isa(args, Symbol)

        # single untyped argument
        push!(new_args, args)
    elseif args.head == :call
    
        # @probabilistic <name>(args...)
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
        error("Invalid @probabilistic function definition syntax")
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

function check_addresses_present(addresses, trace::FlatDictTrace)
    for addr in addresses
        if !haskey(trace, addr)
            error("Trace does not contain value for address $addr")
        end
    end
end

function simulate(p::ProbabilisticProgram, args::Tuple, outputs)
    trace = FlatDictTrace()
    runtime_state = SimulateProgramRuntimeState(trace, outputs)
    value = p.program(runtime_state, args...)
    score = get(runtime_state.score)
    return (score, value, trace)
end

function assess!(p::ProbabilisticProgram, args::Tuple, outputs, trace::FlatDictTrace)
    check_addresses_present(outputs, trace)
    runtime_state = AssessProgramRuntimeState(trace, outputs)
    value = p.program(runtime_state, args...)
    score = get(runtime_state.score)
    return (score, value)
end


# NOTE: this is not generic for all Generators, because Generator is an abstract type
# and abstract types and https://github.com/JuliaLang/julia/issues/14919
(p::ProbabilisticProgram)(args...) = simulate(p, args, ())[2]

export ProbabilisticProgram
export ProbabilisticProgramRuntimeState
export @probabilistic
export @g
export @alias
