##########################
# Probabilistic programs #
##########################

# TODO: add back in AD
# TODO: add back in aliases

## Trace types for probabilistic programs

"""
Trace of a `ProbabilisticProgram`.

"""
mutable struct ProgramTrace <: Trace

    # key is a single element of an address (called `addr_head` in the code)
    subtraces::Dict{Any, Trace}

    # gets reset to 0. after each call to generate!
    score::GenScalar

    # the return value, with address ()
    # NOTE: the return value is an auxiliary variable, that cannot be included in 'output' or 'condition'
    # TODO: maybe we should include it in 'condition'
    return_value
end

function ProgramTrace()
    subtraces = Dict{Any, Trace}()
    return_value = nothing
    score = 0.
    ProgramTrace(subtraces, score, return_value)
end

# serialization
import JSON
function JSON.lower(trace::ProgramTrace)
    trace.subtraces
end

"""
Retrieve the value recorded at a given address.
"""
function Base.getindex(t::ProgramTrace, addr::Tuple)
    if addr == ()
        return t.return_value
    end
    local subtrace::Trace
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        subtrace = t.subtraces[addrhead]
    else
        error("address not found: $addr")
    end
    value(subtrace, addr[2:end])
end


"""
Delete an address from the trace
"""
function Base.delete!(t::ProgramTrace, addr::Tuple)
    if addr == ()
        t.return_value = nothing
    end
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        if length(addr) == 1
            # delete the subtrace completely
            delete!(t.subtraces, addrhead)
        else
            subtrace = t.subtraces[addrhead]
            delete!(subtrace, addr[2:end])
        end
    end
end


"""
Check if value exists for a given address.
"""
function Base.haskey(t::ProgramTrace, addr::Tuple)
    if addr == ()
        return t.return_value != nothing
    end
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        subtrace = t.subtraces[addrhead]
        haskey(subtrace, addr[2:end])
    else
        return false
    end
end


"""
Check if a subtrace exists at a given address prefix.
"""
has_subtrace(t::Trace, addr_head) = haskey(t.subtraces, addr_head)


"""
Retrieve a subtrace at a given address prefix.
"""
get_subtrace(t::Trace, addr_head) = t.subtraces[addr_head]


"""
Set the subtrace at a given address prefix.
"""
set_subtrace!(t::Trace, addr_head, subtrace::Trace) = begin t.subtraces[addr_head] = subtrace end



function Base.print(io::IO, trace::ProgramTrace)
    # TODO make a nice table representaiton, and sort the keys
    println(io, "Trace(")
    indent = "  "
    for (addrhead, subtrace) in trace.subtraces
        subtrace_str = isa(subtrace, AtomicTrace) ? "$subtrace" : "$(typeof(subtrace))"
        print(io, "$addrhead\t$subtrace_str\n")
    end
    println(io, ")")
end

function finalize!(t::ProgramTrace)
    previous_score = t.score
    t.score = 0.
    previous_score
end


"""
A generative process represented by a Julia function and constructed with `@program`.

    ProbabilisticProgram <: Generator{ProgramTrace}
"""
struct ProbabilisticProgram <: Generator{ProgramTrace}
    program::Function
end

# TOOD pass args in
empty_trace(::ProbabilisticProgram) = ProgramTrace()

# this symbol is passed as the first argument to every probabilistic program
# invocation, and each @g and @e macro expands into a function call on the trace
const trace_symbol = gensym()
const output_symbol = gensym()
const condition_symbol = gensym()
const execution_mode_symbol = gensym()

"""
Annotate an invocation of a `Generator` within a `@program` with an address.

The address should uniquely identify this point in the dynamic execution of the program.
The program can process `constrain!`, `propose!`, and `intervene!` requests for this address that are present in the trace passed to `generate!`.
"""
macro g(expr, addr)
    # NOTE: the purpose of this macro is the same as the purpose of the @generate! macro:
    # to allow use of function call syntax generator(args...) while tracing
    if expr.head == :call
        generator = expr.args[1]
        generator_args = expr.args[2:end]
        Expr(:call, 
            tagged!,
            esc(trace_symbol),
            esc(output_symbol),
            esc(condition_symbol),
            esc(execution_mode_symbol),
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(addr))

    else
        error("invalid application of @g, it is only used to address generator invocations")
    end
end

"""
Annotate an arbitrary expression within a `@program` with an address.

The program can process `intervene!` requests for this address that are present in the trace passed to `generate!`.
"""
macro e(expr, addr)
    # TODO?
    Expr(:call, tagged!, esc(trace_symbol), esc(expr), esc(addr))
end

function tagged!(t::ProgramTrace, output::AddressTrie, condition::AddressTrie,
                 is_simulating::Bool, generator::Generator{T}, args::Tuple, addr_head) where {T}
    local subtrace::T
    if haskey(t.subtraces, addr_head)
        # check if the sub-trace is the right type.
        subtrace = t.subtraces[addr_head]
    else
        subtrace = empty_trace(generator) # TODO pass args in
    end

    # NOTE on conditions:
    # TODO: if the address is in the 'condition' set, then we need to verify that it actually can be conditioned upon
    # TODO For now, we assume that is the case---it is up to the user to verify that.

    # If we constrain the output value of an AtomicGenerator, in which case
    # the AtomicGenerator will return the contsrianed vlaue, and a score of 0.

    sub_output = output[addr_head]
    sub_condition = condition[addr_head]
    if is_simulating
        (score, val) = simulate!(generator, args, sub_output, sub_condition, subtrace)
    else
        (score, val) = regenerate!(generator, args, sub_output, sub_condition, subtrace)
    end
    t.score += score
    t.subtraces[addr_head] = subtrace
    @assert value(subtrace, ()) == val

    val
end

# process a generic tagged value 
function tagged!(trace::ProgramTrace, val, addr_head)
    # NOTE: it's not necessary to create an atomic trace here
    # this can be optimized out
    local subtrace::AtomicTrace
    if haskey(trace.subtraces, addr_head)
        subtrace = trace.subtraces[addr_head]
        if subtrace.mode == record
            subtrace.value = val
        elseif subtrace.mode == constrain || subtrace.mode == propose
            error("cannot constrain or propose a non-generator invocation at $addr_head")
        end
        # if the mode is intervene, then don't change the value in the subtrace
    else
        subtrace = AtomicTrace(val)
    end
    trace.subtraces[addr_head] = subtrace
    value(subtrace, ())
end


"""
Define a probabilisic program.

The body of the program is just the body of a regular Julia function, except that the annotation macros [`@g`](@ref) and [`@e`](@ref) can be used.
"""
macro program(args, body)

    # first argument is the trace
    new_args = Any[
        :($trace_symbol::ProgramTrace),
        :($output_symbol::AddressTrie),
        :($condition_symbol::AddressTrie),
        :($execution_mode_symbol::Bool)
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

function regenerate!(p::ProbabilisticProgram, args::Tuple, output::AddressTrie,
                     condition::AddressTrie, trace::ProgramTrace)
    # first argument to the program is the 'is_simulating::Bool'
    val = p.program(false, trace, output, condition, args...)
    score = finalize!(trace)
    # NOTE: intervention on the return value does not modify the procedure by
    # which the score is computed. semantics: the probabilistic model is still
    # running, it is just disconnected from the output by the intervention.
    # constraints or proposals to random choices within the program will still
    # be scored
    (score, val)
end

function simulate!(p::ProbabilisticProgram, args::Tuple, output::AddressTrie,
                     condition::AddressTrie, trace::ProgramTrace)
     # TODO
end


# TODO make this true for all generators:
(p::ProbabilisticProgram)(args...) = generate!(p, args, ProgramTrace())[2]

export ProbabilisticProgram
export ProgramTrace
export @program
export @g
export @e
export has_subtrace
export get_subtrace
export set_subtrace!
export tagged!
