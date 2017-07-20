##########################
# Probabilistic programs #
##########################

# Probabilistic programs are a type of Generator that use dictionary-based
# trace types and are constructed by writing Julia functions annotated with
# 'tags'.

## Trace types for probabilistic programs

mutable struct ProgramTrace <: Trace

    # key is a single element of an address (called `addr_head` in the code)
    subtraces::Dict{Any, Trace}

    # gets reset to 0. after each call to generate!
    score::Float64

    # the return value addressed at () (initially is nothing)
    # TODO would adding return type information to the trace constructor be useful for the compiler?
    return_value
    
    # only the return value with address () can be intervened on
    # if it is, then generate! still runs as usual and returns the same score as usual
    #, except that the return value is fixed to the intervened value
    intervened::Bool
end

ProgramTrace() = ProgramTrace(Dict{Any, Trace}(), 0., nothing, false)


"""
Constrain an address of a generator invocation to a particular value.

Whether or not the address is of a generator invocation is not will be verified
during the call to `generate!`.  A constrained address must be guaranteed to be
visited during executions of the program.

Constrains subtrace, if it already exists, otherwise records the constraint, 
which will be forwarded to the subtrace during program execution.
"""
function constrain!(t::ProgramTrace, addr::Tuple, val)
    if addr == ()
        error("cannot constrain $addr")
    end
    addrhead = addr[1]
    local subtrace::Trace
    if !haskey(t.subtraces, addrhead)
        if length(addr) == 1
            subtrace = AtomicTrace(val)
            t.subtraces[addrhead] = subtrace
        else
            error("cannot constrain $addr. there is no subtrace at $addrhead.")
        end
    else
        subtrace = t.subtraces[addrhead]
    end
    constrain!(subtrace, addr[2:end], val)
end

function intervene!(t::ProgramTrace, addr::Tuple, val)
    if addr == ()
        t.intervened = true
        t.return_value = val
        return
    end
    addrhead = addr[1]
    local subtrace::Trace
    if !haskey(t.subtraces, addrhead)
        if length(addr) == 1
            subtrace = AtomicTrace(val)
            t.subtraces[addrhead] = subtrace
        else
            error("cannot intervene $addr. there is no subtrace at $addrhead.")
        end
    else
        subtrace = t.subtraces[addrhead]
    end
    intervene!(subtrace, addr[2:end], val)
end

function propose!(t::ProgramTrace, addr::Tuple, valtype::Type)
    if addr == ()
        error("cannot propose $addr")
    end
    addrhead = addr[1]
    local subtrace::Trace
    if !haskey(t.subtraces, addrhead)
        if length(addr) == 1
            subtrace = AtomicTrace(valtype)
            t.subtraces[addrhead] = subtrace
        else
            error("cannot propose $addr. there is no subtrace at $addrhead.")
        end
    else
        subtrace = t.subtraces[addrhead]
    end
    propose!(subtrace, addr[2:end], valtype)
end

# syntactic sugars for addreses with a single element NOTE: it is importnt that
# the methods above accept all types for val the methods below can enter an
# infinite recursion if the type of val does not match the method signatures
# above above
constrain!(t::ProgramTrace, addr, val) = constrain!(t, (addr,), val)
intervene!(t::ProgramTrace, addr, val) = intervene!(t, (addr,), val)


"""
Delete an address from the trace, clearing any contraints, interventions, or
proposals applied to the address.

NOTE: Does not delete any subtraces in the trace hierarchy.
"""
function Base.delete!(t::ProgramTrace, addr::Tuple)
    if addr == ()
        t.return_value = nothing
        t.intervened = false
    end
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        subtrace = t.subtraces[addrhead]
        delete!(subtrace, addr[2:end])
    end
end

"""
Check if value exists for a given address.
"""
function Base.haskey(t::ProgramTrace, addr::Tuple)
    if addr == ()
        # TODO should be true for all subtraces make it into a generaal trace interface?
        return true
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
Retrieve the value recorded at a given address.

NOTE: use `subtrace` to retrieve a subtrace at a given address.
"""
function value(t::ProgramTrace, addr::Tuple)
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

"Return the subtrace at the given address element"
function subtrace(t::ProgramTrace, addrhead)
    # NOTE: having a subtrace is not a part of the generic Trace interface
    if haskey(t.subtraces, addrhead)
        t.subtraces[addrhead]
    else
        error("address not found: $addr")
    end
end

"
Set the subtrace and value at the given address.

Uses the existing value, if any, or `nothing` if there is none.
"
function set_subtrace!(t::ProgramTrace, addrhead, subtrace::Trace)
    t.subtraces[addrhead] = subtrace
end

# TODO introduce special syntax for accesing the subtrace (like [] for value but different)

function Base.print(io::IO, trace::ProgramTrace)
    # TODO make a nice table representaiton, and sort the keys
    println(io, "Trace(")
    indent = "  "
    for (addrhead, subtrace) in trace.subtraces
        print(io, "$addrhead\t$subtrace\n")
    end
    println(io, ")")
end

function finalize!(t::ProgramTrace)
    #if !isempty(t.remaining_to_visit)
    #    error("addresses not visited: $(t.remaining_to_visit)")
    #end
    previous_score = t.score
    t.score = 0.0
    previous_score
end


struct ProbabilisticProgram <: Generator{ProgramTrace}
    program::Function
end

empty_trace(::ProbabilisticProgram) = ProgramTrace()

function tag_generator end
function tag_expression end

macro g(expr, addr)
    # NOTE: the purpose of this macro is the same as the purpose of the @generate! macro:
    # to allow use of function call syntax generator(args...) while tracing
    if expr.head == :call
        generator = expr.args[1]
        generator_args = expr.args[2:end]
        # NOTE: tag() is defined in the macro call environment
        Expr(:call, esc(:tag_generator),
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(addr))
    else
        error("invalid application of @g, it is only used to address generator invocations")
    end
end

macro e(expr, addr)
    Expr(:call, esc(:tag_expression), esc(expr), esc(addr))
end



function tagged!(t::ProgramTrace, generator::Generator{T}, args::Tuple, addr_head) where {T}
    local subtrace::T
    if haskey(t, addr_head)
        # check if the sub-trace is the right type.
        # if it's not the right type, we need to recursively copy over all the directives.
        subtrace = t.subtraces[addr_head]
    else
        subtrace = empty_trace(generator)
    end
    # NOTE: if this was an atomic genreator and it was constrained, then value will be unchanged
    (score, val) = generate!(generator, args, subtrace)
    t.score += score
    t.subtraces[addr_head] = subtrace
    @assert value(subtrace, ()) == val
    # record it as visited
    #delete!(t.remaining_to_visit, addr_head)
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


# create a new probabilistic program
macro program(args, body)

    # generate new symbol for this execution trace
    trace_symbol = gensym()

    # first argument is the trace
    new_args = Any[:($trace_symbol::ProgramTrace)]

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

    # overload the tag function to tag values in the correct trace
    tag_overload_defs = quote

        # tagged generator invocation
        function tag_generator(gen::Generator, stuff::Tuple, name)
            $(tagged!)($trace_symbol, gen, stuff, name)
        end

        # other tagged expressions
        tag_expression(other, name) = $(tagged!)($trace_symbol, other, name)
    end
    new_body = quote $tag_overload_defs; $body end

    generator_expr = Expr(:call, ProbabilisticProgram, 
                        Expr(:function, esc(arg_tuple), esc(new_body)))
    if isnull(name)
        generator_expr
    else
        generator_symbol = Base.get(name)
        Expr(Symbol("="), esc(generator_symbol), generator_expr)
    end
end

function generate!(p::ProbabilisticProgram, args::Tuple, trace::ProgramTrace)
    val = p.program(trace, args...)
    score = finalize!(trace)
    # NOTE: intervention on the return value does not modify the procedure by
    # which the score is computed. semantics: the probabilistic model is still
    # running, it is just disconnected from the output by the intervention.
    # constraints or proposals to random choices within the program will still
    # be scored
    if trace.intervened
        val = trace.return_value
    else
        trace.return_value = val
    end
    (score, val)
end

# TODO make this true for all generators:
(p::ProbabilisticProgram)(args...) = generate!(p, args, ProgramTrace())[2]

export ProbabilisticProgram
export ProgramTrace
export @program
export @tag
export @g
export @e
export subtrace
export set_subtrace!
export tagged!
