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
    score::GenScalar

    # the return value addressed at () (initially is nothing)
    # TODO would adding return type information to the trace constructor be useful for the compiler?
    return_value
    
    # only the return value with address () can be intervened on
    # if it is, then generate! still runs as usual and returns the same score as usual
    #, except that the return value is fixed to the intervened value
    intervened::Bool

    # map from subtrace addr_head to a map from subaddres to alias
    aliases::Dict{Any, Dict{Tuple, Any}}

    tape::Tape
end

function ProgramTrace()
    subtraces = Dict{Any, Trace}()
    return_value = nothing
    intervened = false
    aliases = Dict{Any, Dict{Any, Tuple}}()
    tape = Tape()
    score = GenScalar(0., tape)
    ProgramTrace(subtraces, score, return_value, intervened, aliases, tape)
end

# serialization
import JSON
function JSON.lower(trace::ProgramTrace)
    trace.subtraces
end


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

# TODO is this general to all generators or just these programs?
function parametrize!(t::ProgramTrace, addr::Tuple, val)
    intervene!(t, addr, makeGenValue(val, t.tape))
end
parametrize!(t::ProgramTrace, addr, val) = parametrize!(t, (addr,), val)

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

function add_alias!(t::ProgramTrace, alias, addr::Tuple)
    # cannot conflict with an existing subtrace address
    if alias == ()
        error("cannot create alias $alias")
    end
    addrhead = addr[1]
    subaddr = addr[2:end]
    if !haskey(t.aliases, addrhead)
        t.aliases[addrhead] = Dict{Any, Tuple}()
    end
    if haskey(t.aliases[addrhead], subaddr)
        existing = t.aliases[addrhead][subaddr]
        error("cannot create alias $alias of address $addr because there is already an alias: $existing")
    end
    t.aliases[addrhead][subaddr] = alias
end

function alias_to_subtrace!(t::ProgramTrace, subtrace::Trace, subaddr::Tuple, alias)
    # TODO what happens if it was already constrained? The alias overwrites the constraint?
    # It depends on the subtrace implementation. We can make the subtrace specification
    # require that an error be thrown if the same address is constrained, intervened, or proposed twice
    # (without first using delete!)
    if haskey(t.subtraces, alias)
        # some directives were applied to this alias
        alias_subtrace = t.subtraces[alias]
        if alias_subtrace.mode == constrain
            constrain!(subtrace, subaddr, value(alias_subtrace, ()))
        elseif alias_subtrace.mode == intervene
            intervene!(subtrace, subaddr, value(alias_subtrace, ()))
        elseif alias_subtrace.mode == propose
            propose!(subtrace, subaddr, value_type(alias_subtrace))
        end
    end
end

function subtrace_to_alias!(t::ProgramTrace, subtrace::Trace, subaddr::Tuple, alias)
    val = value(subtrace, subaddr)
    if !haskey(t.subtraces, alias)
        t.subtraces[alias] = AtomicTrace(val)
    else
        t.subtraces[alias].value = Nullable(val)
    end
end

# TODO introduce special syntax for accesing the subtrace (like [] for value but different)

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
    if !isempty(t.aliases)
        error("not all aliases were visited")
    end
    backprop(t.score)
    previous_score = concrete(t.score)
    t.tape = Tape()
    t.score = GenScalar(0., t.tape)
    previous_score
end


struct ProbabilisticProgram <: Generator{ProgramTrace}
    program::Function
end

empty_trace(::ProbabilisticProgram) = ProgramTrace()

# this symbol is passed as the first argument to every probabilistic program
# invocation, and each @g and @e macro expands into a function call on the trace
const trace_symbol = gensym()

macro g(expr, addr)
    # NOTE: the purpose of this macro is the same as the purpose of the @generate! macro:
    # to allow use of function call syntax generator(args...) while tracing
    if expr.head == :call
        generator = expr.args[1]
        generator_args = expr.args[2:end]
        Expr(:call, 
            tagged!,
            esc(trace_symbol),
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(addr))

    else
        error("invalid application of @g, it is only used to address generator invocations")
    end
end

macro e(expr, addr)
    Expr(:call, tagged!, esc(trace_symbol), esc(expr), esc(addr))
end

macro alias(alias, addr)
        Expr(:call, 
            add_alias!,
            esc(trace_symbol),
            esc(alias),
            esc(addr))
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

    # forward any aliases
    if haskey(t.aliases, addr_head)
        for (subaddr, alias) in t.aliases[addr_head]
            alias_to_subtrace!(t, subtrace, subaddr, alias)
        end
    end

    # NOTE: if this was an atomic genreator and it was constrained, then value will be unchanged
    (score, val) = generate!(generator, args, subtrace)
    t.score += score
    t.subtraces[addr_head] = subtrace
    @assert value(subtrace, ()) == val

    # copy back data from subtrace to aliases
    if haskey(t.aliases, addr_head)
        for (subaddr, alias) in t.aliases[addr_head]
            subtrace_to_alias!(t, subtrace, subaddr, alias)
        end
        delete!(t.aliases, addr_head)
    end

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

    generator_expr = Expr(:call, ProbabilisticProgram, 
                        Expr(:function, esc(arg_tuple), esc(body)))
    if isnull(name)
        generator_expr
    else
        generator_symbol = Base.get(name)
        Expr(Symbol("="), esc(generator_symbol), generator_expr)
    end
end

function generate!(p::ProbabilisticProgram, args::Tuple, trace::ProgramTrace)
    @assert isempty(trace.aliases)
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
export @g
export @e
export @alias
export subtrace
export set_subtrace!
export tagged!
export parametrize!
