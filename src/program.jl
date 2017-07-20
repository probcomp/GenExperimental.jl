##########################
# Probabilistic programs #
##########################

# Probabilistic programs are a type of Generator that use dictionary-based
# trace types and are constructed by writing Julia functions annotated with
# 'tags'.

## Trace types for probabilistic programs

struct TraceElement
    subtrace::Trace
    value::Any
end

function Base.print(io::IO, element::TraceElement)
    if isa(element.subtrace, AtomicTrace)
        @assert get(element.subtrace) == element.value
        # Don't show the AtomicGenerator subtrace, which juts contains the value
        print(io, element.value)
    else
        print(io, "($(element.value), $(element.subtrace)")
    end
end

abstract type Directive end

struct Constraint <: Directive
    value
end

forward!(d::Constraint, subaddr::Tuple, subtrace::Trace) = constrain!(subtrace, subaddr, d.value)

struct Intervention <: Directive
    value
end

forward!(d::Intervention, subaddr::Tuple, subtrace::Trace) = intervene!(subtrace, subaddr, d.value)

struct Proposal <: Directive
end
    
forward!(d::Proposal, subaddr::Tuple, subtrace::Trace) = propose!(subtrace, subaddr)


mutable struct ProgramTrace <: Trace

    # key is a single element of an address
    elements::Dict{Any, TraceElement}

    # gets reset to 0. after each call to generate!
    score::Float64

    # a map from address head to a map from subaddress to Directive
    directives::Dict{Any, Dict{Any, Directive}}

    # a set of address elements (not complete hierarchical addresses) that must
    # be visited during a call to generate!
    # this is not mutated during generate!, but by e.g. constrain! and delete!
    to_visit::Set

    # the set of addresses elements that have yet to be visited.
    # this set must be empty upon return of generate!
    remaining_to_visit::Set
end

ProgramTrace() = ProgramTrace(Dict{Any, TraceElement}(), 0., Dict(), Set(), Set())


function add_directive!(trace::ProgramTrace, addr::Tuple, directive::Directive)
    addrhead = addr[1]
    # organize the directives by the head of the address NOTE: we don't forward
    # the directives to the subtrace immediately because the may not exist, and
    # we don't know what type they will be this may be optimized in the future
    if !haskey(trace.directives, addrhead)
        trace.directives[addrhead] = Dict()
    end
    trace.directives[addrhead][addr[2:end]] = directive
    push!(trace.to_visit, addrhead)
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
    add_directive!(t, addr, Constraint(val))
end

function intervene!(t::ProgramTrace, addr::Tuple, val)
    add_directive!(t, addr, Intervention(val))
end

function propose!(t::ProgramTrace, addr::Tuple, val)
    add_directive!(t, addr, Proposal(val))
end

"""
Delete an address from the trace and from any directives.
TODO: should these be different functoins?
"""
function Base.delete!(t::ProgramTrace, addr::Tuple)
    local element::TraceElement
    addrhead = addr[1]
    if haskey(t.elements, addrhead)
        element = t.elements[addrhead]
        if length(addr) > 1
            delete!(element.subtrace, addr[2:end])
        else
            delete!(t.elements, addrhead)
        end
    end
    if haskey(t.directives, addrhead)
        delete!(t.directives[addrhead], addr[2:end])
        if isempty(t.directives, addrhead)
            delete!(t.directives, addrhead)
            delete!(t.to_visit, addrhead)
        end
    end
end


"""
Check if value exists for a given address.
"""
function Base.haskey(t::ProgramTrace, addr::Tuple)
    local element::TraceElement
    addrhead = addr[1]
    if haskey(t.elements, addrhead)
        element = t.elements[addrhead]
        if length(addr) == 1
            return true
        else
            haskey(element.subtrace, addr[2:end])
        end
    else
        return false
    end
end

"Retrieve the value at the given address"
function value(t::ProgramTrace, addr::Tuple)
    local element::TraceElement
    addrhead = addr[1]
    if haskey(t.elements, addrhead)
        element = t.elements[addrhead]
    else
        error("address not found: $addr")
    end
    if length(addr) == 1
        element.value
    else
        value(element.subtrace, addr[2:end])
    end
end

"Return the subtrace at the given address element"
function subtrace(t::ProgramTrace, addrhead)
    # NOTE: having a subtrace is not a part of the generic Trace interface
    if haskey(t.elements, addrhead)
        t.elements[addrhead].subtrace
    else
        error("address not found: $addr")
    end
end

"
Set the subtrace and value at the given address.

Uses the existing value, if any, or `nothing` if there is none.
"
function set_subtrace!(t::ProgramTrace, addrhead, subtrace::Trace)
    value = nothing
    if haskey(t.elements, addrhead)
        value = t.elements[addrhead].value
    end
    t.elements[addrhead] = TraceElement(value, subtrace)
end

# TODO introduce special syntax for accesing the subtrace (like [] for value but different)

function Base.print(io::IO, trace::ProgramTrace)
    # TODO make a nice table representaiton, and sort the keys
    println(io, "Trace(")
    indent = "  "
    for (addrhead, element) in trace.elements
        constrained = false
        intervened = false
        proposed = false
        if addrhead in keys(trace.directives)
            for (subaddr, directive) in trace.directives[addrhead]
                if isa(directive, Constraint)
                    constrained = true
                end
                if isa(directive, Intervention)
                    intervened = true
                end
                if isa(directive, Proposal)
                    proposed = true
                end
            end
        end
        constrained_str = constrained ? "*" : " "
        intervened_str = intervened ? "!" : " "
        proposed_str = proposed ? "+" : " "
        mode_str = "$(constrained_str)$(intervened_str)$(proposed_str)"
        println(io, "$mode_str\t$addrhead\t$element")
        
    end
    println(io, ")")
end

function finalize!(t::ProgramTrace)
    if !isempty(t.remaining_to_visit)
        error("addresses not visited: $(t.remaining_to_visit)")
    end
    previous_score = t.score
    t.score = 0.0
    previous_score
end


struct ProbabilisticProgram <: Generator{ProgramTrace}
    program::Function
end

empty_trace(::ProbabilisticProgram) = ProgramTrace()

function tag end

macro tag(expr, addr)
    # NOTE: the purpose of this macro is the same as the purpose of the @generate! macro:
    # to allow use of function call syntax generator(args...) while tracing
    if expr.head == :call && haskey(primitives, expr.args[1])
        generator_type = primitives[expr.args[1]]
        generator_args = vcat(expr.args[2:end])
        # NOTE: tag() is defined in the macro call environment
        Expr(:call, esc(:tag),
            Expr(:call, generator_type),
            esc(Expr(:tuple, generator_args...)),
            esc(addr))
    else
        Expr(:call, esc(:tag), esc(expr), esc(addr))
    end
end

function tagged!(t::ProgramTrace, generator::Generator{T}, args::Tuple, addr_head) where {T}
    local subtrace::T
    if haskey(t, addr_head)
        subtrace = t[addr_head].subtrace
    else
        subtrace = empty_trace(generator)
    end
    if haskey(t.directives, addr_head)
        # forward the directives
        # in the future, this can be optimized to only be done once, not in every call to generate!
        for (subaddr, directive) in t.directives[addr_head]
            forward!(directive, subaddr, subtrace)
        end
    end
    # NOTE: if this was an atomic genreator and it was constrained, then value will be unchanged
    (score, value) = generate!(generator, args, subtrace)
    t.score += score
    t.elements[addr_head] = TraceElement(subtrace, value)
    # record it as visited
    delete!(t.remaining_to_visit, addr_head)
    value
end

# process a generic tagged value 
function tagged!(trace::ProgramTrace, value, addr_head)
    # NOTE: it's not necessary to create an atomic trace here
    # this can be optimized out
    return_value = value
    if haskey(trace.directives, addr_head)
        # process the any interventions
        for (subaddr, directive) in trace.directives[addr_head]
            if isa(directive, Intervention)
                return_value = directive.value
            else
                error("directive $directive only supported for generator invocations")
            end
        end
    end
    trace.elements[addr_head] = TraceElement(AtomicTrace(return_value), return_value)
    delete!(trace.remaining_to_visit, addr_head)
    return_value 
end


# create a new probabilistic program
macro program(args, body)
    #println("args: $args\nbody: $body")

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
    prefix = quote
        tag(gen::Generator, stuff::Tuple, name) = $(tagged!)($trace_symbol, gen, stuff, name)

        # primitives overload invocation syntax () and expand into this:
        #tag(gen_and_args::Tuple{Generator,Tuple}, name) = $(tagged!)($trace_symbol, gen_and_args[1], gen_and_args[2], name)

        # arbitrary non-generator tagged values
        tag(other, name) = $(tagged!)($trace_symbol, other, name)

        # tag((name, subname) => alias) adds an alias 
        #tag(arg::Pair) = Gen.add_alias!($trace_symbol, arg[2], arg[1][1], arg[1][2])
    end
    new_body = quote $prefix; $body end

    # evaluates to a ProbabilisticProgram struct
    if isnull(name)
        Expr(:call, ProbabilisticProgram, 
            Expr(:function, arg_tuple, new_body))
    else
        function_name = Base.get(name)
        Main.eval(quote
            $function_name = $(Expr(:call, ProbabilisticProgram,
                                Expr(:function, arg_tuple, new_body)))
        end)
    end
end

function generate!(p::ProbabilisticProgram, args::Tuple, trace::ProgramTrace)
    trace.remaining_to_visit = trace.to_visit
    value = p.program(trace, args...)
    score = finalize!(trace)
    (score, value)
end

#(p::ProbabilisticProgram)(args...) = (p, args)
(p::ProbabilisticProgram)(args...) = generate!(p, args, ProgramTrace())[2]

export ProbabilisticProgram
export ProgramTrace
export @program
export @tag
export subtrace
