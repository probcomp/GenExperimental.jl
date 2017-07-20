##########################
# Probabilistic programs #
##########################

# Probabilistic programs are a type of Generator that use dictionary-based
# trace types and are constructed by writing Julia functions annotated with
# 'tags'.

## Trace types for probabilistic programs

# TODO remove me:
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

#abstract type Directive end

#struct Constraint <: Directive
    #value
#end

#forward!(d::Constraint, subaddr::Tuple, subtrace::Trace) = constrain!(subtrace, subaddr, d.value)

#struct Intervention <: Directive
    #value
#end

#forward!(d::Intervention, subaddr::Tuple, subtrace::Trace) = intervene!(subtrace, subaddr, d.value)

#struct Proposal <: Directive
#end
    
#forward!(d::Proposal, subaddr::Tuple, subtrace::Trace) = propose!(subtrace, subaddr)


mutable struct ProgramTrace <: Trace

    # key is a single element of an address (called `addr_head` in the code)
    elements::Dict{Any, TraceElement}

    # gets reset to 0. after each call to generate!
    score::Float64

    # a map from address head to a map from subaddress to Directive
    #directives::Dict{Any, Dict{Any, Directive}}

    # a set of address elements (not complete hierarchical addresses) that must
    # be visited during a call to generate!
    # it is not mutated by generate!
    # it is mutated by constrain! and delete!
    #to_visit::Set

    # the set of addresses elements that have yet to be visited.
    # this set must be empty upon return of generate!
    #remaining_to_visit::Set
end

ProgramTrace() = ProgramTrace(Dict{Any, TraceElement}(), 0.)#, Dict(), Set(), Set())


#function add_directive!(trace::ProgramTrace, addr::Tuple, directive::Directive)
    #addrhead = addr[1]
    ## organize the directives by the head of the address NOTE: we don't forward
    ## the directives to the subtrace immediately because the may not exist, and
    ## we don't know what type they will be this may be optimized in the future
    #if !haskey(trace.directives, addrhead)
        #trace.directives[addrhead] = Dict()
    #end
    #trace.directives[addrhead][addr[2:end]] = directive
    ##push!(trace.to_visit, addrhead)
#end

"""
Constrain an address of a generator invocation to a particular value.

Whether or not the address is of a generator invocation is not will be verified
during the call to `generate!`.  A constrained address must be guaranteed to be
visited during executions of the program.

Constrains subtrace, if it already exists, otherwise records the constraint, 
which will be forwarded to the subtrace during program execution.
"""
function constrain!(t::ProgramTrace, addr::Tuple, val)
    addrhead = addr[1]
    if length(addr) == 1
        subtrace = AtomicTrace(val)
        constrain!(subtrace, (), val)
        # we over-write any exisitng subtrace and value at addrhead
        t.elements[addrhead] = TraceElement(subtrace, val)
    else
        if haskey(t, addrhead)
            subtrace = t.elements[addrhead].subtrace
            constrain!(subtrace, addr[2:end], val)
        else
            error("cannot constrain $addr. there is no subtrace at $addrhead.")
        end
    end
end

function intervene!(t::ProgramTrace, addr::Tuple, val)
    addrhead = addr[1]
    if length(addr) == 1
        subtrace = AtomicTrace(val)
        intervene!(subtrace, (), val)
        # we over-write any exisitng subtrace and value at addrhead
        t.elements[addrhead] = TraceElement(subtrace, val)
    else
        if haskey(t, addrhead)
            subtrace = t.elements[addrhead].subtrace
            intervene!(subtrace, addr[2:end], val)
        else
            error("cannot intervene $addr. there is no subtrace at $addrhead.")
        end
    end
    #add_directive!(t, addr, Intervention(val))
end

function propose!(t::ProgramTrace, addr::Tuple, valtype::Type)
    addrhead = addr[1]
    if length(addr) == 1
        subtrace = AtomicTrace(valtype)
        propose!(subtrace, (), valtype)
        # we over-write any exisitng subtrace and value at addrhead
        t.elements[addrhead] = TraceElement(subtrace, Nullable{valtype}())
    else
        if haskey(t, addrhead)
            subtrace = t.elements[addrhead].subtrace
            propose!(subtrace, addr[2:end], valtype) # TODO this is a change
        else
            error("cannot propose $addr. there is no subtrace at $addrhead.")
        end
    end

    #add_directive!(t, addr, Proposal())
end

# syntactic sugars for addreses with a single element NOTE: it is importnt that
# the methods above accept all types for val the methods below can enter an
# infinite recursion if the type of val does not match the method signatures
# above above
constrain!(t::ProgramTrace, addr, val) = constrain!(t, (addr,), val)
intervene!(t::ProgramTrace, addr, val) = intervene!(t, (addr,), val)


"""
Delete an address from the trace.
TODO: should these be different functoins?
"""
function Base.delete!(t::ProgramTrace, addr::Tuple)
    addrhead = addr[1]
    if haskey(t.elements, addrhead)
        element = t.elements[addrhead]
        delete!(element.subtrace, addr[2:end])
    end
    #if haskey(t.directives, addrhead)
        #delete!(t.directives[addrhead], addr[2:end])
        #if isempty(t.directives, addrhead)
            #delete!(t.directives, addrhead)
            #delete!(t.to_visit, addrhead)
        #end
    #end
end

"""
Check if value exists for a given address.
"""
function Base.haskey(t::ProgramTrace, addr::Tuple)
    if addr == ()
        # TODO should be true for all subtraces
        return true
    end
    println("haskey, addr=$addr")
    addrhead = addr[1]
    if haskey(t.elements, addrhead)
        element = t.elements[addrhead]
        haskey(element.subtrace, addr[2:end])
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
        # Q: why don't we forward the request for the value down to the subtrace at addr[2:end] even if length(addr) == 1?
        # A: because values can be associated with composite sub-traces. we don't want to forward the address () to a composite subtrace
        # (although that would be a natural address for the return value of a
        # subtrace that might obviate the need for separate storage of value
        # and subtrace) <--- TODO consider this.
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
    t.elements[addrhead] = TraceElement(subtrace, value)
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
        subtrace = t.elements[addr_head].subtrace
    else
        subtrace = empty_trace(generator)
    end
    #if haskey(t.directives, addr_head)
        ## forward the directives for this addr_head to the subtrace
        ## in the future, this can be optimized to only be done once, not in every call to generate!
        #for (subaddr, directive) in t.directives[addr_head]
            #forward!(directive, subaddr, subtrace)
        #end
    #end
    # NOTE: if this was an atomic genreator and it was constrained, then value will be unchanged
    (score, value) = generate!(generator, args, subtrace)
    t.score += score
    t.elements[addr_head] = TraceElement(subtrace, value)
    # record it as visited
    #delete!(t.remaining_to_visit, addr_head)
    value
end

# process a generic tagged value 
function tagged!(trace::ProgramTrace, value, addr_head)
    # NOTE: it's not necessary to create an atomic trace here
    # this can be optimized out
    local subtrace::AtomicTrace
    if haskey(trace.elements, addr_head)
        subtrace = trace.elements[addr_head].subtrace
        if subtrace.mode == record
            subtrace.value = value
        elseif subtrace.mode == constrain || subtrace.mode == propose
            error("cannot constrain or propose a non-generator invocation at $addr_head")
        end
        # if the mode is intervene, then don't change the value in the subtrace
    else
        subtrace = AtomicTrace(value)
    end
    #if haskey(trace.directives, addr_head)
        ## process the any interventions
        #for (subaddr, directive) in trace.directives[addr_head]
            #if isa(directive, Intervention)
                #return_value = directive.value
            #else
                #error("directive $directive only supported for generator invocations")
            #end
        #end
    #end
    trace.elements[addr_head] = TraceElement(subtrace, subtrace.value)
    #delete!(trace.remaining_to_visit, addr_head)
    subtrace.value
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
    #trace.remaining_to_visit = trace.to_visit
    value = p.program(trace, args...)
    score = finalize!(trace)
    (score, value)
end

# TODO make this true for all generators
#(p::ProbabilisticProgram)(args...) = (p, args)
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
