##########################
# Probabilistic programs #
##########################

# Probabilistic programs are a type of Generator that use dictionary-based
# trace types and are constructed by writing Julia functions annotated with
# 'tags'.


## Trace types for probabilistic programs

using DataStructures

abstract type AbstractTrace end

struct TraceElement{T}
    value::Nullable{T}
    mode::SubtraceMode
end

TraceElement(value, mode::SubtraceMode) = TraceElement(Nullable(value), mode)

function set_value!(element::TraceElement{T}, value::T) where {T}
    element.value = Nullable{T}(value)
end

mutable struct Trace <: AbstractTrace
    elements::Dict{Any, TraceElement}
    subtraces::Dict{Any, Any}
    visited::Set{Any}

    # keys are names of sub-traces
    # valus are mapping from names to (alias, mode, visited, [value])
    # at the end of generate! we should check that all aliases were visited
    # e.g. "cluster-7" => ("x2", constrain, false, 2) for a constraint
    # e.g. "cluster-7" => ("x2", record, false)
    aliases::Dict{Any, Dict{Any, Any}}
    score::Float64
    function Trace()
        elements = Dict{Any, TraceElement}()
        subtraces = Dict{Any, Any}()
        visited = OrderedSet{Any}()
        aliases = Dict{Any, Dict{Any, Any}}()
        new(elements, subtraces, visited, aliases, 0.0)
    end
end

# TODO modify DifferentiableTrace
# TODO add aliases
mutable struct DifferentiableTrace <: AbstractTrace
    elements::Dict{Any, TraceElement}
    subtraces::Dict{Any, Any}
    visited::OrderedSet{Any}
    score::GenScalar # becomes type GenFloat (which can be automatically converted from a Float64)
    tape::Tape
    function DifferentiableTrace()
        elements = Dict{Any, TraceElement}()
        subtraces = Dict{Any, Any}()
        visited = OrderedSet{Any}()
        tape = Tape()
        new(elements, subtraces, visited, GenScalar(0.0, tape), tape)
    end
end

# TODO rename to names or addresses
choices(trace::AbstractTrace) = keys(trace.elements)

function Base.print(io::IO, trace::Trace)
    println(io, "Trace(")
    indent = "  "
    for (name, element) in trace.elements
        if element.mode == record
            println(io, "$indent $name = $(element.value)")
        elseif element.mode == propose
            println(io, "$indent+$name = $(element.value)")
        elseif element.mode == constrain
            println(io, "$indent*$name = $(element.value)")
        elseif element.mode == intervene
            println(io, "$indent!$name = $(element.value)")
        end
    end
    println(io, ")")
end

Base.delete!(trace::AbstractTrace, name) = delete!(trace.elements, name)

function constrain!(trace::AbstractTrace, name, val)
    trace.elements[name] = TraceElement(val, constrain)
end

function intervene!(trace::AbstractTrace, name, val)
    trace.elements[name] = TraceElement(val, intervene)
end

function parametrize!(trace::DifferentiableTrace, name, val::ConcreteValue)
    # just an intervene! that converts it to a GenValue first (with the right tape)
    trace.elements[name] = TraceElement(makeGenValue(val, trace.tape), intervene)
end

propose!(trace::AbstractTrace, name) = push!(trace.proposals, name)

function prepare!(trace::AbstractTrace)
    trace.visited = Set{Any}()
    trace.aliases = Dict{Any, Dict{Any, Any}}()
end

function check_not_visited(trace::AbstractTrace, name)
    if name in trace.visited
        error("$name already visited")
    end
end

function check_visited(trace::AbstractTrace)
    for (name, element) in trace.elements
        if (((element.mode == constrain) ||
            (element.mode == propose) ||
            (element.mode == intervene)) &&
           !(name in trace.visited))
            error("$name in mode $mode was not visited")
        end
    end
end

function finalize!(trace::Trace)
    check_visited(trace)
    previous_score = trace.score
    trace.score = 0.0
    previous_score
end

function finalize!(trace::DifferentiableTrace)
    check_visited(trace)
    previous_score = trace.score
    backprop(previous_score)
    trace.score = GenScalar(0.0, trace.tape)
    trace.tape = Tape()
    concrete(previous_score)
end

derivative(trace::DifferentiableTrace, name) = partial(value(trace, name))

function hasvalue(trace::AbstractTrace, name)
    haskey(trace.elements, name) && !isnull(trace.elements[name].value)
end

function hasconstraint(trace::AbstractTrace, name)
    haskey(trace.constraints, name) && trace.elements[name].mode == constrain
end

value(trace::AbstractTrace, name) = get(trace.elements[name].value)


## Probabilistic program generator

println("defining prob prog")
struct ProbabilisticProgram <: Generator{AbstractTrace}
    program::Function
end

# process a tagged generic generator call
function tagged!(trace::Trace, generator::Generator{TraceType}, args::Tuple, name) where {TraceType}
    check_not_visited(trace, name)
    local subtrace::TraceType
    if haskey(trace.subtraces, name)
        subtrace = trace.subtraces[name]
    else
        subtrace = TraceType()
        trace.subtraces[name] = subtrace
    end

    # convert directives applied to aliases onto subtrace directives
    if haskey(trace.aliases, name)
        for (subname, alias) in trace.aliases[name]
            if alias in trace.elements
                element = trace.elements[alias]
                if element.mode == constrain
                    constrain!(subtrace, subname, get(element.value))
                elseif element.mode == propose
                    propose!(subtrace, subname)
                elseif element.mode == intervene
                    intervene!(subtrace, subname, get(element.value))
                end
            end
        end
    end

    # TODO need to modify generate! to return both!
    (trace.score, value) = generate!(generator, args, subtrace)
    
    # record any values into elements for the alias
    if haskey(trace.aliases, name)
        for (subname, alias) in trace.aliases[name]
            if !(alias in trace.elements)
                # aliases are recorded
                trace.elements[alias] = TraceElement(value, record)
            else
                # retain the same mode for the alias
                set_value!(trace.elements[alias], value)
            end
            push!(trace.visited, alias)
        end
    end
    value
end

# process a tagged atomic generator call
function tagged!(trace::Trace, generator::AtomicGenerator{T}, args::Tuple, name) where {T}
    check_not_visited(trace, name)

    # NOTE: atomic generators cannot be aliased

    # the value is stored in the trace, not the subtrace, as with general generators
    local value::T
    local subtrace::AtomicTrace{T}
    local element::TraceElement{T}
    exists = hasvalue(trace.elements, name)
    if exists
        element = trace.elements[name]
        value = element.value
        if element.mode == constrain
            subtrace = AtomicTrace(value)
            constrain!(subtrace, value)
        elseif element.mode == propose
            subtrace = AtomicTrace(T)
            propose!(subtrace, value)
        elseif element.mode == intervene
            push!(trace.visited, name)
            return value
        else
            subtrace = AtomicTrace(T)
        end
    else
        subtrace = AtomicTrace(T)
    end
    (score, value) = generate!(generator, args, subtrace)
    trace.score += score
    if exists
        element.value = value
    else
        element = TraceElement(value, record)
    end
    push!(trace.visited, name)
    value
end

# process a generic tagged value 
function tagged!(trace::Trace, value::T, name) where {T}
    check_not_visited(trace, name)
    return_value = value
    local element::TraceElement{T}
    if haskey(trace.elements, name)
        element = trace.elements[name]
        if trace.element.mode == constrain 
            error("cannot constrain $name, it is not a generator call")
        elseif trace.element.mode == propose
            error("cannot propose $name, it is not a generator call")
        elseif trace.element.mode == intervene
            return_value = get(trace.element.value)
        end
    end
    push!(trace.visited, name)
    return_value
end

# add an alias mapping a name in a sub-trace to a name in this trace
function add_alias!(trace::Trace, alias, name, subname)

    # the alias must come earlier in the program than the sub-trace it is aliasing
    check_not_visited(trace, name)

    # TODO: enforce that the the alias can only be use once
    # NOTE: we don't use trace.visited to confirm the aliases are not visited
    # because we mark an alias as visited when its target is visited.

    if !haskey(trace.aliases, name)
        trace.aliases[name] = Dict{Any, Any}()
    end
    trace.aliases[name][subname] = alias
    nothing
end


# create a new probabilistic program
macro program(args, body)

    # generate new symbol for this execution trace
    trace_symbol = gensym()

    # first argument is the trace
    new_args = [:($trace_symbol::AbstractTrace)]

    # remaining arguments are the original arguments
    if args.head == :(::)

        # single argument
        push!(new_args, args)
        name = Nullable{Symbol}()
    elseif args.head == :tuple

        # multiple arguments
        for arg in args.args
            push!(new_args, arg)
        end
        name = Nullable{Symbol}()
    else
        error("invalid @program")
    end
    arg_tuple = Expr(:tuple, new_args...)

    # overload the tag function to tag values in the correct trace
    prefix = quote
        tag(gen::AtomicGenerator, stuff::Tuple, name) = tagged!($trace_symbol, gen, stuff, name)

        # primitives overload invocation syntax () and expand into this:
        tag(gen_and_args::Tuple{AtomicGenerator,Tuple}, name) = tagged!($trace_symbol, gen_and_args[1], gen_and_args[2], name)

        # arbitrary non-generator tagged values
        tag(other, name) = tagged!($trace_symbol, other, name)

        # tag((name, subname) => alias) adds an alias 
        tag(arg::Pair{Tuple{Any, Any}, Any}) = add_alias!($trace_symbol, arg[2], arg[1][1], arg[1][2])
    end

    # evaluates to a ProbabilisticProgram struct
    Expr(:call, :ProbabilisticProgram, 
        Expr(:function, arg_tuple, quote $prefix; $body end))
end

function generate!(p::ProbabilisticProgram, args::Tuple, trace::AbstractTrace)
    prepare!(trace)
    value = p.program(trace, args...)
    score = finalize!(trace)
    (score, value)
end

(p::ProbabilisticProgram)(args...) = (p, args)


## Binary generate for two probabilistic program traces

function generate!(model_trace::AbstractTrace, model_program::ProbabilisticProgram,
                   proposal_program::ProbabilisticProgram, mapping::Dict{Any,Any})
    proposal_trace = Trace()
    new_model_constraints = Set{Any}()
    for (proposal_name, model_name) in mapping
        if !hasconstraint(model_trace, model_name)
            propose!(proposal_trace, proposal_name)
            push!(new_model_constraints, model_name)
        else
            constrain!(proposal_trace, proposal_name, value(model_trace, model_name))
        end
    end
    (proposal_score, _) = generate!(proposal_program, proposal_trace)

    # score the values in the mapping, along with any extant constraints in the model trace,
    # under the model program
    for (proposal_name, model_name) in mapping
        if !hasconstraint(model_trace, model_name)
            constrain!(model_trace, model_name, value(proposal_trace, proposal_name))
        end
    end
    (model_score, _) = generate!(model_trace, model_program)

    # convert new constraints to to just recorded values (keep old constraints)
    for model_name in new_model_constraints
        unconstrain!(model_trace, model_name)
    end

    # NOTE: this doesn't return a value, just a score
    model_score - proposal_score
end


export Trace
export DifferentiableTrace
export AbstractTrace
export @program
export constrain!
export unconstrain!
export intervene!
export parametrize!
export derivative
export propose!
export hasvalue
export value 
export score
export hasconstraint
export choices
