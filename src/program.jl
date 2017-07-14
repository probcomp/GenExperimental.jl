##########################
# Probabilistic programs #
##########################

# Probabilistic programs are a type of Generator that use dictionary-based
# trace types and are constructed by writing Julia functions annotated with
# 'tags'.


## Trace types for probabilistic programs

using DataStructures

abstract type AbstractTrace end

mutable struct Trace <: AbstractTrace
    constraints::OrderedDict{String,Any}
    interventions::OrderedDict{String,Any}
    proposals::OrderedSet{String}
    recorded::OrderedDict{String,Any}
    visited::OrderedSet{String}
    score::Float64
    function Trace()
        constraints = OrderedDict{String,Any}()
        interventions = OrderedDict{String,Any}()
        proposals = OrderedSet{String}()
        recorded = OrderedDict{String,Any}()
        visited = OrderedSet{String}()
        new(constraints, interventions, proposals, recorded, visited, 0.0)
    end
end

mutable struct DifferentiableTrace <: AbstractTrace
    constraints::OrderedDict{String,Any}
    interventions::OrderedDict{String,Any}
    proposals::OrderedSet{String}
    recorded::OrderedDict{String,Any}
    visited::OrderedSet{String}
    score::GenScalar # becomes type GenFloat (which can be automatically converted from a Float64)
    tape::Tape
    function DifferentiableTrace()
        tape = Tape()
        constraints = OrderedDict{String,Any}()
        interventions = OrderedDict{String,Any}()
        proposals = OrderedSet{String}()
        recorded = OrderedDict{String,Any}()
        visited = OrderedSet{String}()
        new(constraints, interventions, proposals, recorded, visited, GenScalar(0.0, tape), tape)
    end
end

function choices(trace::AbstractTrace)
    keys(trace.recorded)
end

function Base.print(trace::AbstractTrace)
    println("-- Constraints --")
    for k in keys(trace.constraints)
        v = trace.constraints[k]
        println("$k => $v")
    end
    println("-- Interventions --")
    for k in keys(trace.interventions)
        v = trace.interventions[k]
        println("$k => $v")
    end
    println("-- Proposals --")
    for k in trace.proposals
        println("$k")
    end
    println("-- Recorded --")
    for k in keys(trace.recorded)
        v = trace.recorded[k]
        println("$k => $v")
    end
end

function check_not_exists(trace::AbstractTrace, name::String)
    if haskey(trace.constraints, name)
        error("$name is already marked as a constraint")
    end
    if haskey(trace.interventions, name)
        error("$name is already marked as an intervention")
    end
    if name in trace.proposals
        error("$name is already marked as a proposal")
    end
    if haskey(trace.recorded, name)
        # delete from the recording if we are doing something special with it
        delete!(trace.recorded, name)
    end
end

function constrain!(trace::AbstractTrace, name::String, val::Any)
    check_not_exists(trace, name)
    trace.constraints[name] = val
end

function unconstrain!(trace::AbstractTrace, name::String)
    val = trace.constraints[name]
    delete!(trace.constraints, name)
    trace.recorded[name] = val
end


function intervene!(trace::AbstractTrace, name::String, val::Any)
    check_not_exists(trace, name)
    trace.interventions[name] = val
end

function parametrize!(trace::DifferentiableTrace, name::String, val::ConcreteValue)
    # just an intervene! that converts it to a GenValue first (with the right tape)
    check_not_exists(trace, name)
    trace.interventions[name] = makeGenValue(val, trace.tape)
end

function propose!(trace::AbstractTrace, name::String)
    check_not_exists(trace, name)
    push!(trace.proposals, name)
end

function prepare(trace::Trace)
    trace.visited = OrderedSet{String}()
end

function prepare(trace::DifferentiableTrace)
    trace.visited = OrderedSet{String}()
end

function check_visited(trace::AbstractTrace)
    for name in keys(trace.constraints)
        if !(name in trace.visited)
            error("constraint $name not visited")
        end
    end
    for name in keys(trace.interventions)
        if !(name in trace.visited)
            error("intervention $name not visited")
        end
    end
    for name in trace.proposals
        if !(name in trace.visited)
            error("proposal $name not visited")
        end
    end
end

function finalize(trace::Trace)
    check_visited(trace)
    previous_score = trace.score
    trace.score = 0.0
    previous_score
end

function finalize(trace::DifferentiableTrace)
    check_visited(trace)
    previous_score = trace.score
    backprop(previous_score)
    trace.score = GenScalar(0.0, trace.tape)
    trace.tape = Tape()
    concrete(previous_score)
end


function derivative(trace::DifferentiableTrace, name::String)
    partial(value(trace, name))
end

function Base.delete!(trace::AbstractTrace, name::String)
    if haskey(trace.constraints, name)
        delete!(trace.constraints, name)
    end
    if haskey(trace.interventions, name)
        delete!(trace.interventions, name)
    end
    if name in trace.proposals
        delete!(trace.proposals, name)
    end
    if haskey(trace.recorded, name)
        delete!(trace.recorded, name)
    end
end

function hasvalue(trace::AbstractTrace, name::String)
    if haskey(trace.constraints, name)
        return true
    end
    if haskey(trace.interventions, name)
        return true
    end
    if haskey(trace.recorded, name)
        return true
    end
    return false
end

function hasconstraint(trace::AbstractTrace, name::String)
    haskey(trace.constraints, name)
end

function value(trace::AbstractTrace, name::String)
    if haskey(trace.constraints, name)
        return trace.constraints[name]
    elseif haskey(trace.interventions, name)
        return trace.interventions[name]
    elseif haskey(trace.recorded, name)
        return trace.recorded[name]
    else
        error("trace does not contain a value for $name")
    end
end


## Probabilistic program generator

struct ProbabilisticProgram <: Generator{AbstractTrace}
    program::Function
end

function tagged!(trace::Trace, generator::AtomicGenerator{T}, args::Tuple, name::String) where {T}
	local value::T
	subtrace = AtomicTrace{T}()
	# NOTE: currently the value itself is stored in the trace, not the subtrace
	if haskey(trace.constraints, name)
		value = trace.constraints[name]
		constrain!(subtrace, value)
	elseif name in trace.proposals
		propose!(subtrace)
	end
	trace.score += generate!(generator, args, subtrace)
	value = get(subtrace)
	trace.recorded[name] = value
	push!(trace.visited, name)
	value
end

function tagged!(trace::Trace, value::T, name::String) where {T}
	if haskey(trace.constraints, name)
		error("cannot constrain $name")
	else
		trace.recorded[name] = value
	end
	push!(trace.visited, name)
	value
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
		tag(gen::AtomicGenerator, stuff::Tuple, name::String) = tagged!($trace_symbol, gen, stuff, name)

		# primitives overload invocation syntax () and expand into this:
		tag(gen_and_args::Tuple{AtomicGenerator,Tuple}, name::String) = tagged!($trace_symbol, gen_and_args[1], gen_and_args[2], name)

		# arbitrary non-generator tagged values
		tag(other::Any, name::String) = tagged!($trace_symbol, other, name)
	end

	# evaluates to a ProbabilisticProgram struct
    Expr(:call, :ProbabilisticProgram, 
        Expr(:function, arg_tuple, quote $prefix; $body end))
end

function generate!(p::ProbabilisticProgram, args::Tuple, trace::AbstractTrace)
    prepare(trace)
    p.program(trace, args...)

    # returns the previous score
    finalize(trace)
end

(p::ProbabilisticProgram)(args...) = (p, args)


## Binary generate for two probabilistic program traces

function generate!(model_trace::AbstractTrace, model_program::ProbabilisticProgram,
                   proposal_program::ProbabilisticProgram, mapping::Dict{String,String})
    proposal_trace = Trace()
    new_model_constraints = Set{String}()
    for (proposal_name, model_name) in mapping
        if !hasconstraint(model_trace, model_name)
            propose!(proposal_trace, proposal_name)
            push!(new_model_constraints, model_name)
        else
            constrain!(proposal_trace, proposal_name, value(model_trace, model_name))
        end
    end
    proposal_score = generate!(proposal_program, proposal_trace)

    # score the values in the mapping, along with any extant constraints in the model trace,
    # under the model program
    for (proposal_name, model_name) in mapping
        if !hasconstraint(model_trace, model_name)
            constrain!(model_trace, model_name, value(proposal_trace, proposal_name))
        end
    end
    model_score = generate!(model_trace, model_program)

    # convert new constraints to to just recorded values (keep old constraints)
    for model_name in new_model_constraints
        unconstrain!(model_trace, model_name)
    end
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
