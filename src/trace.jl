using DataStructures
import Distributions

# Probabilistic modules ----------------------------------
#
# Gen.Module{T} is a probabilistic module of return type T
# Each subtype of Module{T} must have two methods:
# 1. simulate(args...)::Tuple{T,N} where N is a number type
# 2. regenerate(::T, args...)::N where N is a number type
# See https://arxiv.org/abs/1612.04759 for the mathematical
# probabilistic module specification

abstract type Module{T} end

modules = Dict{Symbol, Module}()

function register_module(name::Symbol, mod::Module)
    modules[name] = mod
end

export Module
export simulate
export regenerate

# users can register their own modules
export register_module


# Probabilistic programs and traces ----------------------


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
    #trace.score = 0.0
end

function prepare(trace::DifferentiableTrace)
    trace.visited = OrderedSet{String}()
    #trace.score= GenScalar(0.0, trace.tape)
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

function expand_module(expr, name)
    proc = expr.args[1]
    args = map((a) -> esc(a), expr.args[2:end])
   
    # the expression is a module call, it can be intervened, constrained,
    # and proposed, and it will be recorded
    mod = modules[proc]
    return quote
        local name = $(esc(name))
        local val
        if haskey($(esc(:T)).interventions, name)
            val = $(esc(:T)).interventions[name]
        elseif haskey($(esc(:T)).constraints, name)
            val = $(esc(:T)).constraints[name]
            $(esc(:T)).score += regenerate($(mod), val, $(args...))
        else
            (val, score) = simulate($(mod), $(args...))
            # NOTE: will overwrite the previous value if it was already recorded
            $(esc(:T)).recorded[name] = val
            if name in $(esc(:T)).proposals
                $(esc(:T)).score += score 
            end
        end
        push!($(esc(:T)).visited, name)
        val
    end
end

function expand_non_module(expr, name)
    # the expression is not a module call
    # it can only be intervened and recorded, not constrained or proposed
    return quote
        local name = $(esc(name))
        local val
        if haskey($(esc(:T)).interventions, name)
            val = $(esc(:T)).interventions[name]
        elseif haskey($(esc(:T)).constraints, name)
            error("$name cannot be constrained")
        elseif name in $(esc(:T)).proposals
            error("$name cannot be proposed")
        else
            # evaluate the LHS expression and record it
            val = $(esc(expr))
            $(esc(:T)).recorded[name] = val
        end
        push!($(esc(:T)).visited, name)
        val
    end
end

macro tag(expr, name)
    # WARNING: T is a reserved symbol for 'trace'. It is an error if T occurs
    # in the program.
    # TODO: how to do this in a hygenic way?
    is_module_call = (typeof(expr) == Expr) &&
                     (expr.head == :call) && 
                     length(expr.args) >= 1 && 
                     haskey(modules, expr.args[1])
    if is_module_call
        expand_module(expr, name)
    else
        expand_non_module(expr, name)
    end
end

macro program(args, body)
    err() = error("invalid @program definition")
    new_args = [:(T::AbstractTrace)]
    local name::Nullable{Symbol}
    if args.head == :call
        name = args.args[1]
        for arg in args.args[2:end]
            push!(new_args, arg)
        end
    elseif args.head == :(::)
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
        err()
    end
    arg_tuple = Expr(:tuple, new_args...)
    if isnull(name)
        Expr(:function, arg_tuple, body)
    else
        function_name = get(name)
        Main.eval(Expr(:function,
                Expr(:call, function_name, new_args...),
                body))
        #eval(Expr(:export, function_name))
    end
end

macro generate(trace, program)
    err(msg) = error("invalid @generate call: $msg")
    if program.head != :call
        err("program.head != :call")
    end
    function_name = program.args[1]
    quote
        # and the visited set to empty
        prepare($(esc(trace)))

        # run the program, passing in the trace as the first arg.
        local value = $(esc(function_name))(
            $(esc(trace)),
            $(map((a) -> esc(a), program.args[2:end])...))

        # reset the tape (the old tape is preserved through
        # the reference of the score)
        # this is so the old score can still be referenced
        # (and gradients for it can be taken) while new parameters
        # can be meanwhile parameterize!-d 
        previous_score = finalize($(esc(trace)))

        # return the previous score. the return value of the function is discarded.
        previous_score
    end
end

macro generate(model_trace, model_program, proposal_program, mapping)
    quote
        # TODO check that model_trace has no choices set to propose?

        # fill in any missing values in the mapping by sampling from the proposal program
        # constrain the proposal program for any choices that were provided
        proposal_trace = Trace()
        new_model_constraints = Set{String}()
        #local model_trace = $(esc(model_trace))
        for (proposal_name, model_name) in $(esc(mapping))
            if !hasconstraint($(esc(model_trace)), model_name)
                propose!(proposal_trace, proposal_name)
                push!(new_model_constraints, model_name)
            else
                constrain!(proposal_trace, proposal_name, value($(esc(model_trace)), model_name))
            end
        end
        proposal_score = @generate(proposal_trace, $(esc(proposal_program)))

        # score the values in the mapping, along with any extant constraints in the model trace,
        # under the model program
        for (proposal_name, model_name) in $(esc(mapping))
            if !hasconstraint($(esc(model_trace)), model_name)
                constrain!($(esc(model_trace)), model_name, value(proposal_trace, proposal_name))
            end
        end
        model_score = @generate($(esc(model_trace)), $(esc(model_program)))

		# convert new constraints to to just recorded values (keep old constraints)
		for model_name in new_model_constraints
			unconstrain!($(esc(model_trace)), model_name)
		end
        model_score - proposal_score
    end
end

macro generate(model_trace, model_program, proposal_trace, proposal_program, mapping)
    quote
        # TODO check that model_trace has no choices set to propose?

        # fill in any missing values in the mapping by sampling from the proposal program
        # constrain the proposal program for any choices that were provided
        new_model_constraints = Set{String}()
        for (proposal_name, model_name) in $(esc(mapping))
            if !hasconstraint($(esc(model_trace)), model_name)
                propose!($(esc(proposal_trace)), proposal_name)
                push!(new_model_constraints, model_name)
            else
                constrain!($(esc(proposal_trace)), proposal_name, value($(esc(model_trace)), model_name))
            end
        end
        proposal_score = @generate($(esc(proposal_trace)), $(esc(proposal_program)))

        # score the values in the mapping, along with any extant constraints in the model trace,
        # under the model program
        for (proposal_name, model_name) in $(esc(mapping))
            if !hasconstraint($(esc(model_trace)), model_name)
                constrain!($(esc(model_trace)), model_name, value($(esc(proposal_trace)), proposal_name))
            end
        end
        model_score = @generate($(esc(model_trace)), $(esc(model_program)))

		# convert new constraints to to just recorded values (keep old constraints)
		for model_name in new_model_constraints
			unconstrain!($(esc(model_trace)), model_name)
		end
        model_score - proposal_score
    end
end


# TODO: this is substnatially different from the two-arguemnt form of
# @generate in its semantics. perhaps it should be given a 
# different name
macro generate(program)
    err(msg) = error("invalid @generate call: $msg")
    if program.head != :call
        err("program.head != :call")
    end
    function_name = program.args[1]
    Expr(:call, esc(function_name), esc(:T),
         map((a) -> esc(a), program.args[2:end])...)
end

export Trace
export DifferentiableTrace
export AbstractTrace
export @program
export @generate
export @tag
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
