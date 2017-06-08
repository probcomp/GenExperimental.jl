using Distributions
using PyPlot
using DataStructures

abstract AbstractTrace

type Trace <: AbstractTrace
    constraints::OrderedDict{String,Any}
    interventions::OrderedDict{String,Any}
    proposals::OrderedSet{String}
    recorded::OrderedDict{String,Any}
    log_weight::Float64
    function Trace()
        constraints = OrderedDict{String,Any}()
        interventions = OrderedDict{String,Any}()
        proposals = OrderedSet{String}()
        recorded = OrderedDict{String,Any}()
        new(constraints, interventions, proposals, recorded, 0.0)
    end
end

type DifferentiableTrace <: AbstractTrace
    constraints::OrderedDict{String,Any}
    interventions::OrderedDict{String,Any}
    proposals::OrderedSet{String}
    recorded::OrderedDict{String,Any}
    log_weight::GenFloat # becomes type GenFloat (which can be automatically converted from a Float64)
    tape::Tape
    function DifferentiableTrace()
        tape = Tape()
        constraints = OrderedDict{String,Any}()
        interventions = OrderedDict{String,Any}()
        proposals = OrderedSet{String}()
        recorded = OrderedDict{String,Any}()
        new(constraints, interventions, proposals, recorded, GenFloat(0.0, tape), tape)
    end
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

function intervene!(trace::AbstractTrace, name::String, val::Any)
    check_not_exists(trace, name)
    trace.interventions[name] = val
end

function parametrize!(trace::DifferentiableTrace, name::String, val::Float64)
    # just an intervene! that converts it to a GenFloat first (with the right tape)
    check_not_exists(trace, name)
    trace.interventions[name] = GenFloat(val, trace.tape)
end

function parametrize!(trace::DifferentiableTrace, name::String, val::Matrix{Float64})
    check_not_exists(trace, name)
    trace.interventions[name] = GenMatrix(val, trace.tape)
end

function parametrize!(trace::DifferentiableTrace, name::String, val::Array{Float64,1})
    check_not_exists(trace, name)
    # convert to a column vector
    trace.interventions[name] = GenMatrix(reshape(val, length(val), 1), trace.tape)
end




function propose!(trace::AbstractTrace, name::String)
    check_not_exists(trace, name)
    push!(trace.proposals, name)
end

function backprop(trace::DifferentiableTrace)
    backprop(trace.log_weight)
end

function score(trace::AbstractTrace)
    concrete(trace.log_weight)
end

function reset_score(trace::Trace)
    trace.log_weight = 0.0
end

function reset_score(trace::DifferentiableTrace)
    trace.tape = Tape()
    trace.log_weight= GenFloat(0.0, tape)
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

    simulator, regenerator = modules[proc]
    return quote
        local name = $(esc(name))
        local val
        if haskey($(esc(:T)).interventions, name)
            val = $(esc(:T)).interventions[name]
        elseif haskey($(esc(:T)).constraints, name)
            val = $(esc(:T)).constraints[name]
            $(esc(:T)).log_weight += $(Expr(:call, esc(regenerator), :val, args...))
        else
            val, log_weight = $(Expr(:call, esc(simulator), args...))
            # NOTE: will overwrite the previous value if it was already recorded
            $(esc(:T)).recorded[name] = val
            if name in $(esc(:T)).proposals
                $(esc(:T)).log_weight += log_weight 
            end
        end
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
        val
    end
end

macro ~(expr, name)
    # WARNING: T is a reserved symbol for 'trace'. It is an error if T occurs in the program.
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

function fail(trace::AbstractTrace)
    trace.log_weight = -Inf
end

macro in(context, code)
    if typeof(context) == Expr
        if length(context.args) != 3
            error("expected @in <model_trace> <= inference_trace begin ... end")
        end
        symb = context.args[1]
        model_trace = context.args[2]
        inference_trace = context.args[3]
        if symb != :<=
            error("expected @in <model_trace> <= inference_trace begin ... end")
        end
        return quote
            $(esc(:T)) = $(esc(model_trace))
            $(esc(:__T_SEND)) = $(esc(inference_trace))
            $(esc(code))
        end
    else
        return quote
            $(esc(:T)) = $(esc(context))
            $(esc(code))
        end
    end
end

macro constrain(mapping)
    if mapping.head != :call || length(mapping.args) != 3 || mapping.args[1] != :<=
        error("invalid input to @constrain, expected: @constrain(<to_name> <= <from_name>)")
    end
    to = mapping.args[2]
    from = mapping.args[3]
    return quote
        local val = get($(esc(:__T_SEND)), $(esc(from)))
        # TODO check that the name was a 'propose' in the other trace
        constrain($(esc(:T)), $(esc(to))) = val
    end
end

# exports
export Trace
export DifferentiableTrace
export AbstractTrace
export @~
export constrain!
export intervene!
export parametrize!
export derivative
export propose!
export hasvalue
export fail
export value 
export backprop
export score
export reset_score
export hasconstraint
