using Distributions
using PyPlot
using DataStructures

abstract AbstractTrace

type Trace <: AbstractTrace
    constraints::OrderedDict{String,Any}
    interventions::OrderedDict{String,Any}
    proposals::OrderedSet{String}
    recorded::OrderedDict{String,Any}
    visited::OrderedSet{String}
    log_weight::Float64
    function Trace()
        constraints = OrderedDict{String,Any}()
        interventions = OrderedDict{String,Any}()
        proposals = OrderedSet{String}()
        recorded = OrderedDict{String,Any}()
        visited = OrderedSet{String}()
        new(constraints, interventions, proposals, recorded, visited, 0.0)
    end
end

type DifferentiableTrace <: AbstractTrace
    constraints::OrderedDict{String,Any}
    interventions::OrderedDict{String,Any}
    proposals::OrderedSet{String}
    recorded::OrderedDict{String,Any}
    visited::OrderedSet{String}
    log_weight::GenScalar # becomes type GenFloat (which can be automatically converted from a Float64)
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

function intervene!(trace::AbstractTrace, name::String, val::Any)
    check_not_exists(trace, name)
    trace.interventions[name] = val
end

function parametrize!(trace::DifferentiableTrace, name::String, val::Float64)
    # just an intervene! that converts it to a GenScalar first (with the right tape)
    check_not_exists(trace, name)
    trace.interventions[name] = GenScalar(val, trace.tape)
end

function parametrize!(trace::DifferentiableTrace, name::String, val::Vector{Float64})
    check_not_exists(trace, name)
    trace.interventions[name] = GenVector(val, trace.tape)
end

function parametrize!(trace::DifferentiableTrace, name::String, val::Matrix{Float64})
    check_not_exists(trace, name)
    trace.interventions[name] = GenMatrix(val, trace.tape)
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

# should call these 'prepare'
function prepare(trace::Trace)
    trace.visited = OrderedSet{String}()
    trace.log_weight = 0.0
end

# automatically do reset of tape when finishing the @generate?
# this way, future parameterizations aren't on the same tape
# as the just-produced trace.. it's like cutting the tape off.
function prepare(trace::DifferentiableTrace)
    trace.visited = OrderedSet{String}()
    trace.log_weight= GenScalar(0.0, trace.tape)
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

# should call these 'finalize'
function finalize(trace::Trace)
    check_visited(trace)
end

function finalize(trace::DifferentiableTrace)
    check_visited(trace)
    trace.tape = Tape()
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
            $(esc(:T)).log_weight += $(Expr(:call, esc(:regenerate), mod, :val, args...))
        else
            val, log_weight = $(Expr(:call, esc(:simulate), mod, args...))
            # NOTE: will overwrite the previous value if it was already recorded
            $(esc(:T)).recorded[name] = val
            if name in $(esc(:T)).proposals
                $(esc(:T)).log_weight += log_weight 
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
        # reset the score to zero, and the visited set to empty
        prepare($(esc(trace)))

        # run the program, passing in the trace as the first arg.
        local value = $(esc(function_name))($(esc(trace)), $(map((a) -> esc(a), program.args[2:end])...))

        # reset the tape (the old tape is preserved through
        # the reference of the log-weight)
        # this is so the old log-weight can still be referenced
        # (and gradients for it can be taken) while new parameters
        # can be meanwhile parameterize!-d 
        finalize($(esc(trace)))
        value
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


# exports
export Trace
export DifferentiableTrace
export AbstractTrace
export @program
export @generate
export @~
export constrain!
export intervene!
export parametrize!
export derivative
export propose!
export hasvalue
export value 
export backprop
export score
export hasconstraint
export choices
