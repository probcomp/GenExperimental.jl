using Distributions
using PyPlot

type DifferentiableTrace # TODO hack b/c we have separate tapes
    vals::Dict{String,Any}
    outputs::Set{String}
    log_weight::GenNum # becomes type GenNum (which can be automatically converted from a Float64)
    function DifferentiableTrace(tape::Tape)
        new(Dict{String,Any}(), Set{String}(), GenNum(0.0, tape))
    end
end

type Trace 
    vals::Dict{String,Any}
    outputs::Set{String}
    log_weight::Float64
    function Trace()
        new(Dict{String,Any}(), Set{String}(), 0.0)
    end
end

# TODO implement these for DifferentiableTrace as well
function Base.getindex(trace::Trace, name::String) 
    trace.vals[name]
end

function Base.setindex!(trace::Trace, val::Any, name::String)
    trace.vals[name] = val
end

Base.keys(trace::Trace) = keys(trace.vals)

function score(trace::Trace)
    return trace.log_weight
end

function fail(T::Trace)
    T.log_weight = -Inf
end

# TODO duplicated code
function Base.getindex(trace::DifferentiableTrace, name::String) 
    trace.vals[name]
end

function Base.setindex!(trace::DifferentiableTrace, val::Any, name::String)
    trace.vals[name] = val
end

Base.keys(trace::DifferentiableTrace) = keys(trace.vals)

function score(trace::DifferentiableTrace)
    return trace.log_weight
end

function fail(T::DifferentiableTrace)
    T.log_weight = -Inf
end



macro ~(expr, name)
    # TODO: how to do this in a more hygenic way?
    if expr.head != :call
        error("invalid use of ~: expr.head != :call")
    end
    proc = expr.args[1]
    args = map((a) -> esc(a), expr.args[2:end])
    if !haskey(modules, proc)
        error("unknown probabilistic module:", proc)
    end
    simulator, regenerator = modules[proc]
    return quote
        local name = $(esc(name))
        local val
        if haskey($(esc(:T)).vals, name) # T is a reserved symbol for 'trace'
            if name in $(esc(:T)).outputs
                error("$name in both outputs and vals of trace")
            end
            val = $(esc(:T)).vals[name]
            $(esc(:T)).log_weight += $(Expr(:call, regenerator, :val, args...))
        else
            val, log_weight = $(Expr(:call, simulator, args...))
            $(esc(:T)).vals[name] = val
            if name in $(esc(:T)).outputs
                # NOTE: if there are any outputs there should be no constraints
                # the two may be compatible if we use a minus log weight for outputs
                # but we are not doing that now. we should check somewhere that
                # T.vals and T.outputs do not both contains values initially.
                # TODO check
                $(esc(:T)).log_weight += log_weight 
            end
        end
        val
    end
end

macro constrain(name, val)
    # TODO: how to do this in a more hygenic way?
    return quote
        local name = $(esc(name))
        $(esc(:T)).vals[name] = $(esc(val))
    end
end

macro unconstrain(name)
    return quote
        delete!($(esc(:T)).vals, $name)
    end
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
        $(esc(:T)).vals[$(esc(to))] = $(esc(:__T_SEND)).vals[$(esc(from))]
    end
end


# exports
export Trace
export DifferentiableTrace
export @~
export fail
export @in
export @constrain
export @unconstrain
