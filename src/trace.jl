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

macro ~(expr, name)
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
            $(esc(:T)).log_weight = $(esc(:T)).log_weight + $(Expr(:call, regenerator, :val, args...))
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

# exports
export Trace
export DifferentiableTrace
export @~

