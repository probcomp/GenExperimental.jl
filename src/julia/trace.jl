using Distributions
using PyPlot

modules = Dict()
macro register_module(name, simulator, regenerator)
    if name.head != :quote error("invalid module name") end
    name = name.args[1]
    modules[name] = Pair(simulator, regenerator) # simulator returns val and log weight
    eval(quote $name = (args...) -> ($simulator)(args...)[1] end) # todo do this without killing types
end

flip_regenerate(x::Bool, p::Float64) = x ? log(p) : log1p(-p)
flip_simulate(p::Float64) = begin x = rand() < p; x, flip_regenerate(x, p) end
@register_module(:flip, flip_simulate, flip_regenerate)

normal_regenerate(x::Float64, mu::Float64, std::Float64) = logpdf(Normal(mu, std), x)
normal_simulate(mu::Float64, std::Float64) = begin x = rand(Normal(mu, std)); x, normal_regenerate(x, mu, std) end
@register_module(:normal, normal_simulate, normal_regenerate)

type Trace
    vals::Dict{String,Any}
    outputs::Set{String}
    log_weight::Float64 # becomes type GenNum (which can be automatically converted from a Float64)
    function Trace()
        new(Dict{String,Any}(), Set{String}(), 0.0)
    end
end

macro ~(expr, name)
    if expr.head != :call
        error("invalid use of ~: expr.head != :call")
    end
    proc = expr.args[1]
    args = expr.args[2:end]
    if !haskey(modules, proc)
        error("unknown probabilistic module:", proc)
    end
    simulator, regenerator = modules[proc]
    return quote
        local name = $name
        local val
        if haskey(T.vals, name) # T is a reserved symbol for 'trace'
            if name in T.outputs
                error("$name in both outputs and vals of trace")
            end
            val = T.vals[name]
            T.log_weight += $(Expr(:call, regenerator, :val, args...))
        else
            val, log_weight = $(Expr(:call, simulator, args...))
            T.vals[name] = val
            if name in T.outputs
                T.log_weight -= log_weight
            end
        end
        val
    end
end
