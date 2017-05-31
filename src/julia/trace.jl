using Distributions
using PyPlot

modules = Dict()
macro register_module(name, simulator, regenerator)
    if name.head != :quote error("invalid module name") end
    name = name.args[1]
    modules[name] = Pair(simulator, regenerator)
    eval(quote $name = $simulator end)
end

flip_simulate(p::Float64) = rand() < p
flip_regenerate(x::Bool, p::Float64) = x ? log(p) : log1p(-p)
@register_module(:flip, flip_simulate, flip_regenerate)

normal_simulate(mu::Float64, std::Float64) = rand(Normal(mu, std))
normal_regenerate(x::Float64, mu::Float64, std::Float64) = logpdf(Normal(mu, std), x)
@register_module(:normal, normal_simulate, normal_regenerate)

type Trace
    vals::Dict
    log_weight::Float64
    function Trace()
        vals = Dict()
        new(vals, 0.0)
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
            val = T.vals[name]
            T.log_weight += $(Expr(:call, regenerator, :val, args...))
        else
            val = $(Expr(:call, simulator, args...))
            T.vals[name] = val
        end
        val
    end
end

function linear_regression(T::Trace, prior_mu::Float64, prior_std::Float64, 
                           xs::Array{Float64,1})
    inlier_noise = 0.1
    outlier_noise = 10.0
    prob_outlier = 0.1
    slope = normal(prior_mu, prior_std) ~ "slope"
    intercept = normal(prior_mu, prior_std) ~ "intercept"
    ys = Array{Float64, 1}(length(xs))
    for i=1:length(xs)
        y_mean = intercept + slope * xs[i]
        noise = flip(prob_outlier) ? inlier_noise : outlier_noise
        ys[i] = normal(y_mean, noise) ~ "y$i"
    end
end

# example:

function logsumexp(x::Array{Float64,1})
    maxx = maximum(x)
    maxx + log(sum(exp(x - maxx)))
end

xs = collect(linspace(-3, 3, 7))
ys = -xs
# ys[end] = 4 # an outlier 

function linreg_infer(num_samples::Int)
    log_weights = Float64[]
    slopes = Float64[]
    intercepts = Float64[]
    for i=1:num_samples
        trace = Trace()
        for (i, y) in enumerate(ys)
            trace.vals["y$i"] = y
        end
        linear_regression(trace, 0.0, 1.0, xs)
        push!(log_weights, trace.log_weight)
        push!(slopes, trace.vals["slope"])
        push!(intercepts, trace.vals["intercept"])
    end
    weights = exp(log_weights - logsumexp(log_weights))
    chosen = rand(Categorical(weights))
    return (slopes[chosen], intercepts[chosen])
end

for i=1:100
    (slope, intercept) = linreg_infer(100)
    println("slope=$slope, intercept=$intercept")
end



