using Distributions
using PyPlot

modules = Dict()
macro register_module(name, simulator, regenerator)
    if name.head != :quote error("invalid module name") end
    name = name.args[1]
    modules[name] = Pair(simulator, regenerator) # simulator returns val and log weight
    eval(quote $name = (args...) -> ($sampler)(args...)[1] end) # todo do this without killing types
end

flip_regenerate(x::Bool, p::Float64) = x ? log(p) : log1p(-p)
flip_simulate(p::Float64) = begin x = rand() < p; x, flip_regenerate(x, p) end
@register_module(:flip, flip_simulate, flip_regenerate)

normal_regenerate(x::Float64, mu::Float64, std::Float64) = logpdf(Normal(mu, std), x)
normal_simulate(mu::Float64, std::Float64) = begin x = rand(Normal(mu, std)); x, normal_regenerate(x, mu, std) end
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
            val, _ = $(Expr(:call, simulator, args...)) #NOTE: simulator weight is unused here
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
        noise = (flip(prob_outlier) ~ "o$i") ? outlier_noise : inlier_noise 
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
ys[end] = 4 # an outlier 

function linreg_infer(num_samples::Int)
    log_weights = Float64[]
    slopes = Float64[]
    intercepts = Float64[]
    outliers = Array{Bool,1}[]
    for i=1:num_samples
        trace = Trace()
        for (i, y) in enumerate(ys)
            trace.vals["y$i"] = y
        end
        linear_regression(trace, 0.0, 2.0, xs)
        push!(log_weights, trace.log_weight)
        push!(slopes, trace.vals["slope"])
        push!(intercepts, trace.vals["intercept"])
        push!(outliers, map((i) -> trace.vals["o$i"], 1:length(xs)))
    end
    weights = exp(log_weights - logsumexp(log_weights))
    chosen = rand(Categorical(weights))
    return (slopes[chosen], intercepts[chosen], outliers[chosen])
end

for i=1:100
    (slope, intercept, outliers) = linreg_infer(100)
    println("slope=$slope, intercept=$intercept")
    for (x, y, o) in zip(xs, ys, outliers)
        println("($x, $y) $(o? "outlier" : "inlier" )")
    end
end
