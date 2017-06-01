include("../trace.jl")

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

function logsumexp(x::Array{Float64,1})
    maxx = maximum(x)
    maxx + log(sum(exp(x - maxx)))
end

function linreg_infer(num_samples::Int, xs::Array{Float64,1}, ys::Array{Float64,1})
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

function linreg_demo()
    xs = collect(linspace(-3, 3, 7))
    ys = -xs
    ys[end] = 4 # an outlier 
    for i=1:100
        (slope, intercept, outliers) = linreg_infer(100, xs, ys)
        println("slope=$slope, intercept=$intercept")
        for (x, y, o) in zip(xs, ys, outliers)
            println("($x, $y) $(o? "outlier" : "inlier" )")
        end
    end
end
linreg_demo()
