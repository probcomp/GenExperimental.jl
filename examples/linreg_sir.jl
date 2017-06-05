using Gen
import Distributions
using PyPlot

function linear_regression(T::Trace, prior_mu::Float64, prior_std::Float64, 
                           xs::Array{Float64,1})
    inlier_noise = gamma(1., 1.) ~ "inlier_noise"
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
    log_weights = Array{Float64,1}(num_samples)
    traces = Array{Trace,1}(num_samples)
    for sample=1:num_samples
        trace = Trace()
        @in trace begin
            for (i, y) in enumerate(ys)
                @constrain("y$i", y)
            end
        end
        linear_regression(trace, 0.0, 2.0, xs)
        traces[sample] = trace
        log_weights[sample] = trace.log_weight
    end
    weights = exp(log_weights - logsumexp(log_weights))
    chosen = rand(Distributions.Categorical(weights))
    return traces[chosen]
end

function render_linreg_trace(trace::Trace, xs::Array{Float64,1})
    ax = plt[:gca]()
    n = length(xs)
    ys = map((i) -> trace.vals["y$i"], 1:n)
    outlier_statuses = map((i) -> trace.vals["o$i"], 1:n)
    slope = trace.vals["slope"]
    intercept = trace.vals["intercept"]
    inlier_noise = trace.vals["inlier_noise"]
    xmin, xmax = minimum(xs), maximum(xs)
    xspan = xmax - xmin
    ymin, ymax = minimum(ys), maximum(ys)
    yspan = ymax - ymin
    xlim = [xmin - 0.1 * xspan, xmax + 0.1 * xspan]
    ylim = [ymin - 0.1 * yspan, ymax + 0.1 * yspan]
    colors = map((i) -> outlier_statuses[i] ? "orange" : "blue", 1:n)
    plt[:scatter](xs, ys, c=colors)
    xs_line = [xmin, xmax]
    ys_line = intercept + slope * [xmin, xmax]
    plt[:plot](xs_line, ys_line, color="black")
    plt[:fill_between](xs_line, ys_line - inlier_noise, ys_line, color="black", alpha=0.3)
    plt[:fill_between](xs_line, ys_line, ys_line + inlier_noise, color="black", alpha=0.3)
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function linreg_demo()
    srand(1)
    xs = collect(linspace(-3, 3, 7))
    ys = -xs + randn(length(xs)) * 0.1
    ys[end] = 4 # an outlier 
    plt[:figure](figsize=(10, 10))
    num_samples = 25
    for i=1:num_samples
        trace = linreg_infer(1000, xs, ys)
        plt[:subplot](5, 5, i)
        render_linreg_trace(trace, xs)
    end
    plt[:savefig]("samples.pdf")
end
linreg_demo()
