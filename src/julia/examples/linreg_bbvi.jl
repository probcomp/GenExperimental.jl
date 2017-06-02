include("../ad.jl")
include("../trace.jl")
include("../primitives.jl")
using PyPlot

function linear_regression(T::Trace, prior_mu::Float64, prior_std::Float64, 
                           noise_std::Float64, xs::Array{Float64,1})
    slope = normal(prior_mu, prior_std) ~ "slope"
    intercept = normal(prior_mu, prior_std) ~ "intercept"
    ys = Array{Float64, 1}(length(xs))
    for i=1:length(xs)
        y_mean = intercept + slope * xs[i]
        ys[i] = normal(y_mean, noise_std) ~ "y$i"
    end
end

function variational_approximation{M,N,O}(T::DifferentiableTrace, slope_mu::M, 
                                          intercept_mu::N, std::O)
    slope = normal(slope_mu, std) ~ "slope"
    intercept = normal(intercept_mu, std) ~ "intercept"
end

function linreg_infer(xs::Array{Float64,1}, ys::Array{Float64,1})
    iters = 1000
    num_samples = 1000
    step_a = 100.0
    step_b = 0.75
    std = 1.0
    noise_std = 1.0
    prior_std = 2.0
    prior_mu = 0.0
    slope_mu, intercept_mu = 0.0, 0.0
    for iter in 1:iters
        rho = (step_a + iter) ^ (-step_b)
        elbo_est = 0.0
        grad_est = zeros(2)
        for sample=1:num_samples

            # parameters that we can differentiate wrt
            tape = Tape() # todo: reuse the tape, this is quite inefficient
            slope_mu_num = GenNum(slope_mu, tape)
            intercept_mu_num = GenNum(intercept_mu, tape)

            # sample from variational family
            inference_trace = DifferentiableTrace(tape)
            inference_trace.outputs = Set{String}(["slope", "intercept"])
            variational_approximation(inference_trace, slope_mu_num, intercept_mu_num, std)

            # backpropagate to compute gradient
            backprop(inference_trace.log_weight)
            gradient = [adj(slope_mu_num), adj(intercept_mu_num)]

            # compute joint probability under model
            model_trace = Trace()
            for (i, y) in enumerate(ys)
                model_trace.vals["y$i"] = y
            end
            model_trace.vals["slope"] = inference_trace.vals["slope"]
            model_trace.vals["intercept"] = inference_trace.vals["intercept"]
            linear_regression(model_trace, prior_mu, prior_std, noise_std, xs)

            # p(z, x) where z is latents and x is ..
            diff = model_trace.log_weight - concrete(inference_trace.log_weight)
            elbo_est += diff
            grad_est += diff * gradient
        end
        elbo_est /= num_samples
        grad_est /= num_samples
        println("iter: $iter, obj: $elbo_est, intercept_mu: $intercept_mu, slope_mu: $slope_mu")
        slope_mu += rho * grad_est[1]
        intercept_mu += rho * grad_est[2]
    end
end

function linreg_demo()
    srand(1)
    xs = collect(linspace(-3, 3, 7))
    ys = -xs + 1
    println(xs)
    println(ys)
    # intercept should be mean 0, slope should be mean -1
    linreg_infer(xs, ys)
end
linreg_demo()
