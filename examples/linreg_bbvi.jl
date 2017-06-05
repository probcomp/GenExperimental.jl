using Gen
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

function linear_regression_variational_inference(xs::Array{Float64,1},
                                                 ys::Array{Float64,1})

    # optimization parameters
    iters = 1000
    num_samples = 1000
    step_a = 100.0
    step_b = 0.75

    # fixed standard deviation for variational family
    #std = 1.0

    # fixed model parameters
    noise_std = 1.0
    prior_std = 2.0
    prior_mu = 0.0

    # initial values for variational parameters
    # (the parameters that are being optimized)
    slope_mu, intercept_mu, log_std = 0.0, 0.0, 0.0

    # gradient descent to minimize the difference between the approximation and
    # the posterior
    for iter in 1:iters

        # estimate the objective function (ELBO and its gradient with respect
        # to variational parameters using many runs of the approximation program
        elbo_estimates = Array{Float64,1}(num_samples)
        gradient_estimates = Array{Float64,2}(num_samples,3)

        for sample=1:num_samples

            # run the proposal program so that its score can be differentiated
            # with respect to its inputs (which are the variational parameters)
            tape = Tape()
            slope_mu_num = GenNum(slope_mu, tape)
            intercept_mu_num = GenNum(intercept_mu, tape)
            log_std_num = GenNum(log_std, tape)
            inference_trace = DifferentiableTrace(tape)
            inference_trace.outputs = Set{String}(["slope", "intercept"])
            variational_approximation(inference_trace, slope_mu_num, intercept_mu_num, exp(log_std_num))

            # add contraints to model trace
            model_trace = Trace()
            @in model_trace <= inference_trace begin
                for (i, y) in enumerate(ys)
                    @constrain("y$i", y)
                end
                @constrain("slope" <= "slope")
                @constrain("intercept" <= "intercept")
            end
        
            # run model program
            linear_regression(model_trace, prior_mu, prior_std, noise_std, xs)

            # differentiate the inference score with respect to the variational
            # parameters
            backprop(inference_trace.log_weight)
            gradient = [partial(slope_mu_num), partial(intercept_mu_num), partial(log_std_num)]

            # p(z, x) where z is latents and x is ..
            diff = model_trace.log_weight - concrete(inference_trace.log_weight)
            elbo_estimates[sample] = diff
            gradient_estimates[sample,:] = diff * gradient
        end

        # average the estimates of the objective function and the gradient
        # produced by the samples to obtain reduced-variance estimates of both
        elbo_estimate = mean(elbo_estimates)
        grad_estimate = mean(gradient_estimates, 1)

        # print objective function value, and current variational parameters
        println("iter: $iter, objective: $elbo_estimate, intercept_mu: $intercept_mu, slope_mu: $slope_mu, std: $(exp(log_std))")

        # update variational parameters using a gradient step
        rho = (step_a + iter) ^ (-step_b)
        slope_mu += rho * grad_estimate[1]
        intercept_mu += rho * grad_estimate[2]
        log_std += rho * grad_estimate[3]
    end
end

function linreg_demo()
    srand(1)
    xs = collect(linspace(-3, 3, 7))
    ys = -xs + 1
    println(xs)
    println(ys)
    # intercept should be mean 1, slope should be mean -1
    linear_regression_variational_inference(xs, ys)
end
linreg_demo()
