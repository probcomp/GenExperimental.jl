using Gen
using PyPlot

# a generative model for linear regression,
# written as a Gen probabilistic program
function linear_regression(T::Trace, xs::Array{Float64,1})
    slope = normal(0.0, 1.0) ~ "slope"
    intercept = normal(0.0, 1.0) ~ "intercept"
    ys = Array{Float64, 1}(length(xs))
    for i=1:length(xs)
        y_mean = intercept + slope * xs[i]
        ys[i] = normal(y_mean, 1.0) ~ "y$i"
    end
end

# a variational approxiation to the posterior,
# written as a Gen probabilistic program
function approximation{M,N,O,P}(T::DifferentiableTrace,
                                slope_mu::M, intercept_mu::N, 
                                slope_std::O, intercept_std::P)

    slope = normal(slope_mu, slope_std) ~ "slope"
    intercept = normal(intercept_mu, intercept_std) ~ "intercept"
end

# procedure for optimizing the parameters of the variational approximation to
# match the posterior
function linear_regression_variational_inference(xs::Array{Float64,1},
                                                 ys::Array{Float64,1},
                                                 max_iter::Int)

    # optimization parameters
    iters = 300
    num_samples = 1000
    step_a = 2000.0
    step_b = 0.75

    # initial values for variational parameters
    # (the parameters that are being optimized)
    slope_mu = 0.0
    intercept_mu = 0.0
    log_slope_std = 0.0
    log_intercept_std = 0.0

    elbo_estimates_tracked = Array{Float64,1}(max_iter)

    # gradient descent to minimize the difference between the approximation and
    # the posterior
    for iter in 1:max_iter

        # estimate the objective function (ELBO and its gradient with respect
        # to variational parameters using many runs of the approximation program
        elbo_estimates = Array{Float64,1}(num_samples)
        gradient_estimates = Array{Float64,2}(num_samples,4)

        for sample=1:num_samples

            # make the variational parameters GenNums, we can compute
            # derivatives with respect to them
            tape = Tape()
            slope_mu_num = GenNum(slope_mu, tape)
            intercept_mu_num = GenNum(intercept_mu, tape)
            log_slope_std_num = GenNum(log_slope_std, tape)
            log_intercept_std_num = GenNum(log_intercept_std, tape)

            # run the approximate program, treating 'slope' and 'intercept' as
            # outputs
            inference_trace = DifferentiableTrace(tape)
            inference_trace.outputs = Set{String}(["slope", "intercept"])
            approximation(inference_trace, 
                          slope_mu_num, intercept_mu_num,
                          exp(log_slope_std_num), exp(log_intercept_std_num))

            # add constraints to model trace so the model score can be computed
            # (using experimental syntactic sugars)
            model_trace = Trace()
            @in model_trace <= inference_trace begin
                for (i, y) in enumerate(ys)
                    # model_trace["y$i"] = y
                    @constrain("y$i", y) 
                end
                # model_trace["slope"] = inference_trace["slope"]
                @constrain("slope" <= "slope") 
                @constrain("intercept" <= "intercept")
            end
         
            # run model program on the constrained trace
            linear_regression(model_trace, xs)

            # differentiate the inference score with respect to the variational
            # parameters
            backprop(inference_trace.log_weight)
            gradient = [partial(slope_mu_num),
                        partial(intercept_mu_num),
                        partial(log_slope_std_num),
                        partial(log_intercept_std_num)]

            # estimate ELBO objective function and gradient
            diff = model_trace.log_weight - concrete(inference_trace.log_weight)
            elbo_estimates[sample] = diff
            gradient_estimates[sample,:] = diff * gradient
        end

        # average the estimates of the ELBO objective function and the gradient
        # across many samples to reduce their variance
        elbo_estimate = mean(elbo_estimates)
        elbo_estimates_tracked[iter] = elbo_estimate
        grad_estimate = mean(gradient_estimates, 1)

        # print objective function value, and current variational parameters
        #println("iter: $iter, objective: $elbo_estimate, intercept_mu: $intercept_mu, slope_mu: $slope_mu, slope_std: $(exp(log_slope_std)), intercept_std: $(exp(log_intercept_std))")

        # update variational parameters using a estimated gradient
        step_size = (step_a + iter) ^ (-step_b)
        slope_mu += step_size * grad_estimate[1]
        intercept_mu += step_size * grad_estimate[2]
        log_slope_std += step_size * grad_estimate[3]
        log_intercept_std += step_size * grad_estimate[4]
    end

    final_params = (slope_mu, intercept_mu, log_slope_std, log_intercept_std)
    return (final_params, elbo_estimates_tracked)
end

# sampling from exact partial posterior
using Distributions
function linreg_exact_posterior(x::Array{Float64,1}, y::Array{Float64,1}, prior_std::Float64, noise_std::Float64)
    # intercept comes first, then slope
    n = length(x)
    @assert n == length(y)
    S0 = prior_std * prior_std * eye(2)
    phi = hcat(ones(n), x)
    @assert size(phi) == (n, 2)
    noise_var = noise_std * noise_std
    noise_precision = 1./noise_var
    SN = inv(inv(S0) + noise_precision * (phi' * phi))
    mN = SN * ((inv(S0) * zeros(2)) + noise_precision * (phi' * y))
    # return the mean vector and covariance matrix
    (mN, SN)
end

immutable ExactLinregSampler
    dist::MvNormal
    function ExactLinregSampler(prior_std::Float64, noise_std::Float64, x::Array{Float64,1}, y::Array{Float64,1})
        (mN, SN) = linreg_exact_posterior(x, y, prior_std, noise_std)
        dist = MvNormal(mN, SN)
        new(dist)
    end
end

function simulate(sampler::ExactLinregSampler)
    intercept_and_slope = rand(sampler.dist)
    log_weight = logpdf(sampler.dist, intercept_and_slope)
    (intercept_and_slope, log_weight)
end

function regenerate(sampler::ExactLinregSampler, intercept_and_slope::Array{Float64,1})
    @assert length(intercept_and_slope) == 2
    logpdf(sampler.dist, intercept_and_slope)
end

function render_trace_line(trace::Any, ax, xlim, ylim) 
    line = xlim * trace["slope"] + trace["intercept"]
    ax[:plot](xlim, line, color="black", alpha=0.1)
    ax[:set_xlabel]("x")
    ax[:set_ylabel]("y")
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function render_trace_scatter(trace::Any, ax, xlim, ylim) 
    ax[:scatter]([trace["slope"]], [trace["intercept"]], color="black", alpha=0.1, s=3)
    ax[:set_xlabel]("slope")
    ax[:set_ylabel]("intercept")
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function render_dataset(xs, ys, ax, xlim, ylim)
    ax[:scatter](xs, ys, color="white", alpha=1.0, zorder=100, edgecolor="blue", lw=2, s=50)
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function aide(params, exact_sampler)
    num_samples = 100
    estimates = Array{Float64,1}(num_samples)
    for i=1:num_samples

        # sample from the exact posterior
        (exact_intercept, exact_slope), exact_log_weight = simulate(exact_sampler)

        # regenerate from the variational approxiation (log_density)
        trace = DifferentiableTrace(Tape())
        trace["slope"] = exact_slope
        trace["intercept"] = exact_intercept
        approximation(trace, params[1], params[2], exp(params[3]), exp(params[4]))
        variational_log_weight = trace.log_weight

        exact_to_variational_log_weight = exact_log_weight - variational_log_weight

        # sample from the variational approximation
        trace = DifferentiableTrace(Tape())
        approximation(trace, params[1], params[2], exp(params[3]), exp(params[4]))
        (variational_intercept, variational_slope) = (trace["intercept"], trace["slope"])
        variational_log_weight = trace.log_weight
        
        # regenerate from the exact posterior
        exact_log_weight = regenerate(exact_sampler, [variational_intercept, variational_slope])
        
        variational_to_exact_log_weight = variational_log_weight - exact_log_weight

        # unbiased estimate of symmetrized KL divergence
        estimates[i] = concrete(exact_to_variational_log_weight) + concrete(variational_to_exact_log_weight)
    end
    mean(estimates)
end

function linreg_demo()

    # synthetic dataset
    srand(1)
    xs = collect(linspace(-3, 3, 7))
    ys = -xs + 1 + randn(length(xs)) * 0.2
    
    # parameters of model (TODO fix them in the program itself for the demo)
    prior_mu = 0.0
    prior_std = 1.0
    noise_std = 0.2

    plot_dir = "bbvi_plots"
    num_samples = 1000

    # limits for the data and line plots 
    (xlim, ylim) = ([-5, 5], [-5, 5])

    # plot samples from the prior
    traces = []
    for i=1:num_samples
        trace = Trace()
        linear_regression(trace, xs)
        push!(traces, trace)
    end
    plt[:figure](figsize=(6,3))
    plt[:subplot](1, 2, 1)
    ax = plt[:gca]()
    for trace in traces[1:100]
        render_trace_line(trace, ax, xlim, ylim)
    end
    render_dataset(xs, ys, ax, xlim, ylim)
    plt[:subplot](1, 2, 2)
    ax = plt[:gca]()
    slopes = map((t) -> t["slope"], traces)
    intercepts = map((t) -> t["intercept"], traces)
    ax[:set_xlim]([-3, 3])
    ax[:set_ylim]([-3, 3])
    plt[:scatter](slopes, intercepts, s=3, alpha=0.5, color="black")
    ax[:set_xlabel]("Slope")
    ax[:set_ylabel]("Intercept")
    plt[:tight_layout]()
    plt[:savefig]("$plot_dir/prior_samples.png")

    # plot samples from the posterior
    exact_sampler = ExactLinregSampler(1.0, 1.0, xs, ys)
    traces = []
    for i=1:num_samples
        trace = Trace()
        (intercept, slope), _ = simulate(exact_sampler)
        trace["slope"] = slope
        trace["intercept"] = intercept
        push!(traces, trace)
    end
    plt[:figure](figsize=(6,3))
    plt[:subplot](1, 2, 1)
    ax = plt[:gca]()
    for trace in traces[1:100]
        render_trace_line(trace, ax, xlim, ylim)
    end
    render_dataset(xs, ys, ax, xlim, ylim)
    plt[:subplot](1, 2, 2)
    ax = plt[:gca]()
    slopes = map((t) -> t["slope"], traces)
    intercepts = map((t) -> t["intercept"], traces)
    ax[:set_xlim]([-3, 3])
    ax[:set_ylim]([-3, 3])
    plt[:scatter](slopes, intercepts, s=3, alpha=0.5, color="black")
    ax[:set_xlabel]("Slope")
    ax[:set_ylabel]("Intercept")
    plt[:tight_layout]()
    plt[:savefig]("$plot_dir/posterior_samples.png")

    # plot the dataset
    plt[:figure](figsize=(3,3))
    ax = plt[:gca]()
    render_dataset(xs, ys, ax, xlim, ylim)
    ax[:set_xlabel]("x")
    ax[:set_ylabel]("y")
    plt[:tight_layout]()
    plt[:savefig]("$plot_dir/dataset.png")
    
    params_history = Array{Float64,2}(100, 4)
    elbos = []
    aides = []
    # plot samples from black box variational inference
    for max_iter=1:100
        println("max_iter: $max_iter")
        params, elbo_trace = linear_regression_variational_inference(xs, ys, max_iter)
        push!(elbos, elbo_trace[end])
        params_history[max_iter,:] = [params[1], params[2], params[3], params[4]] 
        push!(aides, aide(params, exact_sampler))
        println(elbo_trace[end])
        traces = []
        for i=1:num_samples
            trace = DifferentiableTrace(Tape())
            approximation(trace, params[1], params[2], exp(params[3]), exp(params[4]))
            push!(traces, trace)
        end
        plt[:figure](figsize=(15,3))

        plt[:subplot](1, 5, 1)
        ax = plt[:gca]()
        render_dataset(xs, ys, ax, xlim, ylim)
        for trace in traces[1:100]
            render_trace_line(trace, ax, xlim, ylim)
        end
        ax[:set_xlim](xlim)
        ax[:set_ylim](xlim)

        plt[:subplot](1, 5, 2)
        ax = plt[:gca]()
        slopes = map((t) -> t["slope"], traces)
        intercepts = map((t) -> t["intercept"], traces)
        ax[:set_xlim]([-3, 3])
        ax[:set_ylim]([-3, 3])
        plt[:scatter](slopes, intercepts, s=3, alpha=0.5, color="black")
        ax[:set_xlabel]("Slope")
        ax[:set_ylabel]("Intercept")

        plt[:subplot](1, 5, 3)
        ax = plt[:gca]()
        ax[:plot](collect(1:max_iter), params_history[1:max_iter,1], label="slope mean", color="green", linestyle="-")
        ax[:plot](collect(1:max_iter), params_history[1:max_iter,3], label="slope uncertainty", color="green", linestyle="--")
        ax[:plot](collect(1:max_iter), params_history[1:max_iter,2], label="intercept mean", color="orange", linestyle="-")
        ax[:plot](collect(1:max_iter), params_history[1:max_iter,4], label="intercept uncertainty", color="orange", linestyle="--")
        plt[:legend](loc="upper right")
        ax[:set_xlim]((0, 100))
        ax[:set_ylim]((-2, 4))
        ax[:set_xlabel]("Iterations")
        ax[:set_ylabel]("Parameters")

        plt[:subplot](1, 5, 4)
        ax = plt[:gca]()
        plt[:plot](collect(1:max_iter), elbos, color="black")
        ax[:set_xlabel]("Iterations")
        ax[:set_ylabel]("Objective")
        ax[:set_xlim]((0, 100))
        ax[:set_ylim]((-50, -5))

        plt[:subplot](1, 5, 5)
        ax = plt[:gca]()
        plt[:plot](collect(1:max_iter), aides, color="black")
        ax[:set_xlabel]("Iterations")
        ax[:set_ylabel]("AIDE estimate")
        ax[:set_xlim]((0, 100))
        ax[:set_ylim]((0, 30))

        plt[:tight_layout]()
        plt[:savefig]("$plot_dir/variational_$max_iter.png")
    end
    

    # intercept should be mean 1, slope should be mean -1
    #linear_regression_variational_inference(xs, ys)
end
linreg_demo()

















