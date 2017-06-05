using Gen
using PyPlot

function linear_regression(T::Trace, xs::Array{Float64,1})
    noise_std = 1.0
    prior_std = 1.0
    prior_mu = 0.0
    slope = normal(prior_mu, prior_std) ~ "slope"
    intercept = normal(prior_mu, prior_std) ~ "intercept"
    ys = Array{Float64, 1}(length(xs))
    for i=1:length(xs)
        y_mean = intercept + slope * xs[i]
        ys[i] = normal(y_mean, noise_std) ~ "y$i"
    end
end

function variational_approximation{M,N,O,P}(T::DifferentiableTrace,
                                            slope_mu::M, intercept_mu::N, 
                                            slope_std::O, intercept_std::P)
    slope = normal(slope_mu, slope_std) ~ "slope"
    intercept = normal(intercept_mu, intercept_std) ~ "intercept"
end

function linear_regression_variational_inference(xs::Array{Float64,1},
                                                 ys::Array{Float64,1},
                                                 max_iter::Int)

    # optimization parameters
    iters = 300
    num_samples = 1000 # 10000
    step_a = 1000.0
    step_b = 0.75

    # fixed model parameters

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

            # run the proposal program so that its score can be differentiated
            # with respect to its inputs (which are the variational parameters)
            tape = Tape()
            slope_mu_num = GenNum(slope_mu, tape)
            intercept_mu_num = GenNum(intercept_mu, tape)
            log_slope_std_num = GenNum(log_slope_std, tape)
            log_intercept_std_num = GenNum(log_intercept_std, tape)
            inference_trace = DifferentiableTrace(tape)
            inference_trace.outputs = Set{String}(["slope", "intercept"]) # TODO explain
            variational_approximation(inference_trace, 
                                      slope_mu_num, intercept_mu_num,
                                      exp(log_slope_std_num), exp(log_intercept_std_num))

            # add constraints to model trace so the model score can be computed
            # (using experimental syntactic sugars)
            model_trace = Trace()
            @in model_trace <= inference_trace begin
                for (i, y) in enumerate(ys)
                    @constrain("y$i", y)
                end
                @constrain("slope" <= "slope")
                @constrain("intercept" <= "intercept")
            end
        
            # run model program
            linear_regression(model_trace, xs)

            # differentiate the inference score with respect to the variational
            # parameters
            backprop(inference_trace.log_weight)
            gradient = [partial(slope_mu_num), partial(intercept_mu_num), partial(log_slope_std_num), partial(log_intercept_std_num)]

            # p(z, x) where z is latents and x is ..
            diff = model_trace.log_weight - concrete(inference_trace.log_weight)
            elbo_estimates[sample] = diff
            gradient_estimates[sample,:] = diff * gradient
        end

        # average the estimates of the objective function and the gradient
        # produced by the samples to obtain reduced-variance estimates of both
        elbo_estimate = mean(elbo_estimates)
        elbo_estimates_tracked[iter] = elbo_estimate
        grad_estimate = mean(gradient_estimates, 1)

        # print objective function value, and current variational parameters
        #println("iter: $iter, objective: $elbo_estimate, intercept_mu: $intercept_mu, slope_mu: $slope_mu, slope_std: $(exp(log_slope_std)), intercept_std: $(exp(log_intercept_std))")

        # update variational parameters using a gradient step
        rho = (step_a + iter) ^ (-step_b)
        slope_mu += rho * grad_estimate[1]
        intercept_mu += rho * grad_estimate[2]
        log_slope_std += rho * grad_estimate[3]
        log_intercept_std += rho * grad_estimate[4]
    end

    return (slope_mu, intercept_mu, log_slope_std, log_intercept_std), elbo_estimates_tracked
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
    ax[:scatter](xs, ys, color="red", alpha=0.7, zorder=100)
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
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
    for trace in traces
        render_trace_line(trace, ax, xlim, ylim)
    end
    plt[:subplot](1, 2, 2)
    ax = plt[:gca]()
    for trace in traces
        render_trace_scatter(trace, ax, [-3, 3], [-3, 3])
    end
    plt[:tight_layout]()
    plt[:savefig]("$plot_dir/prior_samples.png")

    # plot the dataset
    plt[:figure](figsize=(3,3))
    ax = plt[:gca]()
    render_dataset(xs, ys, ax, xlim, ylim)
    ax[:set_xlabel]("x")
    ax[:set_ylabel]("y")
    plt[:tight_layout]()
    plt[:savefig]("$plot_dir/dataset.png")
    

    elbos = []
    # plot samples from black box variational inference
    for max_iter=1:100
        println("max_iter: $max_iter")
        params, elbo_trace = linear_regression_variational_inference(xs, ys, max_iter)
        push!(elbos, elbo_trace[end])
        println(elbo_trace[end])
        traces = []
        for i=1:num_samples
            trace = DifferentiableTrace(Tape())
            variational_approximation(trace, params[1], params[2], exp(params[3]), exp(params[4]))
            push!(traces, trace)
        end
        plt[:figure](figsize=(9,3))
        plt[:subplot](1, 3, 1)
        ax = plt[:gca]()
        render_dataset(xs, ys, ax, xlim, ylim)
        for trace in traces[1:100]
            render_trace_line(trace, ax, xlim, ylim)
        end
        ax[:set_xlim](xlim)
        ax[:set_ylim](xlim)
        plt[:subplot](1, 3, 2)
        ax = plt[:gca]()
        slopes = map((t) -> t["slope"], traces)
        intercepts = map((t) -> t["intercept"], traces)
        ax[:set_xlim]([-3, 3])
        ax[:set_ylim]([-3, 3])
        plt[:scatter](slopes, intercepts, s=3, alpha=0.5, color="black")
        ax[:set_xlabel]("slope")
        ax[:set_ylabel]("intercept")
        plt[:subplot](1, 3, 3)
        ax = plt[:gca]()
        plt[:plot](collect(1:max_iter), elbos, color="black")
        ax[:set_xlabel]("Iterations")
        ax[:set_ylabel]("Objective")
        ax[:set_xlim]((0, 100))
        ax[:set_ylim]((-50, -5))
        plt[:tight_layout]()
        plt[:savefig]("$plot_dir/variational_$max_iter.png")
    end
    

    # intercept should be mean 1, slope should be mean -1
    #linear_regression_variational_inference(xs, ys)
end
linreg_demo()

















