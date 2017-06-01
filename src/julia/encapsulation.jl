include("trace.jl")

# -------- generic encapsulation using default regenerator --------------- #

function encapsulate(program, output_names::Array{String,1})
    simulator = function(args...) # TODO use the type signature of the original program
        trace = Trace()
        program(trace, args...)
        map((name) -> trace.vals[name], output_names), trace.log_weight # TODO log_weight isnt right
    end
    regenerator = function(output_vals::Array, args...)
        trace = Trace()
        for (name, val) in zip(output_names, output_vals)
            trace.vals[name] = val
        end
        program(trace, args...)
        trace.log_weight
    end
    (simulator, regenerator)
end

# run it
linreg_simulator, linreg_regenerator = encapsulate(linear_regression, ["y1", "y2", "y3"])
outputs, log_weight = linreg_simulator(0.0, 2.0, [0.1, 0.2, 0.3])
println("$outputs, $log_weight")
log_weight = linreg_regenerator(outputs, 0.0, 2.0, [0.1, 0.2, 0.3])
println(log_weight)
@register_module(:linreg, linreg_simulator, linreg_regenerator)


# ---------- encapsulation using an inference module -------------------- #
# all arguments of program get along passed to the module
function encapsulate(program, output_names::Array{String,1},
                     inference_module, inferred_names::Array{String,1})
    inference_simulator, inference_regenerator = inference_module
    simulator = function(args...) # TODO use the type signature of the original program
        trace = Trace()
        program(trace, args...)
        inference_inputs = map((name) -> trace.vals[name], output_names)
        inference_outputs = map((name) -> trace.vals[name], inferred_names)
        inference_log_weight = inference_regenerator(inference_outputs, inference_inputs, args...)
        program_outputs = map((name) -> trace.vals[name], output_names)
        program_outputs, (trace.log_weight - inference_log_weight) # TODO trace.log_weight isnt right
    end
    regenerator = function(output_vals::Array, args...)
        inference_inputs = output_vals
        inference_outputs, inference_log_weight = inference_simulator(inference_inputs, args...)
        trace = Trace()
        # put outputs of program into program trace
        for (name, val) in zip(output_names, output_vals)
            trace.vals[name] = val
        end
        # put outputs of inference into program trace
        for (name, val) in zip(inferred_names, inference_outputs)
            trace.vals[name] = val
        end
        # run program, to collect log_weight
        program(trace, args...)
        trace.log_weight - inference_log_weight
    end
    (simulator, regenerator)
end

# use a RANSAC algorithm to predict slope and intercept
# TODO: add autodifferentiation and tune the noise of RANSAC using stochastic gradient descent :)
# TODO: also show a big third-party RANSAC implementation being used for the core, with output variance tuning using SGD

function ransac_core(xs::Array{Float64,1}, ys::Array{Float64,1},
                     iters::Int, subset_size::Int, eps::Float64)
    best_num_inliers::Int = -1
    best_slope::Float64 = NaN
    best_intercept::Float64 = NaN
    for i=1:iters
        subset = randperm(length(xs))[1:subset_size]
        # estimate slope and intercept using least squares
        A = hcat(xs, ones(length(xs)))
        slope, intercept = A\ys
        ypred = intercept + slope * xs
        num_inliers = sum(abs(ys - ypred) .< eps)
        if num_inliers > best_num_inliers
            best_slope, best_intercept = slope, intercept
        end
    end
    [best_slope, best_intercept]
end

function make_ransac_module(iters::Int, subset_size::Int, eps::Float64,
                            slope_std::Float64, intercept_std::Float64)
    function ransac_simulate(# output variables of the model program
                             ys::Array{Float64,1},
                             # parameters of the model program
                             prior_mu::Float64, prior_std::Float64, xs::Array{Float64,1})
        best_slope, best_intercept = ransac_core(xs, ys, iters, subset_size, eps)
        # add some noise to output
        output_slope, log_weight_slope = normal_simulate(best_slope, slope_std)
        output_intercept, log_weight_intercept = normal_simulate(best_intercept, intercept_std)
        [output_slope, output_intercept], log_weight_slope + log_weight_intercept
    end
    function ransac_regenerate(# inferred variables of the model program
                               slope_and_intercept::Array{Float64,1},
                               # output variables of the model program
                               ys::Array{Float64,1},
                               # parameters of the model program
                               prior_mu::Float64, prior_std::Float64, xs::Array{Float64,1})
        slope, intercept = slope_and_intercept
        best_slope, best_intercept = ransac_core(xs, ys, iters, subset_size, eps)
        # evaluate log density of output given core value
        log_weight_slope = normal_regenerate(intercept, best_slope, slope_std)
        log_weight_intercept = normal_regenerate(slope, best_intercept, intercept_std)
        log_weight_slope + log_weight_intercept
    end
    (ransac_simulate, ransac_regenerate)
end


# run it
linreg_simulator, linreg_regenerator = encapsulate(linear_regression, ["y1", "y2", "y3"],
                                                   make_ransac_module(10, 2, 0.1, 0.1, 0.1),
                                                   ["slope", "intercept"])
outputs, log_weight = linreg_simulator(0.0, 2.0, [0.1, 0.2, 0.3])
println("$outputs, $log_weight")
log_weight = linreg_regenerator(outputs, 0.0, 2.0, [0.1, 0.2, 0.3])
println(log_weight)
@register_module(:linreg, linreg_simulator, linreg_regenerator)

