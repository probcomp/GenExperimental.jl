# --- generic neural network stuff --- #

abstract AmortizedInference

function sigmoid{T}(val::T)
    1.0 ./ (1.0 + exp(-val))
end

function scale_coordinate{T}(x::T)
    (x - 50.) / 100.
end

function unscale_coordinate{T}(x::T)
    x * 100. + 50.
end

function write_neural_network(parameters::Dict, fname::String)
    println("writing neural network to $fname..")
    json = JSON.json(parameters)
    open(fname, "w") do f
        write(f, json)
    end
end

#function render_neural_network(parameters::Dict, fname::String)
    #plt[:figure]()
    #W_hidden = parameters["W-hidden"]
    #plt[:imshow](W_hidden, interpolation="none", cmap="Greys")
    #plt[:tight_layout]()
    #plt[:savefig](fname)
#end

function convert(::Type{Matrix{Float64}}, arr::Array{Any,1})
    # JSON stores the matrix in column major format
    num_cols = length(arr)
    num_rows = length(arr[1])
    mat = zeros(num_rows, num_cols)
    for i=1:num_rows
        for j=1:num_cols
            mat[i, j] = arr[j][i]
        end
    end
    mat
end

function convert(::Type{Vector{Float64}}, arr::Array{Any,1})
    num_rows = length(arr)
    vec = zeros(num_rows)
    for i=1:num_rows
        vec[i] = arr[i]
    end
    vec
end


# --- neural network for predicting waypoint given measurements --- # 

immutable WaypointNetwork <: AmortizedInference
 
    # Number of samples to use for gradient estimation
    minibatch_size::Int

    # Number of samples to use for objective function evaluation
    num_eval::Int

    # Number of ADAM steps
    max_iter::Int

    # Number of observation time-steps to take as input
    max_t::Int

    # Number of hidden units
    num_hidden::Int
end

function inference_program(T::AbstractTrace, features::Array{Float64,1},
                           inference::WaypointNetwork)
    num_hidden = inference.num_hidden
    W_hidden = zeros(num_hidden, length(features)) ~ "W-hidden"
    b_hidden = zeros(num_hidden) ~ "b-hidden"
    W_output_x_mu = zeros(num_hidden) ~ "W-output-x-mu"
    b_output_x_mu = 0.0 ~ "b-output-x-mu"
    W_output_x_log_std = zeros(num_hidden) ~ "W-output-x-log-std"
    b_output_x_log_std = 0.0 ~ "b-output-x-log-std"
    W_output_y_mu = zeros(num_hidden) ~ "W-output-y-mu"
    b_output_y_mu = 0.0 ~ "b-output-y-mu"
    W_output_y_log_std =  zeros(num_hidden) ~ "W-output-y-log-std"
    b_output_y_log_std = 0.0 ~ "b-output-y-log-std"
    hidden = sigmoid(W_hidden * features + b_hidden)
    x_mu = (W_output_x_mu' * hidden + b_output_x_mu)[1]
    x_std = exp(W_output_x_log_std' * hidden + b_output_x_log_std)[1]
    output_x = normal(x_mu, x_std) ~ "output-x"
    y_mu = (W_output_y_mu' * hidden + b_output_y_mu)[1]
    y_std = exp(W_output_y_log_std' * hidden + b_output_y_log_std)[1]
    output_y = normal(y_mu, y_std) ~ "output-y"
end

function initialize_inference_parameters(inference::WaypointNetwork)
    num_hidden = inference.num_hidden
    num_features = inference.max_t * 2
    parameters = Dict()
    parameters["W-hidden"] = randn(num_hidden, num_features) / num_hidden
    parameters["b-hidden"] = randn(num_hidden) / num_hidden
    parameters["W-output-x-mu"] = randn(num_hidden) / num_hidden
    parameters["b-output-x-mu"] = randn()
    parameters["W-output-x-log-std"] = randn(num_hidden) / num_hidden
    parameters["b-output-x-log-std"] = randn()
    parameters["W-output-y-mu"] = randn(num_hidden) / num_hidden
    parameters["b-output-y-mu"] = randn()
    parameters["W-output-y-log-std"] = randn(num_hidden) / num_hidden
    parameters["b-output-y-log-std"] = randn()
    return parameters
end

function construct_inference_input(inference::WaypointNetwork, model_trace::AbstractTrace)
    xs = map((j) -> value(model_trace, "x$j"), 1:inference.max_t)
    ys = map((j) -> value(model_trace, "y$j"), 1:inference.max_t)
    scale_coordinate(vcat(xs, ys))
end

function constrain_inference_program!(inference::WaypointNetwork,
                                      inference_trace::AbstractTrace, model_trace::AbstractTrace)
    @assert value(model_trace, "use-waypoint")
    if hasvalue(inference_trace, "output-x")
        delete!(inference_trace, "output-x")
    end
    if hasvalue(inference_trace, "output-y")
        delete!(inference_trace, "output-y")
    end
    waypoint = value(model_trace, "waypoint")
    constrain!(inference_trace, "output-x", scale_coordinate(waypoint.x))
    constrain!(inference_trace, "output-y", scale_coordinate(waypoint.y))
end

function neural_network_predict(inference::WaypointNetwork, parameters::Dict, model_trace::Trace)
    features = construct_inference_input(inference, model_trace)
    trace = Trace()
    for key in keys(parameters)
        intervene!(trace, key, parameters[key])
    end
    propose!(trace, "output-x")
    propose!(trace, "output-y")
    inference_program(trace, features, inference)
    output_x = unscale_coordinate(value(trace, "output-x"))
    output_y = unscale_coordinate(value(trace, "output-y"))
    (Point(output_x, output_y), score(trace))
end

function load_waypoint_neural_network(fname::String)
    println("loading neural network from $fname..")
    json = open(fname, "r") do f
        readstring(f)
    end
    data = JSON.parse(json)
    parameters = Dict()
    parameters["W-hidden"] = convert(Matrix{Float64}, data["W-hidden"])
    parameters["b-hidden"] = convert(Vector{Float64}, data["b-hidden"])
    for s in ["x", "y"]
        parameters["W-output-$s-mu"] = convert(Vector{Float64}, data["W-output-$s-mu"])
        parameters["b-output-$s-mu"] = data["b-output-$s-mu"]
        parameters["W-output-$s-log-std"] = convert(Vector{Float64}, data["W-output-$s-log-std"])
        parameters["b-output-$s-log-std"] = data["b-output-$s-log-std"]
    end
    return parameters 
end


# --- neural network for predicting use-waypoint given measurements --- # 

immutable UseWaypointNetwork <: AmortizedInference

    # Number of samples to use for gradient estimation
    minibatch_size::Int

    # Number of samples to use for objective function evaluation
    num_eval::Int

    # Number of ADAM steps
    max_iter::Int

    # Number of observation time-steps to take as input
    max_t::Int

    # Number of hidden units
    num_hidden::Int
end

function inference_program(T::AbstractTrace, features::Array{Float64,1},
                           inference::UseWaypointNetwork)
	num_hidden = inference.num_hidden
	# predict whether there is a waypoint or not
	W_hidden = zeros(num_hidden, length(features)) ~ "bool-W-hidden"
	b_hidden = zeros(num_hidden) ~ "bool-b-hidden"
	W_output = zeros(num_hidden) ~ "bool-W-output"
	b_output = 0.0 ~ "bool-b-output"
	hidden = sigmoid(W_hidden * features + b_hidden)
	output_prob = sigmoid(W_output' * hidden + b_output)[1]
	use_waypoint = flip(output_prob) ~ "use-waypoint"
end

function initialize_inference_parameters(inference::UseWaypointNetwork)
    num_hidden = inference.num_hidden
    num_features = inference.max_t * 2 #+ 2 # start.x and start.y
    parameters = Dict()
    parameters["bool-W-hidden"] = randn(num_hidden, num_features) / num_hidden
    parameters["bool-b-hidden"] = randn(num_hidden) / num_hidden
    parameters["bool-W-output"] = randn(num_hidden) / num_hidden
    parameters["bool-b-output"] = randn()
    return parameters
end

function construct_inference_input(inference::UseWaypointNetwork, model_trace::AbstractTrace)
    xs = map((j) -> value(model_trace, "x$j"), 1:inference.max_t)
    ys = map((j) -> value(model_trace, "y$j"), 1:inference.max_t)
    scale_coordinate(vcat(xs, ys))
end

function constrain_inference_program!(inference::UseWaypointNetwork,
                                      inference_trace::AbstractTrace, model_trace::AbstractTrace)
    use_waypoint = value(model_trace, "use-waypoint")
    if hasvalue(inference_trace, "use-waypoint")
        delete!(inference_trace, "use-waypoint")
    end
    constrain!(inference_trace, "use-waypoint", use_waypoint)
end

function neural_network_predict(inference::UseWaypointNetwork, parameters::Dict, model_trace::Trace)
    features = construct_inference_input(inference, model_trace)
    trace = Trace()
    for key in keys(parameters)
        intervene!(trace, key, parameters[key])
    end
    propose!(trace, "use-waypoint")
    inference_program(trace, features, inference)
    use_waypoint = value(trace, "use-waypoint")
    (use_waypoint, score(trace))
end

function load_use_waypoint_neural_network(fname::String)
    println("loading neural network from $fname..")
    json = open(fname, "r") do f
        readstring(f)
    end
    data = JSON.parse(json)
    parameters = Dict()
    parameters["bool-W-hidden"] = convert(Matrix{Float64}, data["bool-W-hidden"])
    parameters["bool-b-hidden"] = convert(Vector{Float64}, data["bool-b-hidden"])
    parameters["bool-W-output"] = convert(Vector{Float64}, data["bool-W-output"])
    parameters["bool-b-output"] = data["bool-b-output"]
    return parameters 
end


# --- generic training procedure --- #

immutable ADAMParameters
    alpha::Float64
    beta_1::Float64
    beta_2::Float64
    epsilon::Float64
end

function adam_optimize!(objective::Function, gradient::Function,
                         theta::Dict, params::ADAMParameters, max_iter::Int)
    # gradient ascent
    # theta is a dictionary mapping Strings to scalars or arrays
    m = Dict{Any,Any}()
    v = Dict{Any,Any}()
    for key in keys(theta)
        m[key] = 0. * theta[key]
        v[key] = 0. * theta[key]
    end
    for t=2:max_iter
        g = gradient(t, theta)
        for key in keys(theta)
            m[key] = params.beta_1 * m[key] + (1. - params.beta_1) * g[key]
            v[key] = params.beta_2 * v[key] + (1. - params.beta_2) * (g[key] .* g[key])
            mhat = m[key] / (1. - params.beta_1^t)
            vhat = v[key] / (1. - params.beta_2^t)
            theta[key] += params.alpha * mhat ./ (sqrt(vhat + params.epsilon))
        end
        f = objective(t, theta)
    	println("f: $f")
    end
end

function train(inference::AmortizedInference, model_trace_generator::Function)
    theta = initialize_inference_parameters(inference)
    train(inference, model_trace_generator, theta)
end

function train(inference::AmortizedInference, model_trace_generator::Function, theta::Dict)

    function render(trace::Trace, fname::String) 
        camera_location = [50., -30., 120.]
        camera_look_at = [50., 50., 0.]
        light_location = [50., 50., 150.]
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 1
        frame.num_threads = 4
        render_trace(frame, trace)
        finish(frame, fname)
        println(fname)
    end

    # optimization algorithm parameters
    adam_params = ADAMParameters(0.001, 0.9, 0.999, 1e-8)

    # gradient function
    function single_gradient_sample(t::Int, theta::Dict)
        local gradient = Dict{Any,Any}()
        for key in keys(theta)
            gradient[key] = 0. * theta[key]
        end
        #println("minibatch sample number $i")
        model_trace = model_trace_generator()
        @assert value(model_trace, "measurement_noise") == 1.
        @assert value(model_trace, "start").x == 90.
        #render(model_trace, "neural_inference_debugging/training_trace_$i.png")
        inference_trace = DifferentiableTrace()
        for key in keys(theta)
            parametrize!(inference_trace, key, theta[key])
        end
        constrain_inference_program!(inference, inference_trace, model_trace)
        input = construct_inference_input(inference, model_trace)
        inference_program(inference_trace, input, inference)
        backprop(inference_trace)
        for key in keys(gradient)
            gradient[key] += derivative(inference_trace, key)
        end
        return gradient
    end
    function gradient(t::Int, theta::Dict)
        total_gradient = Dict{Any,Any}()
        for key in keys(theta)
            total_gradient[key] = 0. * theta[key]
        end
        gradients = pmap((i) -> single_gradient_sample(t, theta), 1:inference.minibatch_size)
        for key in keys(theta)
            for gradient in gradients
                total_gradient[key] += gradient[key]
            end
        end
        return total_gradient
    end

    # objective function
    function single_objective_sample(t::Int, inference_trace::Trace)
        model_trace = model_trace_generator()
        @assert value(model_trace, "measurement_noise") == 1.
        @assert value(model_trace, "start").x == 90.
        constrain_inference_program!(inference, inference_trace, model_trace)
        input = construct_inference_input(inference, model_trace)
        reset_score(inference_trace)
        inference_program(inference_trace, input, inference)
        return score(inference_trace)

    end
    function objective(t::Int, theta::Dict)
        objective = 0.0
        inference_trace = Trace()
        for key in keys(theta)
            intervene!(inference_trace, key, theta[key])
        end
        return sum(pmap((i) -> single_objective_sample(t, inference_trace), 1:inference.num_eval))
    end

    # writes parameters
    # TODO add a stopping condition
    adam_optimize!(objective, gradient, theta, adam_params, inference.max_iter)

    return theta

end


# ---- neural network demo ---- #

function train_use_waypoint_network()

    # only predict using the first max_t observations
    max_t = 15

    function model_trace_generator()
        model_trace = generate_scene_a()
        intervene!(model_trace, "measurement_noise", 1.0)
        @assert hasconstraint(model_trace, "start")
        @assert value(model_trace, "start").x == 90.
        @assert value(model_trace, "start").y == 10.
        while true
            reset_score(model_trace)
            agent_model(model_trace)
            # reject until path planning succeeded
            !isinf(score(model_trace)) && break
        end
        return model_trace
    end

    max_iter = 10^2
    inference = UseWaypointNetwork(100, 100, max_iter, max_t, 50)
    parameters = train(inference, model_trace_generator)
    write_neural_network(parameters, "use_waypoint_network_scene_a_$(max_iter).json")
    #render_neural_network(parameters, "use_waypoint_network_scene_a_$(max_iter).png")
end

function train_waypoint_network()
    
    # only predict using the first max_t observations
    max_t = 15
    max_iter = 10^4
    num_hidden = 100
    minibatch_size = 60
    num_eval = 60
    inference = WaypointNetwork(minibatch_size, num_eval, max_iter, max_t, num_hidden)

    function model_trace_generator()
        model_trace = generate_scene_a()
        intervene!(model_trace, "measurement_noise", 1.0)
        @assert hasconstraint(model_trace, "start")
        @assert value(model_trace, "start").x == 90.
        @assert value(model_trace, "start").y == 10.
        while true
            reset_score(model_trace)
            agent_model(model_trace)
            # reject until path planning succeeded and use-waypoint = true
            !isinf(score(model_trace)) && value(model_trace, "use-waypoint") && break
        end
		@assert value(model_trace, "use-waypoint")
		optimized_path = get(value(model_trace, "optimized_path"))
		@assert optimized_path.start.x == value(model_trace, "start").x
		@assert optimized_path.start.y == value(model_trace, "start").y
		@assert optimized_path.goal.x == value(model_trace, "destination").x
		@assert optimized_path.goal.y == value(model_trace, "destination").y
		waypoint = value(model_trace, "waypoint")
		waypoint_index = value(model_trace, "waypoint-index")
		waypoint_path_point = optimized_path.points[waypoint_index]
		#println("waypoint: $waypoint, waypoint-index: $waypoint_index, waypoint-path-point: $waypoint_path_point")
		times = value(model_trace, "times")
    	xs = map((j) -> value(model_trace, "x$j"), 1:length(times))
    	ys = map((j) -> value(model_trace, "y$j"), 1:length(times))
		#for i=1:length(times)
			#println("i: $i, x: $(xs[i]), y: $(ys[i])")
		#end
		@assert waypoint_path_point.x == waypoint.x
		@assert waypoint_path_point.y == waypoint.y
        return model_trace
    end

    #parameters = load_waypoint_neural_network("waypoint_network_scene_a_100.json")
    #parameters = train(inference, model_trace_generator, parameters)
    parameters = train(inference, model_trace_generator)
    write_neural_network(parameters, "waypoint_network_scene_a_$(max_iter).json")
    #render_neural_network(parameters, "waypoint_network_scene_a_$(max_iter).png")

    # render predictions of the network for our example dataset
    synthetic_data_trace = generate_scene_a()
    intervene!(synthetic_data_trace, "use-waypoint", true)
    intervene!(synthetic_data_trace, "measurement_noise", 1.0)
    intervene!(synthetic_data_trace, "waypoint", Point(55.,8.))
    intervene!(synthetic_data_trace, "destination", Point(70., 90.))
    agent_model(synthetic_data_trace)
    num_predictions = 60
    camera_location = [50., -30., 120.]
    camera_look_at = [50., 50., 0.]
    light_location = [50., 50., 150.]
    frame = PovrayRendering(camera_location, camera_look_at, light_location)
    frame.quality = 1
    frame.num_threads = 4
    delete!(synthetic_data_trace, "waypoint")
    delete!(synthetic_data_trace, "destination")
    delete!(synthetic_data_trace, "optimized_path")
    render_trace(frame, synthetic_data_trace)
    println("neural network waypoint predictions:")
    for j=1:num_predictions
        (waypoint, _) = neural_network_predict(inference, parameters, synthetic_data_trace)
        println(waypoint)
        render_waypoint(frame, waypoint)
    end
    finish(frame, "neural_inference_debugging/waypoint_predictions_$(num_hidden)_$(max_iter).png")
end


# ----- metropolis-hastings infernece ---- # 

function mh_inference(trace::Trace, num_iter::Int)
    @assert !hasvalue(trace, "tree") # don't want to copy this huge data structure
    @assert hasvalue(trace, "x1")
    trace = deepcopy(trace)
    agent_model(trace)

    # MH steps
    for iter=1:num_iter
        proposal_trace = deepcopy(trace)
        reset_score(proposal_trace)
        agent_model(proposal_trace)
        if log(rand()) < score(proposal_trace) - score(trace)
            trace = proposal_trace
        end
    end
    trace
end

function mh_neural_inference(trace::Trace, num_iter::Int,
                             inference::WaypointNetwork, network_params::Dict)
    @assert !hasvalue(trace, "tree")
    @assert hasvalue(trace, "x1")
    trace = deepcopy(trace)
    (waypoint, prev_proposal_score) = neural_network_predict(inference, network_params, trace)
    constrain!(trace, "waypoint", waypoint)
    agent_model(trace)
    

    # MH steps
    for iter=1:num_iter
        proposal_trace = deepcopy(trace)
        delete!(proposal_trace, "waypoint")
        reset_score(proposal_trace)
        (waypoint, proposal_score) = neural_network_predict(inference, network_params, trace)
        constrain!(proposal_trace, "waypoint", waypoint)
        agent_model(proposal_trace)
         if log(rand()) < score(proposal_trace) - score(trace) + prev_proposal_score - proposal_score
            trace = proposal_trace
            prev_proposal_score = proposal_score
        end
    end
    trace
end
