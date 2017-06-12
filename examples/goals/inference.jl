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

abstract AmortizedInference

immutable NeuralNetworkInference <: AmortizedInference
    neural_network::Function

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

inference_program(inference::NeuralNetworkInference) = inference.neural_network

function initialize_inference_parameters(inference::NeuralNetworkInference)
    num_hidden = inference.num_hidden
    num_features = inference.max_t * 2 #+ 2 # start.x and start.y
    parameters = Dict()
    parameters["bool-W-hidden"] = randn(num_hidden, num_features) / num_hidden
    parameters["bool-b-hidden"] = randn(num_hidden) / num_hidden
    parameters["bool-W-output"] = randn(num_hidden) / num_hidden
    parameters["bool-b-output"] = randn()

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

function construct_inference_input(inference::NeuralNetworkInference, model_trace::AbstractTrace)
    #start = value(model_trace, "start")
    xs = map((j) -> value(model_trace, "x$j"), 1:inference.max_t)
    ys = map((j) -> value(model_trace, "y$j"), 1:inference.max_t)
    #scale_coordinate(vcat([start.x, start.y], xs, ys))
    scale_coordinate(vcat(xs, ys))
end

function constrain_inference_program!(inference::NeuralNetworkInference,
                                      inference_trace::AbstractTrace, model_trace::AbstractTrace)
    use_waypoint = value(model_trace, "use-waypoint")
    if hasvalue(inference_trace, "use-waypoint")
        delete!(inference_trace, "use-waypoint")
    end
    if hasvalue(inference_trace, "output-x")
        delete!(inference_trace, "output-x")
    end
    if hasvalue(inference_trace, "output-y")
        delete!(inference_trace, "output-y")
    end

    constrain!(inference_trace, "use-waypoint", use_waypoint)
    if use_waypoint
        waypoint = value(model_trace, "waypoint")
        constrain!(inference_trace, "output-x", scale_coordinate(waypoint.x))
        constrain!(inference_trace, "output-y", scale_coordinate(waypoint.y))
    end
end

@everywhere function neural_network_predict(inference::NeuralNetworkInference, parameters::Dict, model_trace::Trace)
    features = construct_inference_input(inference, model_trace)
    trace = Trace()
    for key in keys(parameters)
        intervene!(trace, key, parameters[key])
    end
    propose!(trace, "use-waypoint")
    propose!(trace, "output-x")
    propose!(trace, "output-y")
    inference.neural_network(trace, features, inference)
    use_waypoint = value(trace, "use-waypoint")
    if use_waypoint
        output_x = unscale_coordinate(value(trace, "output-x"))
        output_y = unscale_coordinate(value(trace, "output-y"))
    else
        output_x = NaN
        output_y = NaN
    end
    (use_waypoint, Point(output_x, output_y)), score(trace)
end


# TODO generate a module as output?
# and call it 'compile'
# NOTE: what we actually generate are parameters for a probabilistic program (what about variational autoencoder case?)
function train(inference::AmortizedInference, model_trace::Trace, model_program::Function)

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

    program = inference_program(inference)

    # initialize inference program parameters
    theta = initialize_inference_parameters(inference)

    # optimization algorithm parameters
    adam_params = ADAMParameters(0.001, 0.9, 0.999, 1e-8)

    # gradient function
    function gradient(t::Int, theta::Dict)
        gradient = Dict{Any,Any}()
        for key in keys(theta)
            gradient[key] = 0. * theta[key]
        end
        for i=1:inference.minibatch_size
            while true
                reset_score(model_trace)
                model_program(model_trace)
                !isinf(score(model_trace)) && break # reject until it has non-infinite score (really its -inf)
                # TODO: support rejection sampling in the model program
            end
            @assert value(model_trace, "measurement_noise") == 1.
            @assert value(model_trace, "start").x == 90.
            #render(model_trace, "training_trace_$i.png")
            inference_trace = DifferentiableTrace()
            for key in keys(theta)
                parametrize!(inference_trace, key, theta[key])
            end
    
            constrain_inference_program!(inference, inference_trace, model_trace)
            input = construct_inference_input(inference, model_trace)
            program(inference_trace, input, inference)
            backprop(inference_trace)
            for key in keys(gradient)
                gradient[key] += derivative(inference_trace, key)
            end
        end
        return gradient
    end

    # objective function
    function objective(t::Int, theta::Dict)
        objective = 0.0
        inference_trace = Trace()
        for key in keys(theta)
            intervene!(inference_trace, key, theta[key])
        end
        for i=1:inference.num_eval
            while true
                reset_score(model_trace)
                model_program(model_trace)
                !isinf(score(model_trace)) && break # reject until it has non-infinite score (really its -inf)
                # TODO support rejection sampling in the model program
            end
            constrain_inference_program!(inference, inference_trace, model_trace)
            input = construct_inference_input(inference, model_trace)
            reset_score(inference_trace)
            program(inference_trace, input, inference)
            objective += score(inference_trace)
        end
        objective
    end

    # writes parameters
    # TODO add a stopping condition
    adam_optimize!(objective, gradient, theta, adam_params, inference.max_iter)

    return theta

end


@everywhere function sigmoid{T}(val::T)
    1.0 ./ (1.0 + exp(-val))
end

# TODO instead of two separate neural networks, just write one probabilistic
# program that first runs one neural network to predict use-waypoint and then
# runs a second neural network if use-waypoint = true

@everywhere function neural_network(T::AbstractTrace, features::Array{Float64,1}, inference::NeuralNetworkInference)
    num_hidden = inference.num_hidden
    # predict whether there is a waypoint or not
    W_hidden = zeros(num_hidden, length(features)) ~ "bool-W-hidden"
    b_hidden = zeros(num_hidden) ~ "bool-b-hidden"
    W_output = zeros(num_hidden) ~ "bool-W-output"
    b_output = 0.0 ~ "bool-b-output"
    hidden = sigmoid(W_hidden * features + b_hidden)
    output_prob = sigmoid(W_output' * hidden + b_output)[1]
    use_waypoint = flip(output_prob) ~ "use-waypoint"

    if use_waypoint
        # predict the value of the waypoint using a separate neural network
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
end

@everywhere function scale_coordinate{T}(x::T)
    (x - 50.) / 100.
end

@everywhere function unscale_coordinate{T}(x::T)
    x * 100. + 50.
end

# ---- neural network demo ---- #

function train_neural_networks()
    trace = Trace()

    intervene!(trace, "measurement_noise", 1.0)

    # add trees
    intervene!(trace, "is-tree-1", true)
    intervene!(trace, "tree-1", Tree(Point(30, 20), 10.))
    intervene!(trace, "is-tree-2", true)
    intervene!(trace, "tree-2", Tree(Point(83, 80), 10.))
    intervene!(trace, "is-tree-3", true)
    intervene!(trace, "tree-3", Tree(Point(80, 40), 10.))

    # add walls
    wall_height = 30.
    intervene!(trace, "is-wall-1", true)
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 40., 2., wall_height))
    intervene!(trace, "is-wall-2", true)
    intervene!(trace, "wall-2", Wall(Point(60., 40.), 2, 40., 2., wall_height))
    intervene!(trace, "is-wall-3", true)
    intervene!(trace, "wall-3", Wall(Point(60.-15., 80.), 1, 15. + 2., 2., wall_height))
    intervene!(trace, "is-wall-4", true)
    intervene!(trace, "wall-4", Wall(Point(20., 80.), 1, 15., 2., wall_height))
    intervene!(trace, "is-wall-5", true)
    intervene!(trace, "wall-5", Wall(Point(20., 40.), 2, 40., 2., wall_height))
    boundary_wall_height = 2.
    intervene!(trace, "is-wall-6", true)
    intervene!(trace, "wall-6", Wall(Point(0., 0.), 1, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-7", true)
    intervene!(trace, "wall-7", Wall(Point(100., 0.), 2, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-8", true)
    intervene!(trace, "wall-8", Wall(Point(0., 100.), 1, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-9", true)
    intervene!(trace, "wall-9", Wall(Point(0., 0.), 2, 100., 2., boundary_wall_height))

    # prevent the program from adding new wall or trees
    intervene!(trace, "is-tree-4", false)
    intervene!(trace, "is-wall-10", false)
    
    # only predict using the first max_t observations
    max_t = 15

    # scene a (which has a fixed start)
    scene_a_trace = generate_scene_a()
    intervene!(scene_a_trace, "measurement_noise", 1.0)

    # scene b: change the walls to add a bottom passageway
    #wall_height = 30.
    #delete!(trace, "wall-1")
    #intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 15., 2., wall_height))
    #delete!(trace, "is-tree-10")
    #delete!(trace, "is-wall-10")
    #intervene!(trace, "is-wall-10", true)
    #intervene!(trace, "wall-10", Wall(Point(60.- 15, 40.), 1, 15., 2., wall_height))
    #intervene!(trace, "is-wall-11", false)
    #scene_b_trace = deepcopy(trace)

    for exponent in [2]#3, 4, 5, 6] # TODO train for longer..
        # TODO do with the other scene as well
        # TODO show that the two networks are different
        max_iter = 10^exponent
        inference = NeuralNetworkInference(neural_network, 100, 100, max_iter, max_t, 50)
        println("training scene a, $max_iter")
        parameters = train(inference, scene_a_trace, agent_model)
        write_neural_network(parameters, "network_scene_a_$(max_iter).json")
        render_neural_network(parameters, "network_scene_a_$(max_iter).png")
        #println("training scene b, $max_iter")
        #parameters = train(inference, scene_b_trace, agent_model)
        #write_neural_network(parameters, "network_scene_b_$(max_iter).json")
        #render_neural_network(parameters, "network_scene_b_$(max_iter).png")
    end

    return trace
end

function write_neural_network(parameters::Dict, fname::String)
    println("writing neural network to $fname..")
    json = JSON.json(parameters)
    open(fname, "w") do f
        write(f, json)
    end
end

function render_neural_network(parameters::Dict, fname::String)
    plt[:figure]()
    W_hidden = parameters["W-hidden"]
    plt[:imshow](W_hidden, interpolation="none", cmap="Greys")
    plt[:tight_layout]()
    plt[:savefig](fname)
end

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

function load_neural_network(fname::String)
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

# TODO needs to be re-written
function mh_neural_inference(scene_trace::Trace, measured_xs::Array{Float64,1},
                             measured_ys::Array{Float64,1}, t::Int, num_iter::Int,
                             network_params::Dict)

    # the input scene_trace is assumed to contain only the scene
    @assert !hasvalue(scene_trace, "destination")
    @assert !hasvalue(scene_trace, "optimized_path")
    @assert hasvalue(scene_trace, "x1")
    start = value(scene_trace, "start")

    model_score = -Inf
    trace = deepcopy(scene_trace)
    tries = 0
    prev_proposal_score = NaN
    while isinf(model_score)

        # sample initial destination from proposal (until planning succeeds)
        ((initial_use_waypoint, initial_waypoint), prev_proposal_score) = neural_network_predict(network_params, start, measured_xs, measured_ys)
    
        # sample initial model trace given initial values from neural network
        if hasvalue(trace, "use-waypoint")
            delete!(trace, "use-waypoint")
        end
        if hasvalue(trace, "waypoint")
            delete!(trace, "waypoint")
        end
        constrain!(trace, "use-waypoint", initial_use_waypoint)
        if initial_use_waypoint
            constrain!(trace, "waypoint", initial_waypoint)
        end
        reset_score(trace)
        agent_model(trace)
        model_score = score(trace)
        tries += 1
    end

    # MH steps
    for iter=1:num_iter

        prev_model_score = score(trace)

        # sample from proposal
        proposal_trace = deepcopy(trace)
        reset_score(proposal_trace)
        ((use_waypoint_proposed, waypoint_proposed), proposal_score) = neural_network_predict(network_params, start, measured_xs, measured_ys)
        delete!(proposal_trace, "use-waypoint")
        delete!(proposal_trace, "waypoint")

        # sample path given proposal from prior, and evaluate model score
        constrain!(proposal_trace, "use-waypoint", use_waypoint_proposed) # the prior probability gets counted here
        if use_waypoint_proposed
            constrain!(proposal_trace, "waypoint", waypoint_proposed) # the prior probability gets counted here
        end
        agent_model(proposal_trace)

        proposed_model_score = score(proposal_trace)
        if log(rand()) < (score(proposal_trace) - proposal_score) - (prev_model_score - prev_proposal_score)
            trace = proposal_trace
            prev_proposal_score = proposal_score
        end
    end

    @assert hasvalue(trace, "destination")
    @assert hasvalue(trace, "optimized_path")
    @assert hasvalue(trace, "x1")
    trace
end


#srand(4)
#train_neural_networks()
