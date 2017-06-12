@everywhere using Gen
@everywhere using Distributions
@everywhere using PyCall
import JSON
@pyimport matplotlib.patches as patches
using PyPlot

@everywhere include("path_planner.jl")

@everywhere function uniform_2d_simulate(xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    x, log_weight_x = uniform_simulate(xmin, xmax)
    y, log_weight_y = uniform_simulate(ymin, ymax)
    Point(x, y), log_weight_x + log_weight_y
end
@everywhere function uniform_2d_regenerate(point::Point, xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    log_weight_x = uniform_regenerate(point.x, xmin, xmax)
    log_weight_y = uniform_regenerate(point.y, ymin, ymax)
    log_weight_x + log_weight_y
end
@everywhere @register_module2(:uniform_2d, uniform_2d_simulate, uniform_2d_regenerate)
@everywhere export uniform_2d_simulate
@everywhere export uniform_2d_regenerate
@everywhere uniform_2d = (args...) -> (uniform_2d_simulate)(args...)[1]
@everywhere export uniform_2d

@everywhere function agent_model(T::Trace)

    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100
    scene = Scene(xmin, xmax, ymin, ymax)

    # add trees
    i = 1
    while true
        if (flip(0.7) ~ "is-tree-$i")
            location = uniform_2d(xmin, xmax, ymin, ymax)
            size = 15.
            add!(scene, Tree(location, size) ~ "tree-$i")
            i += 1
        else
            break
        end
    end

    # add walls
    wall_height = 30.
    wall_thickness = 2.
    i = 1
    while true
        if (flip(0.7) ~ "is-wall-$i")
            start = uniform_2d(xmin, xmax, ymin, ymax)
            orientation = flip(0.5) ? 1 : 2
            length = uniform(0.0, 100.0)
            add!(scene, Wall(start, orientation, length, wall_thickness, wall_height) ~ "wall-$i")
            i += 1
        else
            break
        end
    end

    # set starting location of the drone
    start = uniform_2d(xmin, xmax, ymin, ymax) ~ "start"

    # set destination of the drone
    destination = uniform_2d(xmin, xmax, ymin, ymax) ~ "destination"

    # plan path to goal, optionally using a waypoint
    planner_params = PlannerParams(2000, 3.0, 10000, 1.)
    local optimized_path::Nullable{Path}
    if (flip(0.5) ~ "use-waypoint")
        waypoint = uniform_2d(xmin, xmax, ymin, ymax) ~ "waypoint"
        tree1, path1, optimized_path1 = plan_path(start, waypoint, scene, planner_params)
        tree2, path2, optimized_path2 = plan_path(waypoint, destination, scene, planner_params)
        if isnull(optimized_path1) || isnull(optimized_path2)
            optimized_path = Nullable{Path}()
        else
            optimized_path = concatenate(get(optimized_path1), get(optimized_path2))
        end
    else
        tree, path, optimized_path = plan_path(start, destination, scene, planner_params)
    end

    if isnull(optimized_path)
        # no path found
        fail(T)
    else
        # walk the path at a constant speed, and record locations at times
        speed = 10. ~ "speed"
        times = collect(linspace(0.0, 15.0, 30)) ~ "times"
        locations = walk_path(get(optimized_path), speed, times) ~ "locations"

        # add measurement noise to the true locations
        measurement_noise = 8.0 ~ "measurement_noise"
        measurements = Array{Point,1}(length(times))
        for (i, loc) in enumerate(locations)
            measurements[i] = Point(normal(loc.x, measurement_noise) ~ "x$i", 
                                    normal(loc.y, measurement_noise) ~ "y$i")
        end
    end

    # record for rendering purposes
    optimized_path ~ "optimized_path"
    
    return nothing
end

@everywhere function mh_inference(trace::Trace, num_iter::Int)
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

@everywhere function mh_neural_inference(scene_trace::Trace, measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1}, t::Int, num_iter::Int,
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



include("povray_rendering.jl")

# ---- amortized inference ----- #

@everywhere function sigmoid{T}(val::T)
    1.0 ./ (1.0 + exp(-val))
end

# TODO instead of two separate neural networks, just write one probabilistic
# program that first runs one neural network to predict use-waypoint and then
# runs a second neural network if use-waypoint = true

@everywhere function neural_network(T::AbstractTrace, features::Array{Float64,1}, num_hidden::Int)
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

type TrainingParams
    # the number of time points to use as observations (currently a separate
    # network would need to be trained for each distinct number of
    # observations)
    max_t::Int 
    num_hidden::Int
    step_a::Float64
    step_b::Float64
    minibatch_size::Int
    max_iter::Int
end

@everywhere function neural_network_predict(parameters::Dict, start::Point, xs::Vector{Float64}, ys::Vector{Float64})
    features = scale_coordinate(vcat([start.x, start.y], xs, ys))
    trace = Trace()
    for key in keys(parameters)
        intervene!(trace, key, parameters[key])
    end
    propose!(trace, "use-waypoint")
    propose!(trace, "output-x")
    propose!(trace, "output-y")
    num_hidden = length(parameters["b-hidden"])
    neural_network(trace, features, num_hidden)
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

function initialize_network_parameters(num_hidden::Int, max_t::Int)
    num_features = max_t * 2 + 2 # start.x and start.y
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

function initialize_network_gradients(num_hidden::Int, max_t::Int)
    num_features = max_t * 2 + 2 # start.x and start.y
    gradients = Dict()
    gradients["bool-W-hidden"] = zeros(num_hidden, num_features)
    gradients["bool-b-hidden"] = zeros(num_hidden)
    gradients["bool-W-output"] = zeros(num_hidden)
    gradients["bool-b-output"] = 0.0

    gradients["W-hidden"] = zeros(num_hidden, num_features)
    gradients["b-hidden"] = zeros(num_hidden)
    gradients["W-output-x-mu"] = zeros(num_hidden)
    gradients["b-output-x-mu"] = 0.0
    gradients["W-output-x-log-std"] = zeros(num_hidden)
    gradients["b-output-x-log-std"] = 0.0
    gradients["W-output-y-mu"] = zeros(num_hidden)
    gradients["b-output-y-mu"] = 0.0
    gradients["W-output-y-log-std"] = zeros(num_hidden)
    gradients["b-output-y-log-std"] = 0.0
    return gradients
end

function update_gradients_network!(network_trace::DifferentiableTrace, model_trace::Trace, parameters::Dict, gradients::Dict)
    use_waypoint = value(model_trace, "use-waypoint")
    max_t = Int((size(parameters["W-hidden"])[2] - 2) / 2)
    start = value(model_trace, "start")
    xs = map((j) -> value(model_trace, "x$j"), 1:max_t)
    ys = map((j) -> value(model_trace, "y$j"), 1:max_t)
    features = scale_coordinate(vcat([start.x, start.y], xs, ys))

    num_hidden = length(parameters["b-hidden"])

    if hasvalue(network_trace, "use-waypoint")
        delete!(network_trace, "use-waypoint")
    end
    if hasvalue(network_trace, "output-x")
        delete!(network_trace, "output-x")
    end
    if hasvalue(network_trace, "output-y")
        delete!(network_trace, "output-y")
    end

    constrain!(network_trace, "use-waypoint", use_waypoint)
    if use_waypoint
        waypoint = value(model_trace, "waypoint")
        constrain!(network_trace, "output-x", scale_coordinate(waypoint.x))
        constrain!(network_trace, "output-y", scale_coordinate(waypoint.y))
    end

    reset_score(network_trace)
    for key in keys(parameters)
        if hasvalue(network_trace, key)
            delete!(network_trace, key)
        end
        parametrize!(network_trace, key, parameters[key])
    end

    neural_network(network_trace, features, num_hidden)
    backprop(network_trace)
    for key in keys(parameters)
        gradients[key] += derivative(network_trace, key)
    end
end

function update_network_parameters!(parameters::Dict, gradients::Dict, num_samples::Int, step_size::Float64)
    if num_samples > 0
        for key in keys(parameters)
            gradient = gradients[key] / num_samples 
            parameters[key] += step_size * gradient
        end
    end
end

function train_neural_network(model_trace::Trace, params::TrainingParams)

    # the input trace contains interventions and/or constraints that define the
    # fixed context
    num_features = params.max_t * 2 + 2 # start.x and start.y
    num_hidden = params.num_hidden

    network_parameters = initialize_network_parameters(params.num_hidden, params.max_t)
    network_trace = DifferentiableTrace()

    for iter=1:params.max_iter
        println("iter: $iter")

        objective_function = 0.0
        network_gradients = initialize_network_gradients(params.num_hidden, params.max_t)
        num_samples = 0

        for i=1:params.minibatch_size
            agent_model(model_trace)
            if isnull(value(model_trace, "optimized_path"))
                continue
            end

            update_gradients_network!(network_trace, model_trace, network_parameters, network_gradients)
            num_samples += 1

            objective_function += score(network_trace) / num_samples
        end

        println("objective function: $(objective_function / num_samples)")

        step_size = (params.step_a + iter) ^ (-params.step_b)
        update_network_parameters!(network_parameters, network_gradients, num_samples, step_size)
    end

    return network_parameters
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

    # scene a
    scene_a_trace = deepcopy(trace)

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

    for exponent in [5]#3, 4, 5, 6] # TODO train for longer..
        # TODO do with the other scene as well
        # TODO show that the two networks are different
        max_iter = 10^exponent
        training_params = TrainingParams(max_t, 50, 1000.0, 0.75, 1, max_iter)
        println("training scene a, $max_iter")
        network_parameters = train_neural_network(scene_a_trace, training_params)
        write_neural_network(network_parameters, "network_scene_a_$(max_iter).json")
        render_neural_network(network_parameters, "network_scene_a_$(max_iter).png")
        #println("training scene b, $max_iter")
        #network_parameters = train_neural_network(scene_b_trace, training_params)
        #write_neural_network(network_parameters, "network_scene_b_$(max_iter).json")
        #render_neural_network(network_parameters, "network_scene_b_$(max_iter).png")
    end

    return trace
end

function write_neural_network(network_parameters::Dict, fname::String)
    println("writing neural network to $fname..")
    json = JSON.json(network_parameters)
    open(fname, "w") do f
        write(f, json)
    end
end

function render_neural_network(network_parameters::Dict, fname::String)
    plt[:figure]()
    W_hidden = network_parameters["W-hidden"]
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

# ---- demo ---- #

function render_text_frame(text::String, fname::String)
    plt[:figure](figsize=(8,8))
    ax = plt[:gca]()
    plt[:text](0.5, 0.5, text, transform=ax[:transAxes], horizontalalignment="center", verticalalignment="center", fontsize=14)
    plt[:axis]("off")
    plt[:savefig](fname)
end

function generate_scene_a()
    trace = Trace()
    intervene!(trace, "is-tree-1", true)
    intervene!(trace, "tree-1", Tree(Point(30, 20), 10.))
    intervene!(trace, "is-tree-2", true)
    intervene!(trace, "tree-2", Tree(Point(83, 80), 10.))
    intervene!(trace, "is-tree-3", true)
    intervene!(trace, "tree-3", Tree(Point(80, 40), 10.))

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

    # add the drone starting position
    constrain!(trace, "start", Point(90., 10.))
    return trace
end

function generate_scene_b()
    trace = generate_scene_a()

    # change the walls to add a bottom passageway
    wall_height = 30.
    delete!(trace, "wall-1")
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 15., 2., wall_height))
    delete!(trace, "is-tree-10")
    delete!(trace, "is-wall-10")
    intervene!(trace, "is-wall-10", true)
    intervene!(trace, "wall-10", Wall(Point(60.- 15, 40.), 1, 15., 2., wall_height))
    intervene!(trace, "is-wall-11", false)
    return trace
end



function demo()

    camera_location = [50., -30., 120.]
    camera_look_at = [50., 50., 0.]
    light_location = [50., 50., 150.]

    function render(trace::Trace, fname::String) 
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 10
        frame.num_threads = 60
        render_trace(frame, trace)
        finish(frame, fname)
        println(fname)
    end

    function render(traces::Array{Trace,1}, fname::String, max_measurement_time::Int)
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 10
        frame.num_threads = 60
        render_traces(frame, traces, max_measurement_time)
        finish(frame, fname)
        println(fname)
    end

    # frame index
    f = 0
    
    trace = generate_scene_a()

    # add a known ground truth goal
    intervene!(trace, "measurement_noise", 1.0)
    intervene!(trace, "destination", Point(40., 60.))
    
    # show scene A, start, and goal position
    #render(trace, "frames/frame_$f.png") ; f += 1

    # run model and extract observations, render ground truth
    # three times
    for i=1:10
        # TODO replace with waypoint model
        agent_model(trace)
        #render(trace, "frames/frame_$f.png") ; f += 1
    end
    times = value(trace, "times")

    # fresh trace with a random goal location
    trace = generate_scene_a()
    agent_model(trace)
    delete!(trace, "optimized_path")
    for i=1:length(times)
        delete!(trace, "x$i")
        delete!(trace, "y$i")
    end
    #render(trace, "frames/frame_$f.png") ; f += 1

    # add observations (generate the observations synthetically in another trace)
    # make the synthetic data be a waypoint path (so that inference has an obvious effect)
    synthetic_data_trace = generate_scene_a()
    intervene!(synthetic_data_trace, "use-waypoint", true)
    intervene!(synthetic_data_trace, "measurement_noise", 1.0)
    intervene!(synthetic_data_trace, "waypoint", Point(55.,8.))
    intervene!(synthetic_data_trace, "destination", Point(70., 90.))
    agent_model(synthetic_data_trace)
    num_obs = 15
    measured_xs = map((i) -> value(synthetic_data_trace, "x$i"), 1:num_obs)
    measured_ys = map((i) -> value(synthetic_data_trace, "y$i"), 1:num_obs)
    
    # show observations in the trace with the random goal locaiton
    intervene!(trace, "times", value(synthetic_data_trace, "times"))
    for i=1:num_obs
        constrain!(trace, "x$i", measured_xs[i])
        constrain!(trace, "y$i", measured_ys[i])
    end
    #render(trace, "frames/frame_$f.png") ; f += 1
   
    # show some inference happening (just the clouds)
    num_particles = 60

    traces::Array{Trace, 1} = map((i) -> deepcopy(trace), 1:num_particles) # infer fork
    traces = pmap((trace) -> mh_inference(trace, 0), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 1), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 10), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 100), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 1000), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 10000), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1

    trace = generate_scene_b()
    render(trace, "frames/frame_$f.png") ; f += 1

    intervene!(trace, "times", value(synthetic_data_trace, "times"))
    for i=1:num_obs
        constrain!(trace, "x$i", measured_xs[i])
        constrain!(trace, "y$i", measured_ys[i])
    end
    render(trace, "frames/frame_$f.png") ; f += 1

    #render(trace, "frames/frame_$f.png") ; f += 1
    traces = map((i) -> deepcopy(trace), 1:num_particles) # infer fork
    traces = pmap((trace) -> mh_inference(trace, 0), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 1), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 10), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 100), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 1000), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 10000), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1

    # neural network experiments
    scene_a_trace = generate_scene_a()
    #scene_b_trace = generate_scene_b()
    for (scene_name, scene_trace) in [("a", scene_a_trace)]#[("a", scene_a_trace), ("b", scene_b_trace)]

        println("using neural network for scene: $scene_name")
        if !hasvalue(scene_trace, "times")
            intervene!(scene_trace, "times", times)
        end

        # show neural network predictions for both scenes
        # the neural network was trained for a specific number of observations (max_t
        max_iter = 10^5 # was 4 # TODO
        network_parameters = load_neural_network("network_scene_$(scene_name)_$(max_iter).json")
        max_t = Int((size(network_parameters["W-hidden"])[2] - 2) / 2)
    
        # constrain the prefix of observed data
        xs = measured_xs[1:max_t]
        ys = measured_ys[1:max_t]
        trace = deepcopy(scene_trace)
        for t=1:max_t
            constrain!(trace, "x$t", xs[t])
            constrain!(trace, "y$t", ys[t])
        end
        start = value(trace, "start")
    
        # show neural network predictions of the waypoint for our observed data set
        println("making predictions..")
        num_predictions = 60
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 1
        frame.num_threads = 4
        render_trace(frame, trace)
        for j=1:num_predictions
            (use_waypoint, waypoint), _ = neural_network_predict(network_parameters, start, xs, ys)
            render_destination(frame, use_waypoint ? waypoint : Point(randn(), randn())) # NOTE: it will be red but it is actually the waypoint
        end
        finish(frame, "frames/frame_$f.png") ; f += 1
    
        # use neural network as proposal and show trajectory predictions
        println("neural MH scene a, max_time: $max_t")
        @assert length(xs) == max_t
        for num_iter=0:100
            println("num_iter: $num_iter")
            particles::Array{Trace,1} = pmap((i) -> mh_neural_inference(trace, xs, ys, max_t, num_iter, network_parameters), 1:num_particles)
            render(particles, "frames/frame_$f.png", max_t) ; f += 1
        end
    end

end

# writes neural networks to files, and returns the model trace with the
# scene context that was used for training, fixed
# e.g.
# network_scene_a_10_3.json (trained on 1000 samples)
# network_scene_a_10_6.json (trained on 1000000 samples)
# network_scene_b_10_3.json (trained on 1000 samples)
# network_scene_b_10_6.json (trained on 1000000 samples)

srand(4)
train_neural_networks()

srand(3)
demo()
