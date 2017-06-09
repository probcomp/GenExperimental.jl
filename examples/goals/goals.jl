@everywhere using Gen
@everywhere using Distributions
@everywhere using PyCall
@pyimport matplotlib.patches as patches

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

    # plan a path from starting location to destination
    planner_params = PlannerParams(2000, 3.0, 10000, 1.)
    tree, path, optimized_path = plan_path(start, destination, scene,
                                           planner_params)

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
    tree ~ "tree"
    path ~ "path"
    optimized_path ~ "optimized_path"
    
    return nothing
end

@everywhere function add_observations!(trace::Trace, measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1}, tmax::Int)
    for s=1:tmax
        constrain!(trace, "x$s", measured_xs[s])
        constrain!(trace, "y$s", measured_ys[s])
    end
end

@everywhere function mh_inference(scene_trace::Trace, measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1}, t::Int, num_iter::Int)

    # the input scene_trace is assumed to contain only the scene
    @assert !hasvalue(scene_trace, "destination")
    @assert !hasvalue(scene_trace, "optimized_path")
    @assert !hasvalue(scene_trace, "x1")
    trace = deepcopy(scene_trace)
    add_observations!(trace, measured_xs, measured_ys, t)
    agent_model(trace)
    delete!(trace, "tree") # avoid copying this huge data structure

    # MH steps
    for iter=0:num_iter
        proposal_trace = deepcopy(trace)
        reset_score(proposal_trace)
        agent_model(proposal_trace)
        if log(rand()) < score(proposal_trace) - score(trace)
            trace = proposal_trace
            delete!(trace, "tree")
        end
    end
    trace
end

include("povray_rendering.jl")

# ---- amortized inference ----- #

function sigmoid{T}(val::T)
    1.0 ./ (1.0 + exp(-val))
end

function neural_network(T::AbstractTrace, features::Array{Float64,1}, num_hidden::Int)
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
    output_x = normal((W_output_x_mu' * hidden + b_output_x_mu)[1],
                      (exp(W_output_x_log_std' * hidden + b_output_x_log_std))[1]) ~ "output-x"
    output_y = normal((W_output_y_mu' * hidden + b_output_y_mu)[1], 
                      (exp(W_output_y_log_std' * hidden + b_output_y_log_std))[1]) ~ "output-y"
end

function train_neural_network(model_trace::Trace)
    # TODO intervene to set the 'times' to 1:15 steps
    # the input trace contains interventions and/or constraints
    times = value(model_trace, "times")
    num_features = length(times) * 2 + 2 # start.x and start.y

    # NOTE: the argument for why you need the causal model at all is something
    # like "we could make a special input to the network for every possible
    # random variable, but that does not scale..". or it could be that you need the causal model to train the net..
    num_hidden = 50
    step_a = 2000.0
    step_b = 0.75
    minibatch_size = 100
    max_iter = 1000

    # initialize parameters
    parameters = Dict()
    parameters["W-hidden"] = randn(num_hidden, num_features)
    parameters["b-hidden"] = randn(num_hidden)
    parameters["W-output-x-mu"] = randn(num_hidden)
    parameters["b-output-x-mu"] = randn()
    parameters["W-output-x-log-std"] = randn(num_hidden)
    parameters["b-output-x-log-std"] = randn()
    parameters["W-output-y-mu"] = randn(num_hidden)
    parameters["b-output-y-mu"] = randn()
    parameters["W-output-y-log-std"] = randn(num_hidden)
    parameters["b-output-y-log-std"] = randn()
    network_trace = DifferentiableTrace()
    for key in keys(parameters)
        parametrize!(network_trace, key, parameters[key])
    end

    for iter=1:max_iter
        objective_function = 0.0
        gradients = Dict()
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
        num_samples = 0
        for i=1:minibatch_size
            # simulate new values for start, destination, and measurments
            agent_model(model_trace)
            path = value(model_trace, "path")
            if isnull(path)
                println("$i planning failed..")
                continue
            end
            start = value(model_trace, "start")
            destination = value(model_trace, "destination")
            xs = map((i) -> value(model_trace, "x$i"), 1:length(times))
            ys = map((i) -> value(model_trace, "y$i"), 1:length(times))
            features = vcat([start.x, start.y], xs, ys)
            neural_network(network_trace, features, num_hidden)
            backprop(network_trace)
            for key in keys(parameters)
                gradients[key] += derivative(network_trace, key)
            end
            objective_function += score(network_trace)
            reset_score(network_trace)
            num_samples += 1
        end
        step_size = (step_a + iter) ^ (-step_b)
        for key in keys(parameters)
            parameters[key] += step_size * gradients[key] / num_samples 
        end
        println("objective function: $(objective_function / num_samples )")
    end
end

# ---- neural network demo ---- #

function neural_network_demo()

    trace = Trace()

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
    
    # only simultae 15 time steps (first 8 seconds)
    intervene!(trace, "times", collect(linspace(0., 8., 15)))

    # TODO modify the model to have some limited uncertainty over the scene (the bottom enclosure)

    # NOTE: the start position, destination, path, and measurements are all unconstrained
    train_neural_network(trace)

end

# ---- demo ---- #

function demo()

    camera_location = [50., -30., 120.]
    camera_look_at = [50., 50., 0.]
    light_location = [50., 50., 150.]

    function render(trace::Trace, fname::String) 
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 10
        frame.num_threads = 16
        render_trace(frame, trace)
        finish(frame, fname)
        println(fname)
    end

    function render(traces::Array{Trace,1}, fname::String) 
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 10
        frame.num_threads = 16
        render_traces(frame, traces)
        finish(frame, fname)
        println(fname)
    end


    # frame index
    f = 0

    # first show a few unconstrained and unintervened samples
    num_unconstrained_samples = 5
    for i=1:num_unconstrained_samples
        trace = Trace()
        agent_model(trace)
        render(trace, "frames/frame_$f.png") ; f += 1
    end

    # build up the scene incrementally, starting from an empty trace
    trace = Trace()
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-tree-1", true)
    intervene!(trace, "tree-1", Tree(Point(30, 20), 10.))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-tree-2", true)
    intervene!(trace, "tree-2", Tree(Point(83, 80), 10.))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-tree-3", true)
    intervene!(trace, "tree-3", Tree(Point(80, 40), 10.))
    render(trace, "frames/frame_$f.png") ; f += 1

    wall_height = 30.
    intervene!(trace, "is-wall-1", true)
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 40., 2., wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-2", true)
    intervene!(trace, "wall-2", Wall(Point(60., 40.), 2, 40., 2., wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-3", true)
    intervene!(trace, "wall-3", Wall(Point(60.-15., 80.), 1, 15. + 2., 2., wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-4", true)
    intervene!(trace, "wall-4", Wall(Point(20., 80.), 1, 15., 2., wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-5", true)
    intervene!(trace, "wall-5", Wall(Point(20., 40.), 2, 40., 2., wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1

    boundary_wall_height = 2.
    intervene!(trace, "is-wall-6", true)
    intervene!(trace, "wall-6", Wall(Point(0., 0.), 1, 100., 2., boundary_wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-7", true)
    intervene!(trace, "wall-7", Wall(Point(100., 0.), 2, 100., 2., boundary_wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-8", true)
    intervene!(trace, "wall-8", Wall(Point(0., 100.), 1, 100., 2., boundary_wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1
    intervene!(trace, "is-wall-9", true)
    intervene!(trace, "wall-9", Wall(Point(0., 0.), 2, 100., 2., boundary_wall_height))
    render(trace, "frames/frame_$f.png") ; f += 1

    # prevent the program from adding new wall or trees
    intervene!(trace, "is-tree-4", false)
    intervene!(trace, "is-wall-10", false)

    # add the drone starting position
    constrain!(trace, "start", Point(90., 10.))
    render(trace, "frames/frame_$f.png") ; f += 1
    
    # copy this trace for future reference (it contains just the scene A)
    scene_a_trace = deepcopy(trace)

    # add a known ground truth goal
    intervene!(trace, "measurement_noise", 1.0)
    intervene!(trace, "destination", Point(40., 60.))
    render(trace, "frames/frame_$f.png") ; f += 1

    # run model and extract observations, render ground druth
    agent_model(trace)
    times = value(trace, "times")
    measured_xs = map((i) -> value(trace, "x$i"), 1:length(times))
    measured_ys = map((i) -> value(trace, "y$i"), 1:length(times))
    render(trace, "frames/frame_$f.png") ; f += 1

    # delete the ground truth information from the trace
    delete!(trace, "tree")
    delete!(trace, "path")
    delete!(trace, "optimized_path")
    delete!(trace, "destination")
    for i=1:length(times)
        delete!(trace, "x$i")
        delete!(trace, "y$i")
    end
    render(trace, "frames/frame_$f.png") ; f += 1

    # add the observations incrementally using constraints
    for t=1:length(times)
        constrain!(trace, "x$t", measured_xs[t])
        constrain!(trace, "y$t", measured_ys[t])
        render(trace, "frames/frame_$f.png") ; f += 1
    end

    # remove observations
    for t=1:length(times)
        delete!(trace, "x$t")
        delete!(trace, "y$t")
    end

    # change the walls to add a bottom passageway
    delete!(trace, "wall-1")
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 15., 2., wall_height))
    delete!(trace, "is-tree-10")
    delete!(trace, "is-wall-10")
    intervene!(trace, "is-wall-10", true)
    intervene!(trace, "wall-10", Wall(Point(60.- 15, 40.), 1, 15., 2., wall_height))
    intervene!(trace, "is-wall-11", false)
    render(trace, "frames/frame_$f.png") ; f += 1

    # copy this trace for future reference (it contains just the scene B)
    scene_b_trace = deepcopy(trace)

    # add the observations incrementally using constraints
    for t=1:length(times)
        constrain!(trace, "x$t", measured_xs[t])
        constrain!(trace, "y$t", measured_ys[t])
        render(trace, "frames/frame_$f.png") ; f += 1
    end

    # remove observations
    for t=1:length(times)
        delete!(trace, "x$t")
        delete!(trace, "y$t")
    end

    # generate particle clouds showing the result of inference at each stage
    num_particles = 60
    num_iter = 100

    # for scene a
    for t=1:length(measured_xs)
        println("scene a, time: $t")
        particles::Array{Trace,1} = pmap((i) -> mh_inference(scene_a_trace, measured_xs, measured_ys, t, num_iter), 1:num_particles)
        render(particles, "frames/frame_$f.png") ; f += 1
    end

    # for scene b
    for t=1:length(measured_xs)
        println("scene b, time: $t")
        particles::Array{Trace,1} = pmap((i) -> mh_inference(scene_b_trace, measured_xs, measured_ys, t, num_iter), 1:num_particles)
        render(particles, "frames/frame_$f.png") ; f += 1
    end

end

#srand(3)
#demo()

srand(3)
neural_network_demo()
