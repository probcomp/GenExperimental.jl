using Gen
using Distributions
using PyCall
@pyimport matplotlib.patches as patches

include("path_planner.jl")

function uniform_2d_simulate(xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    x, log_weight_x = uniform_simulate(xmin, xmax)
    y, log_weight_y = uniform_simulate(ymin, ymax)
    Point(x, y), log_weight_x + log_weight_y
end
function uniform_2d_regenerate(point::Point, xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    log_weight_x = uniform_regenerate(point.x, xmin, xmax)
    log_weight_y = uniform_regenerate(point.y, ymin, ymax)
    log_weight_x + log_weight_y
end
@register_module(:uniform_2d, uniform_2d_simulate, uniform_2d_regenerate)

function agent_model(T::Trace)

    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100
    scene = Scene(xmin, xmax, ymin, ymax) ~ "scene"

    # add trees
    for i=1:(10 ~ "num_trees")
        location = uniform_2d(xmin, xmax, ymin, ymax) ~ "tree-$i-location"
        size = 10.
        add!(scene, Tree(location, size))
    end

    # add walls
    wall_height = 10.
    add!(scene, Wall(Point(20., 40.), 1, 40., 2., wall_height))
    add!(scene, Wall(Point(60., 40.), 2, 40., 2., wall_height))
    add!(scene, Wall(Point(60.-15., 80.), 1, 15. + 2., 2., wall_height))
    add!(scene, Wall(Point(20., 80.), 1, 15., 2., wall_height))
    # TODO: add if (door locked for the bottom wall)
    add!(scene, Wall(Point(20., 40.), 2, 40., 2., wall_height))

    # starting location of the drone
    start = uniform_2d(xmin, xmax, ymin, ymax) ~ "start"

    # destinatoin of the drone
    destination = uniform_2d(xmin, xmax, ymin, ymax) ~ "destination"

    speed = 10. ~ "speed"
    times = collect(linspace(0.0, 10.0, 30)) ~ "times"

    # plan a path from starting location to destination
    planner_params = PlannerParams(2000, 3.0, 10000, 1.)
    tree, path, optimized_path = plan_path(start, destination, scene,
                                           planner_params)

    measurement_noise = 8.0 ~ "measurement_noise"

    if isnull(optimized_path)
        # no path found
        fail(T)
    else
        # walk the path at a constant speed, and record locations at times
        locations = walk_path(get(optimized_path), speed, times) ~ "locations"

        # add measurement noise to the true locations
        measurements = Array{Point,1}(length(times))
        for (i, loc) in enumerate(locations)
            measurements[i] = Point(normal(loc.x, measurement_noise) ~ "x$i", 
                                    normal(loc.y, measurement_noise) ~ "y$i")
        end
    end

    # record for visualization purposes
    tree ~ "tree"
    path ~ "path"
    optimized_path ~ "optimized_path"
    
    return nothing
end

function add_scene_elements!(trace::Trace)
    intervene!(trace, "num_trees", 3)
    constrain!(trace, "tree-1-location", Point(30, 20))
    constrain!(trace, "tree-2-location", Point(83, 80))
    constrain!(trace, "tree-3-location", Point(80, 40))
    constrain!(trace, "tree-1-size", 10.)
    constrain!(trace, "tree-2-size", 10.)
    constrain!(trace, "tree-3-size", 10.)
    constrain!(trace, "start", Point(90., 10.))
end

function add_observations!(trace::Trace, measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1}, tmax::Int)
    for s=1:tmax
        constrain!(trace, "x$s", measured_xs[s])
        constrain!(trace, "y$s", measured_ys[s])
    end
end

function particle_cloud_demo(num_particles::Int, num_iter::Int)

    # first simulate some data from the model
    simulation_trace = Trace()
    add_scene_elements!(simulation_trace)
    intervene!(simulation_trace, "measurement_noise", 1.0)
    agent_model(simulation_trace)
    times = value(simulation_trace, "times")
    measured_xs = map((i) -> value(simulation_trace, "x$i"), 1:length(times))
    measured_ys = map((i) -> value(simulation_trace, "y$i"), 1:length(times))

    # generate the particle clouds (use higher noise in algorithm)
    for t=1:length(measured_xs)
        println("time: $t")
        particles = Trace[]
        for i=1:num_particles
            println("i = $i")

            # initialize
            trace = Trace()
            add_scene_elements!(trace)
            add_observations!(trace, measured_xs, measured_ys, t)
            agent_model(trace)

            for iter=0:num_iter
        
                # propose
                proposal_trace = Trace()
                add_scene_elements!(proposal_trace)
                add_observations!(proposal_trace, measured_xs, measured_ys, t)
                agent_model(proposal_trace)

                if log(rand()) < score(proposal_trace) - score(trace)
                    trace = proposal_trace
                end
            end
            push!(particles, trace)
        end

        # typically, we open a rendering object, and then render (parts of) many
        # traces over it before finalizing the rendering to the image (compositing)
        #include("matplotlib_rendering.jl")
        #rendering = PlotRendering()
        include("povray_rendering.jl")
        rendering = PovrayRendering([50., 50., 150.], [50., 50., 0.], [50., 50., 150.])
        rendering.quality = 10
        render_samples(rendering, particles)
        finish(rendering, "particle_cloud/mh_$(num_iter)_$t.png")
    end
end

srand(1)
particle_cloud_demo(10, 10)
