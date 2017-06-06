using Gen
using Distributions
using PyCall
@pyimport matplotlib.patches as patches

include("path_planner.jl")

function agent_model(T::Trace, start::Point, planner_params::PlannerParams, 
                     scene::Scene, speed::Float64, times::Array{Float64,1}, 
                     measurement_noise::Float64)
    
    # sample goal uniformly from scene area
    goal = Point(uniform(scene.xmin, scene.xmax) ~ "goal_x", 
                 uniform(scene.xmin, scene.xmax) ~ "goal_y")

    # plan a path from start to goal
    tree, path, optimized_path = plan_path(start, goal, scene, planner_params)

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
    start ~ "start"
    scene ~ "scene"
    speed ~ "speed"
    times ~ "times"
    measurement_noise ~ "measurement_noise"

    tree ~ "tree"
    path ~ "path"
    optimized_path ~ "optimized_path"
    
    return nothing
end

# TODO write a better trace rendering
function render_trace(trace::Trace)
    ax = plt[:gca]()

    scene = value(trace, "scene")
    render(scene)

    start = value(trace, "start")
    ax[:scatter]([start.x], [start.y], color="blue", s=200)

    if hasvalue(trace, "goal_x") && hasvalue(trace, "goal_y")
        goal = Point(value(trace, "goal_x"), value(trace, "goal_y"))
        ax[:scatter]([goal.x], [goal.y], color="red", s=200)
    end

    if hasvalue(trace, "tree")
        tree = value(trace, "tree")
        render(tree, 1.0)
    end

    if hasvalue(trace, "path")
        path = value(trace, "path")
        optimized_path = value(trace, "optimized_path")
        if !isnull(path)
            render(get(path), "orange")
            render(get(optimized_path), "purple")
            locations = value(trace, "locations")
            measurement_noise = value(trace, "measurement_noise")
            for loc in locations
                patch = patches.Circle((loc.x, loc.y),
                                    radius=measurement_noise, facecolor="green",
                                    edgecolor="green", alpha=0.2, clip_on=true, zorder=-1)
                ax[:add_artist](patch)
            end
        end
    end

    times = value(trace, "times")
    measured_xs = []
    measured_ys = []
    for i=1:length(times)
        if hasvalue(trace, "x$i") && hasvalue(trace, "y$i")
            push!(measured_xs, value(trace, "x$i"))
            push!(measured_ys, value(trace, "y$i"))
        end
    end
    plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
end

function propose_independent_mh(start::Point, planner_params::PlannerParams,
                 scene::Scene, speed::Float64, times::Array{Float64,1},
                 measurement_noise::Float64,
                 measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1})
    trace = Trace()
    for (i, (xi, yi)) in enumerate(zip(measured_xs, measured_ys))
        constrain!(trace, "x$i", xi)
        constrain!(trace, "y$i", yi)
    end
    agent_model(trace, start, planner_params, scene,
                speed, times, measurement_noise)
    return trace
end

function random_walk_proposal(T::Trace, goal::Point)
    # TODO
    goal_x = normal(goal.x, 5.0) ~ "goal_x"
    goal_y = normal(goal.y, 5.0) ~ "goal_y"
end

function propose_random_walk_mh(start::Point, planner_params::PlannerParams,
                 scene::Scene, speed::Float64, times::Array{Float64,1},
                 measurement_noise::Float64,
                 measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1},
                 goal::Point)

    # evaluate the previous model score
    model_trace = Trace()
    for (i, (xi, yi)) in enumerate(zip(measured_xs, measured_ys))
        constrain!(model_trace, "x$i", xi)
        constrain!(model_trace, "y$i", yi)
    end
    constrain!(model_trace, "goal_x", goal.x)
    constrain!(model_trace, "goal_y", goal.y)
    tree, path, optimized_path, locations = agent_model(
        model_trace, start, planner_params, scene,
        speed, times, measurement_noise)
    prev_model_score = model_trace.log_weight

    # propose 
    proposal_trace = Trace()
    propose!(proposal_trace, "goal_x")
    propose!(proposal_trace, "goal_y")
    random_walk_proposal(proposal_trace, goal) # symmetric proposal

    # evaluate the model score for the proposed goal
    model_trace = Trace()
    for (i, (xi, yi)) in enumerate(zip(measured_xs, measured_ys))
        constrain!(model_trace,"x$i", xi)
        constrain!(model_trace,"y$i", yi)
    end
    constrain!(model_trace, "goal_x", value(proposal_trace, "goal_x"))
    constrain!(model_trace, "goal_y", value(proposal_trace, "goal_y"))
    tree, path, optimized_path, locations = agent_model(
        model_trace, start, planner_params, scene,
        speed, times, measurement_noise)
    goal = Point(value(model_trace, "goal_x"), value(model_trace, "goal_y"))
    new_model_score = model_trace.log_weight

    mh_ratio = new_model_score - prev_model_score
    (mh_ratio, tree, path, optimized_path, locations, goal)
end

function model_demo()
    # parameters that are fixed for now
    speed = 10.
    times = Float64[1, 2, 3, 4, 5]
    measurement_noise = 1.0
    planner_params = PlannerParams(2000, 3.0, 10000, 1.)
    obstacles = []
    push!(obstacles, Wall(Point(20., 40.), 1, 40., 2.))
    push!(obstacles, Wall(Point(60., 40.), 2, 40., 2.))
    push!(obstacles, Wall(Point(60.-15., 80.), 1, 15. + 2., 2.))
    push!(obstacles, Wall(Point(20., 80.), 1, 15., 2.))
    push!(obstacles, Wall(Point(20., 40.), 2, 40., 2.))
    scene = Scene(0, 100, 0, 100, obstacles)

    # first simulate some data from the model
    trace = Trace()
    start = Point(90., 10.)
    agent_model(trace, start, planner_params, scene, speed, times, measurement_noise)
    measured_xs = map((i) -> value(trace, "x$i"), 1:length(times))
    measured_ys = map((i) -> value(trace, "y$i"), 1:length(times))

    # plot it
    plt[:figure](figsize=(10, 10))
    render_trace(trace)
    plt[:tight_layout]()
    plt[:savefig]("simulated_data.png")

    # now, do inference
    # TODO also do inference over the meauremtn noise
    # TODO also do random walk over goal
    measurement_noise = 8.0

    function label(x, y, text)
        t = plt[:text](x, y-2, text, fontsize=30, horizontalalignment="center", verticalalignment="top", zorder=100)
        t[:set_bbox](Dict([("facecolor", "white"), ("alpha",0.8), ("edgecolor", "None")]))
    end
    
    function measurements_label()
        label(90, 55, "observations")
    end

    mh_animation_dir = "mh_animation/"

    # render the prefix frames
    trace = Trace()
    for j=1:length(measured_xs)
        constrain!(trace, "x$j", measured_xs[j])
        constrain!(trace, "y$j", measured_ys[j])
    end
    agent_model(trace, start, planner_params, scene, speed, times, measurement_noise)

    for i=0:length(measured_xs)
        vis_trace = deepcopy(trace)
        println("rendering start frame $i")
        delete!(vis_trace, "goal_x")
        delete!(vis_trace, "goal_y")
        delete!(vis_trace, "tree")
        delete!(vis_trace, "path")
        delete!(vis_trace, "optimized_path")
        for j=i+1:length(measured_xs)
            delete!(vis_trace, "x$j")
            delete!(vis_trace, "y$j")
        end
        fname = @sprintf("start_%03d.png", i)
        plt[:figure](figsize=(10, 10))
        render_trace(vis_trace)
        plt[:tight_layout]()
        plt[:savefig](fname)
    end

    # initialize
    trace = propose_independent_mh(
        start, planner_params, scene, speed, times, measurement_noise,
        measured_xs, measured_ys)
    for iter=0:100
		println("MH iter: $iter")

        proposed_trace = propose_independent_mh(
            start, planner_params, scene, speed, times, measurement_noise,
            measured_xs, measured_ys)

		# render the original and proposed trace
        plt[:figure](figsize=(20, 10))
        plt[:subplot](1, 2, 1)
        render_trace(trace)
        plt[:subplot](1, 2, 2)
        render_trace(proposed_trace)
        plt[:tight_layout]()
        fname = @sprintf("%s/frame_%03d.png", mh_animation_dir, iter * 2)
        plt[:savefig](fname)

		# MH accept / reject
        if log(rand()) < proposed_trace.log_weight - trace.log_weight 
            # accept
            accepted = true
			trace = proposed_trace
        else
            accepted = false
        end

		# add the accept / reject shading
        shade_color = accepted ? "green" : "red"
        plt[:subplot](1, 2, 2)
		ax = plt[:gca]()
		rect = plt[:Rectangle]((0.0, 0.0), 1.0, 1.0, fill=true, alpha=0.3, facecolor=shade_color)
		rect[:set_transform](ax[:transAxes])
		ax[:add_patch](rect)
        fname = @sprintf("%s/frame_%03d.png", mh_animation_dir, iter * 2 + 1)
        plt[:savefig](fname)

    end

end
srand(2)
model_demo()


# TODO it should be possible to extract the value from the trace even if its not random
# and/or a module. it is just not allowed to be *constrain*ed. it can, however, be named
# a special subset of random choices that are not constrained, are called 'requested'
# or 'outputs' and these do have to be modules. still not exactly clear on whether 'outputs' and
# 'constraints' can coexist in a query
