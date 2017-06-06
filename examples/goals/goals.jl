using Gen
using Distributions
include("rrt.jl")
@pyimport matplotlib.patches as patches

immutable PlannerParams
    rrt_iters::Int
    rrt_dt::Float64 # the maximum proposal distance
    refine_iters::Int
    refine_std::Float64
end

immutable Path
    start::Point
    goal::Point
    points::Array{Point,1}
end

function simplify_path(scene::Scene, original::Path)
    new_points = Array{Point,1}()
    push!(new_points, original.start)
    for i=2:length(original.points) - 1
        if !line_of_site(scene, new_points[end], original.points[i + 1])
            push!(new_points, original.points[i])
        end
    end
    @assert line_of_site(scene, new_points[end], original.goal)
    push!(new_points, original.goal)
    Path(original.start, original.goal, new_points)
end

function refine_path(scene::Scene, original::Path, iters::Int, std::Float64)
    # do stochastic optimization
    new_points = deepcopy(original.points)
    num_interior_points = length(original.points) -2
    if num_interior_points == 0
        return original
    end
    for i=1:iters
        point_idx = 2 + (i % num_interior_points)
        @assert point_idx > 1 # not start
        @assert point_idx < length(original.points) # not goal
        prev_point = new_points[point_idx-1]
        point = new_points[point_idx]
        next_point = new_points[point_idx+1]
        adjusted = Point(point.x + randn() * std, point.y + randn() * std)
        cur_dist = dist(prev_point, point) + dist(point, next_point)
        ok_backward = line_of_site(scene, prev_point, adjusted)
        ok_forward = line_of_site(scene, adjusted, next_point)
        if ok_backward && ok_forward
            new_dist = dist(prev_point, adjusted) + dist(adjusted, next_point)
            if new_dist < cur_dist 
                # accept the change
                new_points[point_idx] = adjusted
            end
        end
    end
    Path(original.start, original.goal, new_points)
end

function optimize_path(scene::Scene, original::Path, refine_iters::Int, refine_std::Float64)
    #simplified = simplify_path(scene, original)
    refined = refine_path(scene, original, refine_iters, refine_std)
    refined
end

function plan_path(start::Point, goal::Point, scene::Scene, params::PlannerParams)
    scheme = HolonomicPointRRTScheme(scene)
    tree = rrt(scheme, start, params.rrt_iters, params.rrt_dt)

    # find the best path along the tree to the goal, if one exists
    best_node = tree.nodes[1]
    min_cost = Inf
    path_found = false
    for node in tree.nodes
        # check for line-of-site to the goal
        clear_path = line_of_site(scene, node.conf, goal)
        cost = node.cost_from_start + (clear_path ? dist(node.conf, goal) : Inf)
        if cost < min_cost
            path_found = true
            best_node = node
            min_cost = cost
        end
    end

    local path::Nullable{Path}
    if path_found
        # extend the tree to the goal configuration
        control = Point(goal.x - best_node.conf.x, goal.y - best_node.conf.y)
        goal_node = add_node!(tree, best_node, goal, control, min_cost)
        points = Array{Point,1}()
        node::RRTNode{Point,Point} = goal_node
        push!(points, node.conf)
        # the path will contain the start and goal
        while !isnull(node.parent)
            node = get(node.parent)
            push!(points, node.conf)
        end
        @assert points[end] == start # the start point
        @assert points[1] == goal
        path = Nullable{Path}(Path(start, goal, reverse(points)))
    else
        path = Nullable{Path}()
    end
    
    local optimized_path::Nullable{Path}
    if path_found
        optimized_path = Nullable{Path}(optimize_path(scene, get(path),
                                                      params.refine_iters, params.refine_std))
    else
        optimized_path = Nullable{Path}()
    end
    tree, path, optimized_path # if path is null, then no path was found
end

function walk_path(path::Path, speed::Float64, times::Array{Float64,1})
    distances_from_start = Array{Float64,1}(length(path.points))
    distances_from_start[1] = 0.0
    for i=2:length(path.points)
        distances_from_start[i] = distances_from_start[i-1] + dist(path.points[i-1], path.points[i])
    end
    locations = Array{Point,1}(length(times))
    locations[1] = path.points[1]
    for (time_idx, t) in enumerate(times)
        if t < 0.0
            error("times must be positive")
        end
        desired_distance = t * speed
        used_up_time = false
        # NOTE: can be improved (iterate through path points along with times)
        for i=2:length(path.points)
            prev = path.points[i-1]
            cur = path.points[i]
            dist_to_prev = dist(prev, cur)
            if distances_from_start[i] >= desired_distance
                # we overshot, the location is between i-1 and i
                overshoot = distances_from_start[i] - desired_distance
                @assert overshoot <= dist_to_prev
                past_prev = dist_to_prev - overshoot
                frac = past_prev / dist_to_prev
                locations[time_idx] = Point(prev.x * (1. - frac) + cur.x * frac,
                                     prev.y * (1. - frac) + cur.y * frac)
                used_up_time = true
                break
            end
        end
        if !used_up_time
            # sit at the goal indefinitely
            locations[time_idx] = path.goal
        end
    end
    locations
end

function render(path::Path, line_color)
    for i=1:length(path.points) - 1
        a = path.points[i]
        b = path.points[i + 1]
        plt[:plot]([a.x, b.x], [a.y, b.y], color=line_color, lw=3, alpha=0.5)
    end
end

function planner_demo()
    # plot them
    obstacles = []
    push!(obstacles, Polygon([Point(30, 30), Point(80, 30), Point(80, 35), Point(30, 35)]))
    push!(obstacles, Polygon([Point(30, 30), Point(30, 80), Point(35, 80), Point(35, 30)]))
    push!(obstacles, Polygon([Point(60, 30), Point(60, 80), Point(65, 80), Point(65, 30)]))
    scene = Scene(0, 100, 0, 100, obstacles)
    start = Point(50, 50)
    goal = Point(50, 10)
    @time tree, path, optimized_path = plan_path(start, goal, scene, PlannerParams(1000, 5.0, 1000, 1.0))
    locations = walk_path(get(optimized_path), 5.0, collect(linspace(0.0, 40.0, 10)))
    plt[:figure](figsize=(10, 10))
    render(scene)
    render(tree, 1.0)
    if !isnull(path)
        render(get(path), "black")
        render(get(optimized_path), "purple")
    end
    plt[:scatter](map((p) -> p.x, locations), map((p) -> p.y, locations), s=200)
    plt[:savefig]("planner.png")
end
#planner_demo()


function agent_model(T::Trace, start::Point, planner_params::PlannerParams, 
                     scene::Scene, speed::Float64, times::Array{Float64,1}, 
                     measurement_noise::Float64)
    
    # sample goal uniformly from scene area
    goal = Point(uniform(scene.xmin, scene.xmax) ~ "goal_x", 
                 uniform(scene.xmin, scene.xmax) ~ "goal_y")

    # plan a path from start to goal
    tree, path, optimized_path = plan_path(start, goal, scene, planner_params)
    tree ~ "tree"
    path ~ "path"
    optimized_path ~ "optimized_path"

    if isnull(optimized_path)
        # no path found
        fail(T)
        locations = Nullable{Array{Point,1}}() # null
        measurements = Nullable{Array{Point,1}}() # null
    else
        # walk the path at a constant speed, and record locations at times
        locations = walk_path(get(optimized_path), speed, times)

        # add measurement noise to the true locations
        measurements = Array{Point,1}(length(times))
        for (i, loc) in enumerate(locations)
            measurements[i] = Point(normal(loc.x, measurement_noise) ~ "x$i", 
                                    normal(loc.y, measurement_noise) ~ "y$i") 
        end
    end

    # return values for debugging and visualization
    return tree, path, optimized_path, locations, measurements 
end

function render_trace(trace::Trace)

    # data that may be useful for the rendering:
    # 1. parameters of the program
    # 2. constants defined in the program
    # 3. random choices in the program that are labelled (OK)
    # 4. random choics in the program that are not labelled
    # 5. deterministic choices in the program

    # alternative solution:
    # make the program build up the visualization inline?

    # what about 'proposed' random variables?
    # we may want to show the proposed and the true goal on the same map?
    # for MH, we can show the 'accepted' trace in one pane, and the proposals in another pane? (flash red or green)
    # for SIR, show a trace in each pane

    # each program execution can build up a visualization itself?
    # this ties each program to a particular visualization...
    # we need to support different visualizations for one model program (e.g. grid world vs Unreal engine)
    
    # the simplest option seems to be to allow for arbitrary storage in the trace data structure?
    # the program can record arbitrary things it wants?
    # but why not just make the trace itself an arbitrary data structure, that may differ from program to 
    # program..
    # [ the program records values by setting them in this trace ]

    # automatic accumulation of log-weight?

    # render the scene
    #   - an example of a constant defined in the program
    # render the start location
    #   - an example of a parameter
    # render the goal location
    #   - from the trace (if it exists)
    # render
    # render the path
    #   - which is random but is not traced
    # [ a solution is to reconstruct the path by running that part of the program? ]
    # [ another solution is to allow us to record arbitrary pieces of state in the program? ]
    # problem: the tree is not traced.. -- an example of a random
end

# the purpose of the rendering is to encode the trace for input to the human
# visual system

function propose_independent_mh(start::Point, planner_params::PlannerParams,
                 scene::Scene, speed::Float64, times::Array{Float64,1},
                 measurement_noise::Float64,
                 measured_xs::Array{Float64,1}, measured_ys::Array{Float64,1})

    trace = Trace()
    for (i, (xi, yi)) in enumerate(zip(measured_xs, measured_ys))
        trace.vals["x$i"] = xi
        trace.vals["y$i"] = yi
    end
    tree, path, optimized_path, locations = agent_model(
        trace, start, planner_params, scene,
        speed, times, measurement_noise)
    goal = Point(trace.vals["goal_x"], trace.vals["goal_y"])
    (trace.log_weight, tree, path, optimized_path, locations, goal)
end

function random_walk_proposal(T::Trace, goal::Point)
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
        model_trace.vals["x$i"] = xi
        model_trace.vals["y$i"] = yi
    end
    model_trace.vals["goal_x"] = goal.x
    model_trace.vals["goal_y"] = goal.y
    tree, path, optimized_path, locations = agent_model(
        model_trace, start, planner_params, scene,
        speed, times, measurement_noise)
    prev_model_score = model_trace.log_weight

    # propose 
    proposal_trace = Trace()
    push!(proposal_trace.outputs, "goal_x", "goal_y")
    random_walk_proposal(proposal_trace, goal) # symmetric proposal

    # evaluate the model score for the proposed goal
    model_trace = Trace()
    for (i, (xi, yi)) in enumerate(zip(measured_xs, measured_ys))
        model_trace.vals["x$i"] = xi
        model_trace.vals["y$i"] = yi
    end
    model_trace.vals["goal_x"] = proposal_trace.vals["goal_x"]
    model_trace.vals["goal_y"] = proposal_trace.vals["goal_y"]
    tree, path, optimized_path, locations = agent_model(
        model_trace, start, planner_params, scene,
        speed, times, measurement_noise)
    goal = Point(model_trace.vals["goal_x"], model_trace.vals["goal_y"])
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
    @time tree, path, optimized_path, locations = agent_model(trace, start, planner_params, scene,
                                                        speed, times, measurement_noise)
    measured_xs = map((i) -> trace.vals["x$i"], 1:length(times))
    measured_ys = map((i) -> trace.vals["y$i"], 1:length(times))

    # plot it
    plt[:figure](figsize=(10, 10))
    render(scene)
    plt[:scatter]([start.x], [start.y], color="blue", s=200)
    render(tree, 1.0)
    if !isnull(path)
        render(get(path), "orange")
        render(get(optimized_path), "purple")
    end
    goal = Point(trace.vals["goal_x"], trace.vals["goal_y"])
    plt[:scatter]([goal.x], [goal.y], color="red", s=200)
    plt[:scatter](map((p) -> p.x, locations), 
                  map((p) -> p.y, locations), s=200, color="green")
    plt[:scatter](measured_xs, measured_ys, s=200, color="orange")
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

    function render_prefix_frames(dir::String, scene::Scene, start::Point, 
                                  measured_xs::Array{Float64,1},
                                  measured_ys::Array{Float64,1}, framenum::Int)
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        #measurements_label()
        fname = @sprintf("%s/start_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()
    end

    function render_frames(dir::String, iter::Int, scene::Scene,
                           active_goal::Point, prop_goal::Point,
                           start::Point,
                           measured_xs::Array{Float64,1},
                           measured_ys::Array{Float64,1},
                           tree, path, optimized_path, locations)

        println("active_goal: $active_goal")
        println("prop_goal: $prop_goal")

        figsize=(10,10)
    
        num_frames = 6
                           
        # frame 1 : map, start, observations, current goal
        framenum = (iter -1) * num_frames + 1
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        measurements_label()
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

        # frame 2 : map, start, observations, current goal, proposed goal
        framenum = (iter -1) * num_frames + 2
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        measurements_label()
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        if iter < 5
            plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
            label(prop_goal.x, prop_goal.y, "proposed\n goal")
        end
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

        # frame 3 : map, start, observations, current_goal, proposed_goal,
        #           tree
        framenum = (iter -1) * num_frames + 3
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        measurements_label()
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        if iter < 5
            plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
            label(prop_goal.x, prop_goal.y, "proposed\n goal")
        end
        if iter < 5
            render(tree, 0.5)
        end
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

        # frame 4 : map, start, observations, current_goal, proposed_goal,
        #           orange path
        framenum = (iter -1) * num_frames + 4
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        measurements_label()
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        if iter < 5
            plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
            label(prop_goal.x, prop_goal.y, "proposed\n goal")
        end
        render(get(path), "black")
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

        # frame 5 : map, start, observations, current_goal, proposed_goal,
        #           purple path
        framenum = (iter -1) * num_frames + 5
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        measurements_label()
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        if iter < 5
            plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
            label(prop_goal.x, prop_goal.y, "proposed\n goal")
        end
        render(get(optimized_path), "purple")
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

        # frame 6 : map, start, observations, current_goal, proposed_goal,
        #           green points
        framenum = (iter -1) * num_frames + 6
        plt[:figure](figsize=(10,10))
        render(scene)
        plt[:scatter]([start.x], [start.y], color="blue", s=200)
        label(start.x, start.y, "start")
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
        measurements_label()
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        if iter < 5
            plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
            label(prop_goal.x, prop_goal.y, "proposed\n goal")
        end
        plt[:scatter](map((p) -> p.x, locations), 
                    map((p) -> p.y, locations), s=200, color="green")
        ax = plt[:gca]()
        for loc in locations
            patch = patches.Circle((loc.x, loc.y), radius=measurement_noise, facecolor="green", edgecolor="green", alpha=0.2, clip_on=true, zorder=-1)
            ax[:add_artist](patch)
        end
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

    end
   
    mh_animation_dir = "mh_animation/"

    # render the prefix frames
    for i=1:length(measured_xs)+1
        println(i)
        render_prefix_frames(mh_animation_dir, scene, start, measured_xs[1:i-1], measured_ys[1:i-1], i)
    end
    return Nothing

    # initialize
    prev_model_score, tree, path, optimized_path, locations, goal = propose_independent_mh(
        start, planner_params, scene, speed, times, measurement_noise,
        measured_xs, measured_ys)
    frame = 1
    for iter=1:100
        #mh_ratio, tree, path, optimized_path, locations, prop_goal = propose_random_walk_mh(
            #start, planner_params, scene, speed, times, measurement_noise,
            #measured_xs, measured_ys, goal)

        new_model_score, tree, path, optimized_path, locations, prop_goal = propose_independent_mh(
            start, planner_params, scene, speed, times, measurement_noise,
            measured_xs, measured_ys)
        if !isnull(path)
            render_frames(mh_animation_dir, frame, scene,  goal, prop_goal, start,
                        measured_xs, measured_ys, tree, path, optimized_path,
                        locations)
            frame += 1
        end
        if log(rand()) < new_model_score - prev_model_score
            # accept
            println("accept")
            prev_model_score = new_model_score
            goal = prop_goal
        else
            println("reject")
        end
    end

end
srand(2)
model_demo()


# TODO it should be possible to extract the value from the trace even if its not random
# and/or a module. it is just not allowed to be *constrain*ed. it can, however, be named
# a special subset of random choices that are not constrained, are called 'requested'
# or 'outputs' and these do have to be modules. still not exactly clear on whether 'outputs' and
# 'constraints' can coexist in a query
