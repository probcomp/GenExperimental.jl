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
        render(get(path), "orange")
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

function propose(start::Point, planner_params::PlannerParams,
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
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200, zorder=1000)
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
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200, zorder=1000)
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
        label(prop_goal.x, prop_goal.y, "proposed\n goal")
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
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200, zorder=1000)
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
        label(prop_goal.x, prop_goal.y, "proposed\n goal")
        if iter < 2
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
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200, zorder=1000)
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
        label(prop_goal.x, prop_goal.y, "proposed\n goal")
        render(get(path), "orange")
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
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
        label(prop_goal.x, prop_goal.y, "proposed\n goal")
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
        plt[:scatter](measured_xs, measured_ys, color="orange", s=200, zorder=1000)
        plt[:scatter]([active_goal.x], [active_goal.y], color="red", s=200)
        label(goal.x, goal.y, "goal")
        plt[:scatter]([prop_goal.x], [prop_goal.y], color="magenta", s=200)
        label(prop_goal.x, prop_goal.y, "proposed\n goal")
        plt[:scatter](map((p) -> p.x, locations), 
                    map((p) -> p.y, locations), s=200, color="green")
        ax = plt[:gca]()
        for loc in locations
            patch = patches.Circle((loc.x, loc.y), radius=measurement_noise, facecolor="green", edgecolor="green", alpha=0.2, clip_on=true)
            ax[:add_artist](patch)
        end
        fname = @sprintf("%s/frame_%03d.png", dir, framenum)
        plt[:savefig](fname)
        plt[:close]()

    end
   
    mh_animation_dir = "mh_animation/"

    # initialize
    score, tree, path, optimized_path, locations, goal = propose(
        start, planner_params, scene, speed, times, measurement_noise,
        measured_xs, measured_ys)
    for iter=1:100
        prop_score, tree, path, optimized_path, locations, prop_goal = propose(
            start, planner_params, scene, speed, times, measurement_noise,
            measured_xs, measured_ys)
        render_frames(mh_animation_dir, iter, scene,  goal, prop_goal, start,
                      measured_xs, measured_ys, tree, path, optimized_path,
                      locations)
        println("prop_score: $prop_score, score: $score")
        if log(rand()) < prop_score - score
            # accept
            println("accept")
            score = prop_score
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
