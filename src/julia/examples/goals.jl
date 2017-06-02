include("rrt.jl")
using Distributions

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
    for i=1:iters
        point_idx = 2 + (i % num_interior_points)
        @assert point_idx > 1 # not start
        @assert point_idx < length(original.points) # not goal
        point = original.points[point_idx]
        adjusted = Point(point.x + rand() * std, point.y + rand() * std)
        prev_dist = dist(new_points[point_idx - 1], point) + dist(point, new_points[point_idx + 1])
        ok_backward = line_of_site(scene, new_points[point_idx - 1], adjusted)
        ok_forward = line_of_site(scene, adjusted, new_points[point_idx + 1])
        if ok_backward && ok_forward
            new_dist = dist(new_points[point_idx - 1], adjusted) + dist(adjusted, new_points[point_idx + 1])
            if new_dist < prev_dist
                # accept the change
                new_points[point_idx] = adjusted
            end
        end
    end
    Path(original.start, original.goal, new_points)
end

function optimize_path(scene::Scene, original::Path, refine_iters::Int, refine_std::Float64)
    simplified = simplify_path(scene, original)
    refined = refine_path(scene, simplified, refine_iters, refine_std)
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

function render(path::Path, line_color, start_color, goal_color)
    plt[:scatter]([path.start.x], [path.start.y], color=start_color, s=200)
    plt[:scatter]([path.goal.x], [path.goal.y], color=goal_color, s=200)
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
    plt[:figure](figsize=(10, 10))
    render(scene)
    render(tree)
    if !isnull(path)
        render(get(path), "orange", "blue", "red")
        render(get(optimized_path), "purple", "blue", "red")
    end
    plt[:savefig]("planner.png")
end
planner_demo()






