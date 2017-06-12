using Gen

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
@register_module2(:uniform_2d, uniform_2d_simulate, uniform_2d_regenerate)
#export uniform_2d_simulate
#export uniform_2d_regenerate
uniform_2d = (args...) -> (uniform_2d_simulate)(args...)[1]
export uniform_2d

function agent_model(T::Trace)

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
