using PyPlot
using PyCall
@pyimport matplotlib.path as mplPath
@pyimport matplotlib.patches as patches

immutable PlotRendering
end

function render(rendering::PlotRendering, poly::Polygon)
    points = map((p) -> Float64[p.x, p.y], poly.vertices)
    path = mplPath.Path(points)
    ax = plt[:gca]()
    patch = patches.PathPatch(path, facecolor="black")
    ax[:add_patch](patch)
end

function render(rendering::PlotRendering, scene::Scene)
    for obstacle in scene.obstacles
        render(rendering, obstacle)
    end
    ax = plt[:gca]()
    ax[:set_xlim](scene.xmin, scene.xmax)
    ax[:set_ylim](scene.ymin, scene.ymax)
end

function render(rendering::PlotRendering, tree::RRTTree{Point,Point}, alpha)
    for node in tree.nodes
        if !isnull(node.parent)
            # it is not the root
            x1 = get(node.parent).conf.x
            y1 = get(node.parent).conf.y
            x2 = node.conf.x
            y2 = node.conf.y
            plt[:plot]([x1, x2], [y1, y2], color="k", alpha=alpha)
        end
    end
end

function render(rendering::PlotRendering, wall::Wall)
    render(rendering, wall.poly)
end

function render(rendering::PlotRendering, tree::Tree)
    render(rendering, tree.poly)
end

function render(rendering::PlotRendering, trace::Trace)
    ax = plt[:gca]()
    #print(trace)

    if hasvalue(trace, "scene")
        scene = value(trace, "scene")
        render(rendering, scene)
    end

    if hasvalue(trace, "start")
        start = value(trace, "start")
        ax[:scatter]([start.x], [start.y], color="blue", s=200)
    end

    if hasvalue(trace, "destination")
        dest = value(trace, "destination")
        ax[:scatter]([dest.x], [dest.y], color="red", s=200)
    end

    if hasvalue(trace, "tree")
        tree = value(trace, "tree")
        render(rendering, tree, 1.0)
    end

    if hasvalue(trace, "optimized_path")
        optimized_path = value(trace, "optimized_path")
        if !isnull(optimized_path)
            render(rendering, get(optimized_path), "purple")
        end
    end

    if hasvalue(trace, "path")
        path = value(trace, "path")
        if !isnull(path)
            render(rendering, get(path), "orange")
        end
    end

    if hasvalue(trace, "locations")
        locations = value(trace, "locations")
        if !isnull(loctaions)
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
        if hasconstraint(trace, "x$i") && hasconstraint(trace, "y$i")
            push!(measured_xs, value(trace, "x$i"))
            push!(measured_ys, value(trace, "y$i"))
        end
    end
    plt[:scatter](measured_xs, measured_ys, color="orange", s=200)
end

function render_samples(rendering::PlotRendering, particles::Array{Trace})

    times = value(particles[1], "times")

    # don't show the tree for any particles
    for trace in particles
        delete!(trace, "tree")
        delete!(trace, "path")
        delete!(trace, "locations")
    end

    for trace in particles[2:end]
        for i=1:length(times)
            delete!(trace, "x$i")
            delete!(trace, "y$i")
        end
        delete!(trace, "scene")
        delete!(trace, "start")
    end

    for trace in particles
        render(rendering, trace)
    end
end
