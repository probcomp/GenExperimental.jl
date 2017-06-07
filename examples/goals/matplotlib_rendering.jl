using PyPlot

immutable PlotRendering
end

function render(rendering::PlotRendering, trace::Trace)
    ax = plt[:gca]()
    #print(trace)

    if hasvalue(trace, "scene")
        scene = value(trace, "scene")
        render(scene)
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
        render(tree, 1.0)
    end

    if hasvalue(trace, "optimized_path")
        optimized_path = value(trace, "optimized_path")
        if !isnull(optimized_path)
            render(get(optimized_path), "purple")
        end
    end

    if hasvalue(trace, "path")
        path = value(trace, "path")
        if !isnull(path)
            render(get(path), "orange")
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
