using PyCall
@pyimport vapory

type PovrayRendering
    width::Int
    height::Int
    antialiasing::Float64
    quality::Int
    num_threads::Int
    camera
    scale::Float64
    light
    objects::Array
    included::Array{String,1}
    global_settings::Array
    function PovrayRendering(camera_loc::Array{Float64,1},
                             camera_look_at::Array{Float64,1},
                             light_loc::Array{Float64,1})
        show_axes = false
        scale = 0.4
        povray_inc_dir=ENV["POVRAY_INC_DIR"]

        included = String[]
        #included.append("base.inc")
        # TODO: get more trees and correct textures for them from: "Trees by Arbaro"
        # http://www.f-lohmueller.de/pov_tut/objects/obj_430e.htm
        # the POV file
        push!(included, "$povray_inc_dir/wall_texture.inc")
        push!(included, "$povray_inc_dir/quaking_aspen_13m.inc")
        push!(included, "$povray_inc_dir/lombardy_poplar_13m.inc")
        
        # Model from: http://www.blendswap.com/blends/view/75754
        # exported from Blender to .stl
        # deleted all objects except for the basic frame in order to get correct export (can try later with other pieces included)
        # converted rom .stl to .pov using python stl2pov.py ../Quadrocopter2.stl ../Quadrocopter2.pov
        # then the object was named "quadcopter" appropriately, and the file was named an .inc file
        push!(included, "$povray_inc_dir/Quadrocopter.inc")
        push!(included, "$povray_inc_dir/quaking_aspen_textures.inc")

        camera = vapory.Camera(
                "location", camera_loc * scale,
                "look_at", camera_look_at * scale)
        light = vapory.LightSource(light_loc * scale,
                                   "color", [2, 2, 2])
        objects = [camera, light]
        #vapory.Polygon(4, p1, p2, p3, p4, 
            #vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt), vapory.Finish("ambient", 1.0)),
            #"no_shadow"))
        x_axis = vapory.Polygon(4, [0, 0, 0.1] * scale,
                                   [0, 1, 0.1] * scale,
                                   [100, 1, 0.1] * scale,
                                   [100, 0, 0.1] * scale,
                                   vapory.Texture(vapory.Pigment("color", [1, 0, 0]))) # red X
        y_axis = vapory.Polygon(4, [0, 0, 0.1] * scale,
                                   [1, 0, 0.1] * scale,
                                   [1, 100, 0.1] * scale,
                                   [0, 100, 0.1] * scale,
                                   vapory.Texture(vapory.Pigment("color", [0, 1, 0]))) # green Y
        if show_axes
            push!(objects, x_axis)
            push!(objects, y_axis)
        end
        global_settings = []

        # set defaults (TODO write setters for thesek, for now just set hte fields)
        width = 800
        height = 800
        antialiasing = 0.01
        quality = 10
        num_threads = 1
        new(width, height, antialiasing, quality, num_threads, camera, scale, light, objects, included, global_settings)
    end
end

function add_grass(scene::PovrayRendering)
    grass = vapory.Plane([0, 0, 1.], 0., 
        vapory.Texture(
            vapory.Pigment("color", [0.25, 0.45, 0.15] * 0.85), 
            vapory.Normal("bumps", 0.75 ,"scale", 0.15), # 0.015
            vapory.Finish("phong", 0.1),
            vapory.Finish("ambient", 0.5)))
    push!(scene.objects, grass)
end

function render(scene::PovrayRendering, tree::Tree)
    #Trees from http://www.f-lohmueller.de/pov_tut/objects/obj_430e.htm
    variant = "quaking_aspen_13"
    yaw = 0 #rand() * 350
    push!(scene.objects, vapory.Object("$(variant)_stems", 
        vapory.Texture("Stem_Texture"), 
        "rotate", [90, 0, yaw], 
        "translate", [tree.center.x * scene.scale, tree.center.y * scene.scale, 0]))
    push!(scene.objects, vapory.Object("$(variant)_leaves", 
        "double_illuminate", 
        vapory.Texture("Leaves_Texture_1"), 
        "rotate", [90, 0, yaw], 
        "translate", [tree.center.x * scene.scale, tree.center.y * scene.scale, 0]))
end

function render_agent(scene::PovrayRendering, location::Point, color)
    yaw = 0
    z = 20.
    push!(scene.objects, vapory.Object("quadrocopter", "scale", 3.0 * scene.scale,
        "translate", [0, 0, 0], # translate to origin first so we can rotate
        "rotate", [-15, 0, yaw], # the drone model is angled a bit
        # the drone model is centered at some vertical offset of about 1.0
        "translate", [location.x * scene.scale, location.y * scene.scale, scene.scale * z - 1.0],
        vapory.Texture(vapory.Pigment("color", color))))
end

function render(scene::PovrayRendering, wall::Wall)
    corner1 = wall.start
    if wall.orientation == 1 # x 
        corner2 = Point(corner1.x + wall.length, corner1.y + wall.thickness)
    else # y
        corner2 = Point(corner1.x + wall.thickness, corner1.y + wall.length)
    end
    color_rgbt = [0.5, 0.5, 0.5, 0.5]
    push!(scene.objects, vapory.Box(
        [corner1.x * scene.scale, corner1.y * scene.scale, 0 * scene.scale], 
        [corner2.x * scene.scale, corner2.y * scene.scale, wall.height * scene.scale],
        vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt))))
end

function render(scene::PovrayRendering, path::Path)
    height = 20.0 # TODO ??
    width = 0.2
    color_rgbt=[1, 1, 1, 0.1]
    for i=1:length(path.points)-1
        segment_start = path.points[i]
        segment_end = path.points[i+1]

        # add a flat ribbon
        # compute the normal vector in the x-y plane
        forward = array(segment_end) - array(segment_start)
        forward = forward / norm(forward)
        side = [-forward[2], forward[1]]
        @assert isapprox(norm(side), 1.0)
        offset_ribbon = [side[1], side[2], 0]
        p1 = ([segment_start.x * scene.scale, segment_start.y * scene.scale, scene.scale * height]
             + scene.scale * width * offset_ribbon)
        p2 = ([segment_start.x * scene.scale, segment_start.y * scene.scale, scene.scale * height]
             - scene.scale * width * offset_ribbon)
        p3 = ([segment_end.x * scene.scale, segment_end.y * scene.scale, scene.scale * height]
             - scene.scale * width * offset_ribbon)
        p4 = ([segment_end.x * scene.scale, segment_end.y * scene.scale, scene.scale * height]
             + scene.scale * width * offset_ribbon)
        push!(scene.objects, vapory.Polygon(4, p1, p2, p3, p4, 
            vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt), vapory.Finish("ambient", 1.0)),
            "no_shadow"))
    end
end


function render_destination(scene::PovrayRendering, location::Point)
    height = 20.0
    radius = 1.0
    color_rgbt=[1.0, 0.0, 0.0, 0.6]
    location = scene.scale * [location.x, location.y, height]
    push!(scene.objects, vapory.Sphere(location, radius * scene.scale,
        vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt), vapory.Finish("ambient", 0.6)),
        "no_shadow"))
end

function render(scene::PovrayRendering, out_fname::String, )
    scene = vapory.Scene(scene.camera, objects=scene.objects, included=scene.included, global_settings=scene.global_settings)
    scene.render(out_fname, width=width, height=height, 
            antialiasing=antialiasing, quality=quality, 
            other_opts=[("+WT%d", num_threads)])
end

function finish(scene::PovrayRendering, fname::String)
    println("finishing frame $fname")
    vapory_scene = vapory.Scene(scene.camera, objects=scene.objects, included=scene.included, global_settings=scene.global_settings)
    vapory_scene[:render](fname, width=scene.width, height=scene.height, 
                antialiasing=scene.antialiasing, quality=scene.quality, 
                other_opts=[("+WT%d", scene.num_threads)])

end

function add_trees(trace::Trace, povray_scene::PovrayRendering)
    i = 1
    while true
        if hasvalue(trace, "tree-$i")
            tree = value(trace, "tree-$i")
            render(povray_scene, tree)
            i += 1
        else
            break
        end
    end
end

function add_walls(trace::Trace, povray_scene::PovrayRendering)
    i = 1
    while true
        if hasvalue(trace, "wall-$i")
            wall = value(trace, "wall-$i")
            render(povray_scene, wall)
            i += 1
        else
            break
        end
    end
end

function add_start(trace::Trace, povray_scene::PovrayRendering)
    if hasvalue(trace, "start")
        start = value(trace, "start")
        render_agent(povray_scene, start, [0, 0, 1])
    end
end

function add_destination(trace::Trace, povray_scene::PovrayRendering)
    if hasvalue(trace, "destination")
        dest = value(trace, "destination")
        render_destination(povray_scene, dest)
    end
end

function add_optimized_path(trace::Trace, povray_scene::PovrayRendering)
    if hasvalue(trace, "optimized_path")
        optimized_path = value(trace, "optimized_path")
        if !isnull(optimized_path)
            render(povray_scene, get(optimized_path))
        end
    end
end

function add_measurements(trace::Trace, povray_scene::PovrayRendering)
    #println("add masurements")
    println("has times? $(hasvalue(trace, "times"))")
    if hasvalue(trace, "times")
        times = value(trace, "times")
        add_measurements(trace, povray_scene, length(times))
    end
end

function add_measurements(trace::Trace, povray_scene::PovrayRendering, max_measurement_time::Int)
    #println("add masurements")
    if hasvalue(trace, "times")
        times = value(trace, "times")
        #println("times: $times")
        measured_xs = []
        measured_ys = []
        println("max_measurement_time: $max_measurement_time")
        for i=1:max_measurement_time
            if hasvalue(trace, "x$i") && hasvalue(trace, "y$i")
            #if hasconstraint(trace, "x$i") && hasconstraint(trace, "y$i")
                location = Point(value(trace, "x$i"), value(trace, "y$i"))
                render_agent(povray_scene, location, [1, 0.5, 0.5])
            end
        end
    end
end




# design pattern: a given rendering (in this case a scene::PovrayRendering) is opened.
# then we composite multiple traces onto this rendering, manually controlling which aspects of each trace are renderered
function render_trace(povray_scene::PovrayRendering, trace::Trace)

    add_grass(povray_scene)
    add_trees(trace, povray_scene)
    add_walls(trace, povray_scene)
    add_start(trace, povray_scene)
    add_destination(trace, povray_scene)
    add_optimized_path(trace, povray_scene)
    add_measurements(trace, povray_scene)

    # TODO
    #if hasvalue(trace, "tree")
        #tree = value(trace, "tree")
        #render(tree, 1.0)
    #end

    # TODO render this?
    #if hasvalue(trace, "path")
        #path = value(trace, "path")
        #if !isnull(path)
            #render(get(path), "orange")
        #end
    #end

    # TODO render this?
    #if hasvalue(trace, "locations")
        #locations = value(trace, "locations")
        #if !isnull(loctaions)
            #measurement_noise = value(trace, "measurement_noise")
            #for loc in locations
                #patch = patches.Circle((loc.x, loc.y),
                                        #radius=measurement_noise, facecolor="green",
                                        #edgecolor="green", alpha=0.2, clip_on=true, zorder=-1)
                #ax[:add_artist](patch)
            #end
        #end
    #end

end

function render_traces(povray_scene::PovrayRendering, traces::Array{Trace}, max_measurement_time::Int)

    add_grass(povray_scene)

    # the traces are assumed to share the same scene and measuremnets
    # the traces only differ in the destination and the optimized path
    # only render the scene and measurements from the first 
    # render trees
    trace = traces[1]
    add_grass(povray_scene)
    add_trees(trace, povray_scene)
    add_walls(trace, povray_scene)
    add_start(trace, povray_scene)
    add_measurements(trace, povray_scene, max_measurement_time)

    for trace in traces
        add_destination(trace, povray_scene)
        add_optimized_path(trace, povray_scene)
    end

end
