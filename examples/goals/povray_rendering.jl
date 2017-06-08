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
        show_axes = true
        scale = 0.5
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
    grass = vapory.Plane([0, 0, 1], 0.0, 
        vapory.Texture(
            vapory.Pigment("color", [0.35, 0.35, 0.35] * 0.85), 
            vapory.Normal("bumps", 0.75 ,"scale", 0.15), # 0.015
            vapory.Finish("phong", 0.1)))
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
    z = 10
    push!(scene.objects, vapory.Object("quadrocopter", "scale", 2.0 * scene.scale,
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
    color_rgbt = [0.5, 0.5, 0.5, 0.0]
    push!(scene.objects, vapory.Box(
        [corner1.x * scene.scale, corner1.y * scene.scale, 0 * scene.scale], 
        [corner2.x * scene.scale, corner2.y * scene.scale, wall.height * scene.scale],
        vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt))))
end

function render(scene::PovrayRendering, path::Path)
    height = 10.0 # TODO ??
    width = 0.2
    color_rgbt=[1, 1, 1, 0.3]
    for i=1:length(path.points)-1
        segment_start = path.points[i]
        segment_end = path.points[i+1]
        println("rendering path from $segment_start to $segment_end")

        # add a flat ribbon
        # compute the normal vector in the x-y plane
        forward = array(segment_end) - array(segment_start)
        forward = forward / norm(forward)
        side = [-forward[2], forward[1]]
        @assert isapprox(norm(side), 1.0)
        offset_ribbon = [side[1], side[2], 0]
        println("offset_ribbon: $offset_ribbon, scene.scale: $(scene.scale), width: $width")
        p1 = ([segment_start.x * scene.scale, segment_start.y * scene.scale, scene.scale * height]
             + scene.scale * width * offset_ribbon)
        p2 = ([segment_start.x * scene.scale, segment_start.y * scene.scale, scene.scale * height]
             - scene.scale * width * offset_ribbon)
        p3 = ([segment_end.x * scene.scale, segment_end.y * scene.scale, scene.scale * height]
             - scene.scale * width * offset_ribbon)
        p4 = ([segment_end.x * scene.scale, segment_end.y * scene.scale, scene.scale * height]
             + scene.scale * width * offset_ribbon)
        println("$p1, $p2, $p3, $p4")
        push!(scene.objects, vapory.Polygon(4, p1, p2, p3, p4, 
            vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt), vapory.Finish("ambient", 1.0)),
            "no_shadow"))
    end
end


function render_destination(scene::PovrayRendering, location::Point)
    radius = 0.5 # TODO??
    color_rgbt=[1.0, 0.0, 0.0, 0.6]
    location = array(location) * scene.scale
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
    vapory_scene = vapory.Scene(scene.camera, objects=scene.objects, included=scene.included, global_settings=scene.global_settings)
    vapory_scene[:render](fname, width=scene.width, height=scene.height, 
                antialiasing=scene.antialiasing, quality=scene.quality, 
                other_opts=[("+WT%d", scene.num_threads)])

end

# design pattern: a given rendering (in this case a scene::PovrayRendering) is opened.
# then we composite multiple traces onto this rendering, manually controlling which aspects of each trace are renderered
function render(povray_scene::PovrayRendering, trace::Trace)

    if hasvalue(trace, "scene")
        println("rendering scene..")
        scene = value(trace, "scene")
        add_grass(povray_scene)
        for object in scene.obstacles
            render(povray_scene, object)
        end
    end

    if hasvalue(trace, "start")
        println("rendering start..")
        start = value(trace, "start")
        render_agent(povray_scene, start, [1, 0.5, 0.5])
    end

    if hasvalue(trace, "destination")
        println("rendering destination..")
        dest = value(trace, "destination")
        render_destination(povray_scene, dest)
    end

    # TODO render this?
    #if hasvalue(trace, "tree")
        #tree = value(trace, "tree")
        #render(tree, 1.0)
    #end

    if hasvalue(trace, "optimized_path")
        println("rendering optimized path..")
        optimized_path = value(trace, "optimized_path")
        if !isnull(optimized_path)
            render(povray_scene, get(optimized_path))
            #render(get(optimized_path), "purple")
        end
    end

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

    times = value(trace, "times")
    measured_xs = []
    measured_ys = []
    for i=1:length(times)
        if hasconstraint(trace, "x$i") && hasconstraint(trace, "y$i")
            println("rendering (x$i, y$i)..")
            location = Point(value(trace, "x$i"), value(trace, "y$i"))
            render_agent(povray_scene, location, [1, 1, 1])
        end
    end
end

function render_samples(rendering::PovrayRendering, particles::Array{Trace})

    times = value(particles[1], "times")
    println(times)

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

    for (i, trace) in enumerate(particles)
        println("rendering trace $i")
        render(rendering, trace)
    end
end
