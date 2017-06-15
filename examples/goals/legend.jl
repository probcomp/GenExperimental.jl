@everywhere using Gen
@everywhere using PyCall
using PyPlot

@everywhere include("path_planner.jl")
@everywhere include("model.jl")
@everywhere include("scenes.jl")
include("povray_rendering.jl")

function make_legend()
    camera_location = [50., 50., 30.]
    camera_look_at = [50., 50., 0.]
    light_location = [00., 50., 80.]
    render_quality = 10
    render_num_threads = 4

    # just render the start position (blue drone)
    frame = PovrayRendering(camera_location,
                            camera_look_at,
                            light_location)
    grass = vapory.Plane([0, 0, 1.], 0., 
        vapory.Texture(
            vapory.Pigment("color", [209/255.,211/255.,212/255.])))
    push!(frame.objects, grass)
    render_agent(frame, Point(50.,49.5), [0, 0, 1])
    #finish(frame, "legend_start.png")

    # just render the observed location (orange drone)
    frame = PovrayRendering(camera_location,
                            camera_look_at,
                            light_location)
    grass = vapory.Plane([0, 0, 1.], 0., 
        vapory.Texture(
            vapory.Pigment("color", [209/255.,211/255.,212/255.])))
    push!(frame.objects, grass)
    render_agent(frame, Point(50.,49.5), [1, 0.5, 0.5])
    #finish(frame, "legend_observed.png")

    # just render the destinations
    frame = PovrayRendering(camera_location,
                            camera_look_at,
                            light_location)
    grass = vapory.Plane([0, 0, 1.], 0., 
        vapory.Texture(
            vapory.Pigment("color", [209/255.,211/255.,212/255.])))
    push!(frame.objects, grass)
    for point in [Point(52., 47.5), Point(53., 52.5), Point(46.8, 51.2)]
        radius = 1.0
        color_rgbt=[1.0, 0.0, 0.0, 0.0]
        location = frame.scale * [point.x, point.y, frame.path_height]
        push!(frame.objects, vapory.Sphere(location, radius * frame.scale,
            vapory.Texture(vapory.Pigment("color", "rgbt", color_rgbt), vapory.Finish("ambient", 0.6)),
            "no_shadow"))
    end
    finish(frame, "legend_dest.png")
  

end

make_legend()
