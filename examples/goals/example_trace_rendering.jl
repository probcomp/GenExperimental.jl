# Instructions:
#
# 1. ensure povray executable is on the PATH
#
# 2. ensure vapory Python package is on the PYTHONPATH
#
# 3. set the environment variable POVRAY_INC_DIR to point to
#    $(pwd)/povray_include before running this script
#
# 4. run the script with julia <script-name>


# contains types for elements of the trace that get rendered (Wall, Tree, Path)
include("path_planner.jl")

# functions that construct PovRay scene from a trace
include("povray_rendering.jl")

# contains the model program, which fills in traces
include("model.jl")

# contains our fixed example scenes
include("scenes.jl")

function example_single_trace_rendering()

    # fix random seed
    srand(1)

    # start with a trace containing a fixed scene
    # note that the start location is already fixed as a constraint
    # to un-constrain or un-intervene use delete!(trace, "start")
    trace = generate_scene_a()

    # fix the destination
    intervene!(trace, "destination", Point(40., 60.))

    # fix the measurement noise to a lower value than defined in the program
    intervene!(trace, "measurement_noise", 1.0)
    
    # run the model program, populating the trace, subject to interventions
    agent_model(trace)
    
    # location of the camera, where it is looking, and the light source
    camera_location = [50., -30., 120.]
    camera_look_at = [50., 50., 0.]
    light_location = [50., 50., 150.]
    
    # set up the povray scene (see povray_rendering.jl)
    frame = PovrayRendering(camera_location, camera_look_at, light_location)
    frame.quality = 10 # set to 1 for a quicker rendering
    frame.num_threads = 4
    render_trace(frame, trace)

    # actually call povray and write image to file
    fname = "example_single_trace_rendering.png"
    finish(frame, fname)
end

function example_multiple_trace_rendering()

    # fix random seed
    srand(1)

    # generate a vector of traces, which are independen given interventions
    # intervene on the start positions, but not the destinations
    traces = Array{Trace,1}()
    for i=1:60
        trace = generate_scene_a()
        intervene!(trace, "measurement_noise", 1.0)
        agent_model(trace)
        push!(traces, trace)
    end

    # location of the camera, where it is looking, and the light source
    camera_location = [50., -30., 120.]
    camera_look_at = [50., 50., 0.]
    light_location = [50., 50., 150.]

    # instructs rendering to only draw the first 15 measurements
    max_number_of_measurements_to_draw = 15
    
    # set up the povray scene (see povray_rendering.jl)
    # the scene and measurements and destination are rendered from trace 1
    # only the destination is rendered for traces 2-end
    frame = PovrayRendering(camera_location, camera_look_at, light_location)
    frame.quality = 10 # set to 1 for a quicker rendering
    frame.num_threads = 4
    render_traces(frame, traces, max_number_of_measurements_to_draw)

    # actually call povray and write image to file
    fname = "example_multiple_trace_rendering.png"
    finish(frame, fname)
end

example_single_trace_rendering()
example_multiple_trace_rendering()
