@everywhere using Gen
@everywhere using Distributions
@everywhere using PyCall
#@pyimport matplotlib.patches as patches
import JSON
using PyPlot

@everywhere include("path_planner.jl")
@everywhere include("model.jl")
@everywhere include("scenes.jl")
include("povray_rendering.jl")
@everywhere include("inference.jl")

function demo()

    camera_location = [50., -30., 120.]
    camera_look_at = [50., 50., 0.]
    light_location = [50., 50., 150.]
    render_quality = 10
    render_num_threads = 4

    function render(trace::Trace, fname::String) 
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = render_quality
        frame.num_threads = render_num_threads
        render_trace(frame, trace)
        finish(frame, fname)
        println(fname)
    end

    function render(traces::Array{Trace,1}, fname::String, max_measurement_time::Int)
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = render_quality
        frame.num_threads = render_num_threads
        render_traces(frame, traces, max_measurement_time)
        finish(frame, fname)
        println(fname)
    end

    # frame index
    f = 0
    
    trace = generate_scene_a()

    # add a known ground truth goal
    intervene!(trace, "measurement_noise", 1.0)
    intervene!(trace, "destination", Point(40., 60.))
    
    # show scene A, start, and goal position
    #render(trace, "frames/frame_$f.png") ; f += 1

    # run model and extract observations, render ground truth
    # three times
    for i=1:10
        agent_model(trace)
        #render(trace, "frames/frame_$f.png") ; f += 1
    end
    times = value(trace, "times")

    # fresh trace with a random goal location
    trace = generate_scene_a()
    agent_model(trace)
    delete!(trace, "optimized_path")
    for i=1:length(times)
        delete!(trace, "x$i")
        delete!(trace, "y$i")
    end
    #render(trace, "frames/frame_$f.png") ; f += 1

    # add observations (generate the observations synthetically in another trace)
    # make the synthetic data be a waypoint path (so that inference has an obvious effect)
    synthetic_data_trace = generate_scene_a()
    intervene!(synthetic_data_trace, "use-waypoint", true)
    intervene!(synthetic_data_trace, "measurement_noise", 1.0)
    intervene!(synthetic_data_trace, "waypoint", Point(55.,8.))
    intervene!(synthetic_data_trace, "destination", Point(70., 90.))
    agent_model(synthetic_data_trace)
    num_obs = 15
    measured_xs = map((i) -> value(synthetic_data_trace, "x$i"), 1:num_obs)
    measured_ys = map((i) -> value(synthetic_data_trace, "y$i"), 1:num_obs)
    
    # show observations in the trace with the random goal locaiton
    intervene!(trace, "times", value(synthetic_data_trace, "times"))
    for i=1:num_obs
        constrain!(trace, "x$i", measured_xs[i])
        constrain!(trace, "y$i", measured_ys[i])
    end
    #render(trace, "frames/frame_$f.png") ; f += 1
   
    # show some inference happening (just showing the cloud of destination particles)
    num_particles = 60

    intervene!(trace, "use-waypoint", true)

    traces::Array{Trace, 1} = map((i) -> deepcopy(trace), 1:num_particles) # infer fork
    traces = map((trace) -> mh_inference(trace, 0), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 1), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 10), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    #traces = pmap((trace) -> mh_inference(trace, 100), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    #traces = pmap((trace) -> mh_inference(trace, 1000), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    #traces = pmap((trace) -> mh_inference(trace, 10000), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1

    # do the same with the sceond scene
    trace = generate_scene_b()
    #render(trace, "frames/frame_$f.png") ; f += 1

    intervene!(trace, "times", value(synthetic_data_trace, "times"))
    for i=1:num_obs
        constrain!(trace, "x$i", measured_xs[i])
        constrain!(trace, "y$i", measured_ys[i])
    end
    #render(trace, "frames/frame_$f.png") ; f += 1

    #render(trace, "frames/frame_$f.png") ; f += 1
    traces = map((i) -> deepcopy(trace), 1:num_particles) # infer fork
    traces = pmap((trace) -> mh_inference(trace, 0), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 1), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    traces = pmap((trace) -> mh_inference(trace, 10), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    #traces = pmap((trace) -> mh_inference(trace, 100), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    #traces = pmap((trace) -> mh_inference(trace, 1000), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1
    #traces = pmap((trace) -> mh_inference(trace, 10000), traces)
    #render(traces, "frames/frame_$f.png", num_obs) ; f += 1

    println("timing mh")
    for i=1:10
        @time mh_inference(trace, 10)
    end

    # neural network experiments
    println("neural network experiments...")
    scene_a_trace = generate_scene_a()
    #scene_b_trace = generate_scene_b()
    for (scene_name, scene_trace) in [("a", scene_a_trace)]#[("a", scene_a_trace), ("b", scene_b_trace)]

        println("using neural network for scene: $scene_name")
        if !hasvalue(scene_trace, "times")
            intervene!(scene_trace, "times", times)
        end

        # show neural network predictions for both scenes
        # the neural network was trained for a specific number of observations (max_t
        max_iter = 10^4 # was 4 # TODO
        parameters = load_waypoint_neural_network("waypoint_network_scene_$(scene_name)_$(max_iter).json")
        max_t = 15
        # TODO should be loaded from JSON alongside parameters
        inference = WaypointNetwork(60, 60, max_iter, max_t, 100)
    
        # constrain the prefix of observed data
        xs = measured_xs[1:inference.max_t]
        ys = measured_ys[1:inference.max_t]
        trace = deepcopy(scene_trace)
        for t=1:max_t
            constrain!(trace, "x$t", xs[t])
            constrain!(trace, "y$t", ys[t])
        end
        start = value(trace, "start")
    
        # show neural network predictions of the waypoint for our observed data set
        println("making predictions..")
        num_predictions = 60
        frame = PovrayRendering(camera_location, camera_look_at, light_location)
        frame.quality = 1
        frame.num_threads = 4
        render_trace(frame, trace)
        for j=1:num_predictions
            (waypoint, _) = neural_network_predict(inference, parameters, trace)
            render_waypoint(frame, waypoint)
        end
        #finish(frame, "frames/frame_$f.png") ; f += 1

        # use neural network as proposal and show trajectory predictions
        println("timing neural mh..")
        @assert length(xs) == max_t
        intervene!(trace, "use-waypoint", true)
        for i=1:10
            @time mh_neural_inference(trace, 10, inference, parameters)
        end
        println("neural MH scene a, max_time: $max_t")
        for num_iter in [10]
            println("num_iter: $num_iter, $num_particles")
            particles::Array{Trace,1} = pmap((i) -> mh_neural_inference(trace, num_iter, inference, parameters), 1:num_particles)
            #render(particles, "frames/frame_$f.png", max_t) ; f += 1
        end
    end

end

#srand(4)
#train_use_waypoint_network()
#train_waypoint_network()

srand(3)
demo()


