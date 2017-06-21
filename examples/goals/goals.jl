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
    render_num_threads = 60

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
    
    trace = Trace()
    #render(trace, "frames/frame_$f.png") ; f += 1

    intervene!(trace, "is-tree-1", true)
    intervene!(trace, "tree-1", Tree(Point(30, 20), 10.))
    intervene!(trace, "is-tree-2", true)
    intervene!(trace, "tree-2", Tree(Point(83, 80), 10.))
    intervene!(trace, "is-tree-3", true)
    intervene!(trace, "tree-3", Tree(Point(80, 40), 10.))
    #render(trace, "frames/frame_$f.png") ; f += 1

    wall_height = 30.
    intervene!(trace, "is-wall-1", true)
    intervene!(trace, "wall-1", Wall(Point(20., 40.), 1, 40., 2., wall_height))
    intervene!(trace, "is-wall-2", true)
    intervene!(trace, "wall-2", Wall(Point(60., 40.), 2, 40., 2., wall_height))
    intervene!(trace, "is-wall-3", true)
    intervene!(trace, "wall-3", Wall(Point(60.-15., 80.), 1, 15. + 2., 2., wall_height))
    intervene!(trace, "is-wall-4", true)
    intervene!(trace, "wall-4", Wall(Point(20., 80.), 1, 15., 2., wall_height))
    intervene!(trace, "is-wall-5", true)
    intervene!(trace, "wall-5", Wall(Point(20., 40.), 2, 40., 2., wall_height))

    boundary_wall_height = 2.
    intervene!(trace, "is-wall-6", true)
    intervene!(trace, "wall-6", Wall(Point(0., 0.), 1, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-7", true)
    intervene!(trace, "wall-7", Wall(Point(100., 0.), 2, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-8", true)
    intervene!(trace, "wall-8", Wall(Point(0., 100.), 1, 100., 2., boundary_wall_height))
    intervene!(trace, "is-wall-9", true)
    intervene!(trace, "wall-9", Wall(Point(0., 0.), 2, 100., 2., boundary_wall_height))

    # prevent the program from adding new wall or trees
    intervene!(trace, "is-tree-4", false)
    intervene!(trace, "is-wall-10", false)
    #render(trace, "frames/frame_$f.png") ; f += 1

    # add the drone starting position
    constrain!(trace, "start", Point(90., 10.))
    #render(trace, "frames/frame_$f.png") ; f += 1

    # add a known ground truth goal
    intervene!(trace, "measurement_noise", 1.0)
    intervene!(trace, "destination", Point(40., 60.))
    
    # show scene A, start, and goal position
    #render(trace, "frames/frame_$f.png") ; f += 1

    # show some simulations
    for i=1:10
        agent_model(trace)
        #render(trace, "frames/frame_$f.png") ; f += 1
    end
    times = value(trace, "times")

    # fresh trace with a 'random' goal location
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
    num_particles = 120 # TODO try 240 particles?

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
    traces = pmap((trace) -> mh_inference(trace, 10000), traces)
    render(traces, "frames/frame_$f.png", num_obs) ; f += 1

    # neural network experiments
    println("neural network experiments...")
    trace = generate_scene_a()
    intervene!(trace, "times", value(synthetic_data_trace, "times"))
    for i=1:num_obs
        constrain!(trace, "x$i", measured_xs[i])
        constrain!(trace, "y$i", measured_ys[i])
    end

    # fix use-waypoint = true for the neural experiments
    intervene!(trace, "use-waypoint", true) # TODO to make a fair comparison...

    # render just the observations, without the dot
    delete!(trace, "destination")
    #render(trace, "frames/frame_$f.png") ; f += 1

    # load the neural network
    max_iter = 10^4
    parameters = load_waypoint_neural_network("waypoint_network_scene_a_$(max_iter).json")
    max_t = 15
    inference = WaypointNetwork(60, 60, max_iter, max_t, 100)

    # show neural network predictions of the waypoint for our observed data set
    println("making predictions..")
    num_predictions = 60
    frame = PovrayRendering(camera_location, camera_look_at, light_location)
    frame.quality = 10
    frame.num_threads = 60
    render_trace(frame, trace)
    for j=1:num_predictions
        (waypoint, _) = neural_network_predict(inference, parameters, trace)
        render_waypoint(frame, waypoint)
    end
    finish(frame, "frames/frame_$f.png") ; f += 1

    for num_neural_mh_iter in [10, 50]
        println("neurally assisted MH with $num_neural_mh_iter iteratons")
        traces = pmap((i) -> mh_neural_inference(trace, num_neural_mh_iter, inference, parameters), 1:num_particles)
        render(traces, "frames/frame_$f.png", max_t) ; f += 1
        println("timing neural mh with $num_neural_mh_iter iterations..")
        for i=1:10
            @time mh_neural_inference(trace, num_neural_mh_iter, inference, parameters)
        end
    end

    for num_prior_mh_iter in [7, 33]
        println("prior assisted MH with $num_prior_mh_iter iteratons")
        traces = pmap((i) -> mh_inference(trace, num_prior_mh_iter), 1:num_particles)
        render(traces, "frames/frame_$f.png", max_t) ; f += 1
        println("timing prior mh with $num_prior_mh_iter iterations")
        for i=1:10
            @time mh_inference(trace, num_prior_mh_iter)
        end
    end


end

#srand(4)
#train_use_waypoint_network()
#train_waypoint_network()

srand(3)
demo()


