using Gen

function encapsulate(program, output_names::Array{String,1})
    simulator = function(args...) # TODO use the type signature of the original program
        trace = Trace()
        program(trace, args...)
        map((name) -> trace.vals[name], output_names), trace.log_weight # TODO check log weight
    end
    regenerator = function(output_vals::Array, args...)
        trace = Trace()
        for (name, val) in zip(output_names, output_vals)
            trace.vals[name] = val
        end
        program(trace, args...)
        trace.log_weight
    end
    (simulator, regenerator)
end

function generic_encapsulation_demo()
    linreg_simulator, linreg_regenerator = encapsulate(linear_regression, ["y1", "y2", "y3"])
    outputs, log_weight = linreg_simulator(0.0, 2.0, [0.1, 0.2, 0.3])
    println("$outputs, $log_weight")
    log_weight = linreg_regenerator(outputs, 0.0, 2.0, [0.1, 0.2, 0.3])
    println(log_weight)
    @register_module(:linreg, linreg_simulator, linreg_regenerator)
end
#generic_encapsulation_demo()

# ---------- encapsulation using an inference module -------------------- #
# all arguments of program get along passed to the module
function encapsulate(program, output_names::Array{String,1},
                     inference_module, inferred_names::Array{String,1})
    inference_simulator, inference_regenerator = inference_module
    simulator = function(args...) # TODO use the type signature of the original program
        trace = Trace()
        program(trace, args...)
        inference_inputs = map((name) -> trace.vals[name], output_names)
        inference_outputs = map((name) -> trace.vals[name], inferred_names)
        inference_log_weight = inference_regenerator(inference_outputs, inference_inputs, args...)
        program_outputs = map((name) -> trace.vals[name], output_names)
        program_outputs, (trace.log_weight + inference_log_weight) # TODO check
    end
    regenerator = function(output_vals::Array, args...)
        inference_inputs = output_vals
        inference_outputs, inference_log_weight = inference_simulator(inference_inputs, args...)
        trace = Trace()
        # put outputs of program into program trace
        for (name, val) in zip(output_names, output_vals)
            trace.vals[name] = val
        end
        # put outputs of inference into program trace
        for (name, val) in zip(inferred_names, inference_outputs)
            trace.vals[name] = val
        end
        # run program, to collect log_weight
        program(trace, args...)
        trace.log_weight - inference_log_weight # TODO check
    end
    (simulator, regenerator)
end

# use a RANSAC algorithm to predict slope and intercept
# TODO: add autodifferentiation and tune the noise of RANSAC using stochastic gradient descent :)
# TODO: also show a big third-party RANSAC implementation being used for the core, with output variance tuning using SGD

#function make_ransac_module(iters::Int, subset_size::Int, eps::Float64,
#                            slope_std::Float64, intercept_std::Float64)
    #function ransac_simulate(# output variables of the model program
                             #ys::Array{Float64,1},
                             ## parameters of the model program
                             #prior_mu::Float64, prior_std::Float64, xs::Array{Float64,1})
        #best_slope, best_intercept = ransac_core(xs, ys, iters, subset_size, eps)
        ## add some noise to output
        #output_slope, log_weight_slope = normal_simulate(best_slope, slope_std)
        #output_intercept, log_weight_intercept = normal_simulate(best_intercept, intercept_std)
        #[output_slope, output_intercept], log_weight_slope + log_weight_intercept
    #end
    #function ransac_regenerate(# inferred variables of the model program
                               #slope_and_intercept::Array{Float64,1},
                               ## output variables of the model program
                               #ys::Array{Float64,1},
                               ## parameters of the model program
                               #prior_mu::Float64, prior_std::Float64, xs::Array{Float64,1})
        #slope, intercept = slope_and_intercept
        #best_slope, best_intercept = ransac_core(xs, ys, iters, subset_size, eps)
        ## evaluate log density of output given core value
        #log_weight_slope = normal_regenerate(intercept, best_slope, slope_std)
        #log_weight_intercept = normal_regenerate(slope, best_intercept, intercept_std)
        #log_weight_slope + log_weight_intercept
    #end
    #(ransac_simulate, ransac_regenerate)
#end

#function ransac_demo()
    #linreg_simulator, linreg_regenerator = encapsulate(linear_regression, ["y1", "y2", "y3"],
                                                       #make_ransac_module(10, 2, 0.1, 0.1, 0.1),
                                                       #["slope", "intercept"])
    #outputs, log_weight = linreg_simulator(0.0, 2.0, [0.1, 0.2, 0.3])
    #println("$outputs, $log_weight")
    #log_weight = linreg_regenerator(outputs, 0.0, 2.0, [0.1, 0.2, 0.3])
    #println(log_weight)
    #@register_module(:linreg, linreg_simulator, linreg_regenerator)
#end
#ransac_demo()

# --------------------------- MCMC inference -------------------------- #



# --------------- SMC inference and inference module -------------------#
#using SMC
#
#immutable LinregParams
    #prior_mu::Float64
    #prior_std::Float64
    #xs::Array{Float64,1}
#end
#
#immutable LinregSMCState 
    #params::LinregParams
    #trace::Trace
    #ys::Array{Float64,1}
#end
#
#immutable LinregSMCInitializer 
    #params::LinregParams
    #ys::Array{Float64,1}
#end
#
#function delete_ys_from_trace!(trace::Trace, n::Int)
    #for i=1:n
        ## todo: try only adding those that you request to the trace, so that deleting
        ## is not necessary
        #delete!(trace.vals, "y$i") 
    #end
#end
#
#function SMC.forward(init::LinregSMCInitializer)
    ## sample from prior, weight by likelihood of first datum
    ## note: can just pass in prefix of xs for a performance optimization
    #trace = Trace()
    #trace.vals["y1"] = init.ys[1]
    #linear_regression(trace, init.prior_mu, init.prior_std, init.xs)
    #delete_ys_from_trace!(trace, length(xs))
    #state = LinregSMCstate(init.linreg_params, trace, init.ys)
    #(state, trace.log_weight)
#end
#
#function SMC.backward(init::LinregSMCInitializer, state::LinregSMCState)
    #tmp_state = deepcopy(state)
    #trace = tmp_state.trace
    #delete_ys_from_trace!(trace, length(xs)) # maybe unecessary
    #trace.vals["y1"] = init.ys[1]
    #linear_regression(trace, init.prior_mu, init.prior_std, init.xs)
    #trace.log_weight
#end
#
#immutable LinregSMCIncrementer
    #ys::Array{Float64,1}
    #rejuvenation_steps::Int
    #datum_index::Int
#end
#
#function resimulation_mh_step!(state::LinregSMCState, max_datum_index::Int)
    ## get previous log_weight (todo: can be stored in the state for performance optimization?)
    #delete_ys_from_trace!(state.trace, length(xs))
    #for i=1:max_datum_index
        #state.trace.vals["y$i"] = state.params.xs[i]
    #end
    #linear_regression(state.trace, state.params.prior_mu, state.params.prior_std, state.params.xs)
    #prev_log_weight = state.trace.log_weight
    ## propose new values
    #proposed_trace = Trace()
    #for i=1:max_datum_index
        #proposed_trace.vals["y$i"] = state.params.xs[i]
    #end
    #linear_regression(proposed_trace, state.params.prior_mu, state.params.prior_std, state.params.xs)
    #proposed_log_weight = proposed_trace.log_weight
    #if log(rand()) <= proposed_log_weight - prev_log_weight
        ## accept
        #state.trace = proposed_trace
    #end
    #delete_ys_from_trace!(state.trace, length(xs))
#end
#
#function SMC.forward(incr::LinregSMCIncrementer, state::LinregSMCState)
    #new_state = deepcopy(state) # shouldn't be deepcopy.. only need to copy trace
    ## rejuvenation steps targeting the partial 
    ## posterior for data 1:incr.datum_index - 1
    #for step=1:incr.rejuvenation_steps
        #resimulation_mh_step!(new_state, incr.datum_index - 1)
    #end
    ## add the next datum_index
    #for i=1:incr.datum_index
        #new_state.trace.vals["y$i"] = state.params.xs[i]
    #end
    #linear_regression(new_state.trace, state.params.prior_mu,
                      #state.params.prior_std, state.params.xs)
	#(new_state, new_state.trace.log_weight)
#end
#
#function SMC.backward(incr::DPMMIncrementer, new_state::DPMMState)
    #state = deepcopy(new_state)
    ## compute log_likelihood of just this datum
    #if incr.proposal_type == optimal_proposal
	    #log_weight = remove_datum_conditional!(state, incr.datum_index, datum)
    #else
	    #log_weight = remove_datum_prior!(state, incr.datum_index, datum)
    #end
	#@assert num_assignments(state.crp) == incr.datum_index - 1
    ## Rejuvenation sweep (in reverse order)
    #for sweep=1:incr.rejuvenation_sweeps
        #nign_s_update!(state)
        #nign_nu_update!(state)
        #nign_r_update!(state)
        #nign_m_update!(state)
        #nign_params_update!(state)
        #for i=incr.datum_index-1:1
            #nign_s_update!(state)
            #nign_nu_update!(state)
            #nign_r_update!(state)
            #nign_m_update!(state)
            #nign_params_update!(state)
            #alpha_mh_update!(state)
            #gibbs_step!(state, i, incr.data[i])
        #end
    #end
	#(state, log_weight)
#end
#
#function DPMMSMCScheme(proposal_type::DPMM_CLUSTER_PROPOSAL_TYPE,
                       #data::Array{Float64,1},
                       #alpha_prior::Gamma,
                       #nign_params_prior::NIGNParamsPrior,
                       #rejuvenation_sweeps::Int,
                       #num_particles::Int)
    #initializer = DPMMInitializer(proposal_type, data[1], alpha_prior, nign_params_prior)
    #incrementers = Array{DPMMIncrementer,1}(length(data)-1)
    #for i=2:length(data)
        #incrementers[i-1] = DPMMIncrementer(proposal_type, data, i, rejuvenation_sweeps)
    #end
    #SMCScheme(initializer, incrementers, num_particles)
#end
#
## implement an SMC scheme defined by adding one data point at a time, with
## random walk rejuvenation kernels
#
#
## implement an SMC scheme defined by annealing the 
