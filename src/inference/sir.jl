struct SIRGenerator{T,U} <: Generator{DictTrace}
    proposal::Generator{T}
    model::Generator{U}
    model_data_addresses
    model_latents_to_proposal_outputs::Dict
    model_args::Tuple
    num_particles::Int
    model_outputs::AddressTrie
    model_conditions::AddressTrie
    proposal_outputs::AddressTrie
    proposal_conditions::AddressTrie
end

empty_trace(::SIRGenerator) = DictTrace()

function make_model_query(model_data_addresses, model_latents_to_proposal_outputs)
    model_outputs = AddressTrie(model_data_addresses...)
    for addr in keys(model_latents_to_proposal_outputs)
        push!(model_outputs, addr)
    end
    model_conditions = AddressTrie()
    return (model_outputs, model_conditions)
end

function make_proposal_query(model_latents_to_proposal_outputs)
    proposal_outputs = AddressTrie(values(model_latents_to_proposal_outputs)...)
    proposal_conditions = AddressTrie()
    return (proposal_outputs, proposal_conditions)
end

function SIRGenerator(proposal::Generator{T}, model::Generator{U}, model_data_addresses, 
                      model_latents_to_proposal_outputs::Dict, model_args::Tuple, num_particles::Int) where {T,U}
    (model_outputs, model_conditions) = make_model_query(model_data_addresses, model_latents_to_proposal_outputs)
    (proposal_outputs, proposal_conditions) = make_proposal_query(model_latents_to_proposal_outputs)
    SIRGenerator(proposal, model, model_data_addresses, model_latents_to_proposal_outputs,
                 model_args, num_particles,
                 model_outputs, model_conditions, proposal_outputs, proposal_conditions)
end

function check_query(g::SIRGenerator, outputs, conditions)
    # no conditions are allowed
    if !isempty(conditions)
        error("Invalid query: Conditions is not empty.")
    end

    # the outputs of the query must be all the model latent addresses
    for model_addr in keys(g.model_latents_to_proposal_outputs)
        if !(model_addr in outputs)
            error("Invalid query: $model_addr is a model latent but not an output of SIR.")
        end
    end
    for model_addr in outputs
        if !haskey(g.model_latents_to_proposal_outputs, model_addr)
            error("Invalid query: $model_addr is an output of SIR but not a model latent.")
        end
    end
end

function make_model_trace(g::SIRGenerator{T,U}, model_data) where {T,U}
    model_trace = empty_trace(g.model)
    for (addr, value) in model_data
        model_trace[addr] = value
    end
    return model_trace
end

function make_output_value(g::SIRGenerator{T,U}, proposal_trace::T) where {T,U}
    value = Dict()
    for (model_addr, proposal_addr) in g.model_latents_to_proposal_outputs
        value[model_addr] = proposal_trace[proposal_addr]
    end
    return value
end

function populate_trace!(g::SIRGenerator{T,U}, proposal_trace::T, trace::DictTrace) where {T,U}
    for (model_addr, proposal_addr) in g.model_latents_to_proposal_outputs
        trace[model_addr] = proposal_trace[proposal_addr]
    end
end

function compute_score(model_scores::Vector{Float64}, log_weights::Vector{Float64}, chosen::Int)
    num_particles = length(model_scores)
    return model_scores[chosen] - (logsumexp(log_weights) - log(num_particles))
end

function copy_proposal_to_model!(g::SIRGenerator{T,U}, proposal_trace::T, model_trace::U) where {T,U}
    for (model_addr, proposal_addr) in g.model_latents_to_proposal_outputs
        model_trace[model_addr] = proposal_trace[proposal_addr]
    end
end

function copy_model_to_proposal!(g::SIRGenerator{T,U}, model_trace::U, proposal_trace::T) where {T,U}
    for (model_addr, proposal_addr) in g.model_latents_to_proposal_outputs
        proposal_trace[proposal_addr] = model_trace[model_addr]
    end
end

function check_model_data(g::SIRGenerator, model_data)
    for addr in g.model_data_addresses
        if !haskey(model_data, addr)
            error("model data address mismatch")
        end
    end
    for addr in keys(model_data)
        if !(addr in g.model_data_addresses)
            error("model data address mismatch")
        end
    end
end


function simulate!(g::SIRGenerator{T}, args::Tuple, outputs, conditions, trace::DictTrace) where {T}
    
    # we only accept one query
    check_query(g, outputs, conditions)

    # map from model address to value, for each observed random choice
    model_data = args[1]
    check_model_data(g, model_data)

    log_weights = fill(NaN, g.num_particles)
    model_scores = fill(NaN, g.num_particles)
    proposal_traces = Vector{T}(g.num_particles)
    for i=1:g.num_particles

        # sample latents from proposal, estimate proposal probability
        proposal_traces[i] = empty_trace(g.proposal)
        proposal_score, _ = simulate!(g.proposal, (model_data,), g.proposal_outputs, g.proposal_conditions, proposal_traces[i])

        # copy latents from proposal trace to model trace
        model_trace = make_model_trace(g, model_data)
        copy_proposal_to_model!(g, proposal_traces[i], model_trace)

        # estimate model joint probability of latents and data
        model_scores[i], _ = regenerate!(g.model, g.model_args, g.model_outputs, g.model_conditions, model_trace)

        # log importance weight
        log_weights[i] = model_scores[i] - proposal_score
    end
    @assert !any(isnan.(log_weights))
    @assert !any(isnan.(model_scores))

    chosen = categorical_log(log_weights)
    chosen_trace = proposal_traces[chosen]

    score = compute_score(model_scores, log_weights, chosen)

    # output is the mapping from model latents to value
    value = make_output_value(g, chosen_trace)
    
    # the trace of SIR is a DictTrace whose addresses are just the model latent addresses
    populate_trace!(g, chosen_trace, trace)

    return (score, value)
end

function handle_distinguished_particle(g::SIRGenerator{T}, trace::DictTrace, model_trace, model_data) where {T}
    proposal_trace = empty_trace(g.proposal)
    copy_model_to_proposal!(g, trace, proposal_trace)
    proposal_score, _ = regenerate!(g.proposal, (model_data,), g.proposal_outputs, g.proposal_conditions, proposal_trace)
    copy_proposal_to_model!(g, proposal_trace, model_trace)
    model_score, _ = regenerate!(g.model, g.model_args, g.model_outputs, g.model_conditions, model_trace)
    log_weight = model_score - proposal_score
    return (proposal_trace, model_score, log_weight)
end

function regenerate!(g::SIRGenerator{T}, args::Tuple, outputs, conditions, trace::DictTrace) where {T}
    
    # we only accept one query
    check_query(g, outputs, conditions)

    # map from model address to value, for each observed random choice
    model_data = args[1]
    check_model_data(g, model_data)

    log_weights = fill(NaN, g.num_particles)
    model_scores = fill(NaN, g.num_particles)
    proposal_traces = Vector{T}(g.num_particles)

    # handle the distinguished particle
    const CHOSEN = 1
    model_trace = make_model_trace(g, model_data)
    (proposal_traces[CHOSEN], model_scores[CHOSEN], log_weights[CHOSEN]) = handle_distinguished_particle(g, trace, model_trace, model_data)

    for i=2:g.num_particles

        # sample latents from proposal, estimate proposal probability
        proposal_traces[i] = empty_trace(g.proposal)
        proposal_score, _ = simulate!(g.proposal, (model_data,), g.proposal_outputs, g.proposal_conditions, proposal_traces[i])

        # copy latents from proposal trace to model trace
        model_trace = make_model_trace(g, model_data)
        copy_proposal_to_model!(g, proposal_traces[i], model_trace)

        # estimate model joint probability of latents and data
        model_scores[i], _ = regenerate!(g.model, g.model_args, g.model_outputs, g.model_conditions, model_trace)

        # log importance weight
        log_weights[i] = model_scores[i] - proposal_score
    end
    @assert !any(isnan.(log_weights))
    @assert !any(isnan.(model_scores))

    chosen_trace = proposal_traces[CHOSEN]

    score = compute_score(model_scores, log_weights, CHOSEN)

    # output is the mapping from model latents to value
    value = make_output_value(g, chosen_trace)
    
    return (score, value)
end

export SIRGenerator
