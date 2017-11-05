"""
    inference Generator must take a single argument defining a dataset simulated from the model.
    the data is a Dict mapping model addresses for data points to values
"""
struct ELBOAnalysis
    model::Generator
    inference::Generator
    model_latents_to_inference_outputs::Dict
    data::Dict
    model_args::Tuple
    model_addresses_to_score::AddressTrie
    inference_addresses_to_score::AddressTrie
end

function ELBOAnalysis(model::Generator, inference::Generator,
                      model_latents_to_inference_outputs::Dict,
                      data::Dict,
                      model_args::Tuple)

    # the set of addresses to score for the model
    model_addresses_to_score = AddressTrie()
    for addr in keys(data) 
        push!(model_addresses_to_score, addr)
    end
    for addr in keys(model_latents_to_inference_outputs)
        push!(model_addresses_to_score, addr)
    end

    # the set of addresses to score for the inference
    inference_addresses_to_score = AddressTrie()
    for (_, inference_addr) in model_latents_to_inference_outputs
        push!(inference_addresses_to_score, inference_addr)
    end

    ELBOAnalysis(model, inference, model_latents_to_inference_outputs,
                 data, model_args, model_addresses_to_score,
                 inference_addresses_to_score)
end

function estimate_elbo(experiment::ELBOAnalysis)
    
    # simulate inference, scoring outputs
    inference_simulate_trace = empty_trace(experiment.inference)
    inference_simulate_score, _ = simulate!(experiment.inference, (data,),
                                            experiment.inference_addresses_to_score, AddressTrie(),
                                            inference_simulate_trace)
    
    # regenerate model, scoring outputs and latents
    model_regenerate_trace = empty_trace(experiment.model)
    for (model_addr, inference_addr) in experiment.model_latents_to_inference_outputs
        model_regenerate_trace[model_addr] = inference_simulate_trace[inference_addr]
    end
    for model_addr in experiment.model_data_addresses
        model_regenerate_trace[model_addr] = data[model_addr]
    end
    model_regenerate_score, _ = regenerate!(experiment.model, experiment.model_args,
                                            experiment.model_addresses_to_score, AddressTrie(),
                                            model_regenerate_trace)

    # unbiased estimate of a lower bound on the ELBO
    # log p(z', x) - log q(z'; x)
    # for  z'|x ~ q(.; x)
    return model_regenerate_score - inference_simulate_score
end

export ELBOAnalysis
export estimate_elbo
