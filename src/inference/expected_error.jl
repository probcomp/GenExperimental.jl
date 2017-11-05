"""
    inference Generator must take a single argument defining a dataset simulated from the model.
    the data is a Dict mapping model addresses for data points to values
"""
struct ExpectedErrorAnalysis
    model::Generator
    inference::Generator
    model_latents_to_inference_outputs::Dict
    model_data_addresses
    model_args::Tuple
    model_addresses_to_score::AddressTrie
    inference_addresses_to_score::AddressTrie
end

function ExpectedErrorAnalysis(model::Generator, inference::Generator,
                                  model_latents_to_inference_outputs::Dict,
                                  model_data_addresses,
                                  model_args::Tuple)

    # the set of addresses to score for the model
    model_addresses_to_score = AddressTrie()
    for addr in model_data_addresses
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

    ExpectedErrorAnalysis(model, inference, model_latents_to_inference_outputs,
                             model_data_addresses, model_args, model_addresses_to_score,
                             inference_addresses_to_score)
end

function estimate_expected_error(experiment::ExpectedErrorAnalysis)
    
    # simulate model, scoring both latents and outputs
    model_simulate_trace = empty_trace(experiment.model)
    model_simulate_score, _ = simulate!(experiment.model, experiment.model_args,
                                        experiment.model_addresses_to_score, AddressTrie(),
                                        model_simulate_trace)
    
    # construct arguments to the inference algorithm from the data 
    data = Dict()
    for addr in experiment.model_data_addresses
        data[addr] = model_simulate_trace[addr]
    end

    # regenerate inference, scoring outputs
    inference_regenerate_trace = empty_trace(experiment.inference)
    for (model_addr, inference_addr) in experiment.model_latents_to_inference_outputs
        inference_regenerate_trace[inference_addr] = model_simulate_trace[model_addr]
    end
    inference_regenerate_score, _ = regenerate!(experiment.inference, (data,),
                                                experiment.inference_addresses_to_score, AddressTrie(),
                                                inference_regenerate_trace)
    
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
        model_regenerate_trace[model_addr] = model_simulate_trace[model_addr]
    end
    model_regenerate_score, _ = regenerate!(experiment.model, experiment.model_args,
                                            experiment.model_addresses_to_score, AddressTrie(),
                                            model_regenerate_trace)

    # unbiased estimate of the expected symmetric KL divergence between model
    # posterior and inference sampling distribution
    # log p(z, x) - log q(z; x) + log q(z'; x) - log p(z', x)
    # for z, x ~ p(.,.); z'|x ~ q(.; x)
    return model_simulate_score - inference_regenerate_score + inference_simulate_score - model_regenerate_score
end

export ExpectedErrorAnalysis
export estimate_expected_error
