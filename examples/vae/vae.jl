using Gen
import StatsBase

function sigmoid{T}(x::T)
    1.0 ./ (1.0 + exp(-x))
end

immutable Architecture
    num_latent = 2
    num_hidden = 50
    dim_data = 784
end

@program generative_network(arch::Architecture) begin
    num_hidden = arch.num_hidden
    num_latent = arch.num_latent
    num_data = arch.num_data

    latents = map((i) -> normal(0,1) ~ "latent-$i", 1:num_latent)

    hidden_b = zeros(num_hidden) ~ "hidden-biases"
    hidden_W = zeros(num_hidden, num_latent) ~ "hidden-weights"

    hidden_loadings = hidden_b + hidden_W' * latents
    hidden_activations = sigmoid(hidden_loadings)

    data_b = zeros(dim_data) ~ "data-biases"
    data_W = zeros(dim_data, num_hidden) ~ "data-weights"

    data_loadings =  data_b + data_W' * hidden_activations
    data_activations = sigmoid(hidden_loadings)
    
    data = map((i) -> flip(data_activations[i]), "x$i")
end

@program recognition_network(arch::Architecture, data::Vector{Bool}) begin
    num_hidden = arch.num_hidden
    num_latent = arch.num_latent
    num_data = arch.num_data

    hidden_b = zeros(num_hidden) ~ "hidden-biases"
    hidden_W = zeros(num_hidden, dim_data) ~ "hidden-weights"

    hidden_loadings = hidden_b + hidden_W' * data
    hidden_activations = sigmoid(hidden_loadings)

    latent_mu_b = zeros(dim_data) ~ "latent-mu-biases"
    latent_mu_W = zeros(dim_data, num_hidden) ~ "latent-mu-weights"
    
    latent_mu = latent_mu_b + latent_mu_W' * hidden_activations

    latent_log_std_b = zeros(dim_data) ~ "latent-logstd-biases"
    latent_log_std_W = zeros(dim_data, num_hidden) ~ "latent-logstd-weights"

    latent_log_std = latent_log_std_b + latent_log_std_W' * hidden_activations

    # the noise (this is the reparameterization trick)
    eps = map((i) -> normal(0,1), 1:num_latent)
    latents = map((i) -> normal(latent_mu[i], exp(latent_log_std[i])) ~ "latent-$i", 1:num_latent)
end

function generative_network_init_params()
    theta = Dict()
    theta["hidden-biases"] = randn(generative_num_hidden)
    theta["hidden-weights"] = randn(generative_num_hidden, num_latents)
    theta["data-biases"] = randn(dim_data)
    theta["data-weights"] = randn(dim_data, generative_num_hidden)
    theta
end

function recognition_network_init_params()
    phi = Dict()
    phi["hidden-biases"] = randn(recognition_num_hidden)
    phi["hidden-weights"] = randn(recognition_num_hidden, dim_data)
    phi["data-biases"] = randn(num_latents)
    phi["data-weights"] = randn(num_latents, recognition_num_hidden)
    phi
end

function generative_network_zero_params()
    theta = Dict()
    theta["hidden-biases"] = zeros(generative_num_hidden)
    theta["hidden-weights"] = zeros(generative_num_hidden, num_latents)
    theta["data-biases"] = zeros(dim_data)
    theta["data-weights"] = zeros(dim_data, generative_num_hidden)
    theta
end

function recognition_network_zero_params()
    phi = Dict()
    phi["hidden-biases"] = zeros(recognition_num_hidden)
    phi["hidden-weights"] = zeros(recognition_num_hidden, dim_data)
    phi["data-biases"] = zeros(num_latents)
    phi["data-weights"] = zeros(num_latents, recognition_num_hidden)
    phi
end


function parametrize!(trace::DifferentiableTrace, params::Dict)
    # TODO that should be a built-in Gen function
    for key in keys(params)
        parameterize!(trace, key, params[key])
    end
end

function update_gradient!(gradient::Dict, trace::DifferentiableTrace)
    for key in keys(gradient)
        gradient[key] += derivative(trace, key)
    end
end

function train(dataset::Matrix{Bool})

    # architecture
    (num_data, dim_data) = size(dataset)
    num_latents = 2
    generative_num_hidden = 50
    recognition_num_hidden = 50

    # initialize generative network parameters
    theta = generative_network_init_params()

    # initialize recognition network parameters
    phi = recognition_network_init_params()

    # stochastic gradient ascent to maximize the expected ELBO 
    num_iters = 1000
    minibatch_size = 100
    for iter=1:num_iters

        # TODO use ADAM optimizer
        theta_grad = generative_network_zero_params()
        phi_grad = recognition_network_zero_params()

        minibatch = StatsBase.sample(1:num_data, minibatch_size, replace=false)
        
        # use one latent sample per data point
        generative_scores = zeros(length(minibatch))
        recognition_scores = zeros(length(minibatch))
        for (i, datum_idx) in enumerate(minibatch)

            # score and gradient for recognition network
            recognition_trace = DifferentiableTrace()
            parametrize!(recognition_trace, phi)
            @generate(recognition_trace, recognition_network(dataset[datum_idx,:]))
            backprop(recognition_trace)
            update_gradient!(theta_grad, recognition_trace)
            generative_scores[i] = score(recognition_trace)

            # score and gradient for generative network
            generative_trace = DifferentiableTrace()
            parametrize!(generative_trace, theta)
            for j=1:arch.num_latent
                constrain!(generative_trace, "latent-$j",
                           value(recognition_trace, "latent-$j"))
            end
            @generate(generative_trace, generative_network())
            backprop(trace)
            update_gradient!(phi_grad, generative_trace)
            generative_scores[i] = score(generative_trace)


        end

        # use the naive estimator
        # use 
    end

    generative_trace = DifferentiableTrace()


    recognition_trace = DifferentiableTrace()

end

# NOTE: given IID data, this is equivalent to minimizing the KL from the
# data-generator / recognition network to the generative model, right?

