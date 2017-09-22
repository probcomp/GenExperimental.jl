struct ADAMParams
    alpha::Float64
    beta_1::Float64
    beta_2::Float64
    epsilon::Float64
end

"""
Generic implementation of Adam optimizer, from:

Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." 
arXiv preprint arXiv:1412.6980 (2014).

The user implements a custom objective type `T` which must implement the following method:

    fgrad_estimate(::T, params::Vector{Float64})

The method must return a tuple `(f, g)` where `f` is an unbiased estimate of the objective function for the parameters in `params` and where `g` is an unbiased estimate of the gradient of the objective with respect to these parameters.
"""
struct Optimizer{T}
    objective::T
    adam_params::ADAMParams
    minibatch_size::Int
    verbose::Bool
end

function optimize(opt::Optimizer, params::Vector{Float64}, num_steps::ADAMParams)

    # ADAM state
    num_params = length(params)
    adam_m = zeros(num_params)
    adam_v = zeros(num_params)
    history = Vector{Float64}(num_steps)
    for t=1:num_steps

        # TODO parallelize across minibatch
        results = map((i) -> fgrad_estimate(opt.objective, params), 1:opt.minibatch_size)
        
        # unbiased estimate of gradient
        grads = map((r) -> r[2], results)
        grad = (1. / opt.minibatch_size) * sum(grads) 
        params, adam_m, adam_v = adam_update(opt.adam_params, t, params, grad, adam_m, adam_v)

        # print unbiased estimate of objective 
        scores = map((r) -> r[1], results)
        objective_est = mean(scores)
        history[t] = objective_est
        if opt.verbose
            println("iter: $t, objective est: $objective_est")
        end
    end
    return params, history
end

function adam_update(adam_params::ADAMParams, t::Integer,
                     params::Vector{Float64}, grad::Vector{Float64},
                     m::Vector{Float64}, v::Vector{Float64})
    m = adam_params.beta_1 * m + (1. - adam_params.beta_1) * grad
    v = adam_params.beta_2 * v + (1. - adam_params.beta_2) * (grad .* grad)
    mhat = m / (1. - adam_params.beta_1^t)
    vhat = v / (1. - adam_params.beta_2^t)
    params += adam_params.alpha * mhat ./ (sqrt.(vhat + adam_params.epsilon))
    return (params, m, v)
end
