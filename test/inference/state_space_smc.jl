import Distributions

struct SMCScheme <: StateSpaceSMCScheme{Int}
    ess_threshold::Float64
    num_particles::Int
    prior::Vector{Float64}
    transition::Matrix{Float64}
    likelihoods::Vector{Vector{Float64}}
end

function Gen.init(scheme::SMCScheme)
    state = rand(Distributions.Categorical(scheme.prior))
    ll = log(scheme.likelihoods[1][state])
    (state, ll)
end

function Gen.forward(scheme::SMCScheme, prev_state::Int, t::Int)
    @assert t > 1
    state = rand(Distributions.Categorical(scheme.transition[:,prev_state]))
    ll = log(scheme.likelihoods[t][state])
    (state, ll)
end

Gen.get_num_steps(scheme::SMCScheme) = length(scheme.likelihoods)
Gen.get_num_particles(scheme::SMCScheme) = scheme.num_particles
Gen.get_ess_threshold(scheme::SMCScheme) = scheme.ess_threshold

function compute_true_log_marginal_likelihood(prior, transition, likelihoods)

    # use forward algorithm to compute marginal likelihood
    alpha = prior .* likelihoods[1]
    for likelihood in likelihoods[2:end]
        alpha = likelihood .* (transition * alpha)
    end
    marginal_likelihood = sum(alpha)
    return log(marginal_likelihood)
end

@testset "State space SMC for discrete HMM" begin

    srand(1)

    # A simple HMM with three hidden states
    prior = [0.2, 0.3, 0.5]

    # transition matrix: transition[i,j] = Prob(next-hidden = i | prev-hidden = j)
    transition = [0.1 0.4 0.5;
                  0.3 0.3 0.4;
                  0.6 0.3 0.1]

    # likelihoods[t][i] = Prob(obs-t | hidden_t = i)
    likelihoods = []
    push!(likelihoods, [0.1, 0.5, 4.5])
    push!(likelihoods, [0.5, 0.8, 1.2])
    push!(likelihoods, [0.2, 1.5, 1.2])
    push!(likelihoods, [2.5, 2.8, 2.2])

    true_log_ml = compute_true_log_marginal_likelihood(prior, transition, likelihoods)

    num_particles = 10000
    smc_scheme = SMCScheme(num_particles / 2, num_particles, prior, transition, likelihoods)
    result = smc(smc_scheme)
    @test isapprox(result.log_ml_estimate, true_log_ml, atol=1e-2)
end
