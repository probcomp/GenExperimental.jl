@testset "normal inverse Wishart" begin

function test_multivariate_lgamma()
    # single dimension
    @test isapprox(multivariate_lgamma(1, 1.123), lgamma(1.123))
    
    # multiple dimension 
    expected = log(pi^(0.5)*Base.gamma(1.123)*Base.gamma(1.123 - 0.5))
    @test isapprox(multivariate_lgamma(2, 1.123), expected)
end
test_multivariate_lgamma()

function test_no_data()
    prior = NIWParams([0.1, 0.2], 0.1, 4, eye(2))

    # posterior is prior
    posterior_params = posterior(prior, NIWNState(2))
    @test isapprox(posterior_params.mu, prior.mu)
    @test isapprox(posterior_params.k, prior.k)
    @test isapprox(posterior_params.m, prior.m)
    @test isapprox(posterior_params.Psi, prior.Psi)

    # marginal likelihood is 1.
    @test isapprox(log_marginal_likelihood(NIWNState(2), prior), 0.)

end
test_no_data()

function test_univariate()
    niw_params = NIWParams([1.1], 2.2, 3.3, ones(1, 1) * 2.1)
    nign_params = NIGNParams(1.1, 2.2, 3.3, 2.1)
    niwn_state = NIWNState(1)
    nign_state = NIGNState()
    x = [3.1, -5.4, 2.3, 5.2]
    for xi in x
        incorporate!(niwn_state, [xi])
        incorporate!(nign_state, xi)
    end

    # log marginal likelihood
    @test isapprox(log_marginal_likelihood(niwn_state, niw_params),
                   log_joint_density(nign_state, nign_params))

    # posterior parameters
    niw_posterior = posterior(niw_params, niwn_state)
    nign_posterior = posterior_params(nign_state, nign_params)
    @test isapprox(niw_posterior.mu[1], nign_posterior.m)
    @test isapprox(niw_posterior.k, nign_posterior.r)
    @test isapprox(niw_posterior.m, nign_posterior.nu)
    @test isapprox(niw_posterior.Psi[1,1], nign_posterior.s)

    # predictive probability
    x_new = [4.1]
    @test isapprox(predictive_logp(x_new, niwn_state, niw_params),
                   predictive_logp(x_new[1], nign_state, nign_params))
end
test_univariate()

function test_log_marginal_likelihood_faster()
    prior = NIWParams([0.1, 0.2], 0.1, 4, eye(2))
    state = NIWNState(2)
    incorporate!(state, [1., 2.])
    incorporate!(state, [3.1, 4.2])
    @test isapprox(log_marginal_likelihood(state, prior),
                   log_marginal_likelihood_faster(state, prior))
end
test_log_marginal_likelihood_faster()

function test_incorporate_unincorporate()
    state = NIWNState(2)
    incorporate!(state, [1.1, 2.2])
    incorporate!(state, [1.2, 7.1])
    unincorporate!(state, [1.1, 2.2])
    unincorporate!(state, [1.2, 7.1])
    @test state.n == 0
    @test norm(state.x_total) < 1e-13
    @test norm(state.S_total) < 1e-13
end
test_incorporate_unincorporate()


end


@testset "normal inverse Wishart generator" begin

function test_regenerate_no_outputs_no_conditions()
    # regenerate with no outputs and no conditions does not populate the trace,
    # and returns a score of 0.
    params = NIWParams([0.1, 0.2], 0.1, 4, eye(2))
    trace = NIWNTrace(Int, 2)
    n = 10
    args = (1:n, params, true)
    outputs = AddressTrie()
    conditions = AddressTrie()
    (score, values) = regenerate!(NIWNGenerator(Int, 2), args, outputs, conditions, trace)
    for i=1:n
        @test !haskey(trace, i)
        @test !haskey(values, i)
    end
    @test !haskey(trace, n+1)
    @test score == 0.
end
test_regenerate_no_outputs_no_conditions()

function test_simulate_no_outputs_no_conditions()
    # simulate with no outputs and no conditions populates the trace with all
    # data and returns a score of 0.
    params = NIWParams([0.1, 0.2], 0.1, 4, eye(2))
    trace = NIWNTrace(Int, 2)
    n = 10
    args = (1:n, params, true)
    outputs = AddressTrie()
    conditions = AddressTrie()
    (score, values) = simulate!(NIWNGenerator(Int, 2), args, outputs, conditions, trace)
    for i=1:n
        @test haskey(trace, i)
        @test trace[i] == values[i]
    end
    @test !haskey(trace, n+1)
    @test score == 0.
end
test_simulate_no_outputs_no_conditions()

function test_regenerate_outputs_conditions()
    # regenerate returns the conditional probability of the outputs given the
    # conditions, and does not add any data to the trace.
    params = NIWParams([0.1, 0.2], 0.1, 4, eye(2))
    trace = NIWNTrace(Int, 2)
    t1 = [1.0, 1.1]
    trace[1] = t1
    t2 = [2.2, 1.1]
    trace[2] = t2
    t3 = [1.1, 3.4]
    trace[3] = t3
    outputs = AddressTrie(1, 3)
    conditions = AddressTrie(2)
    (score, values) = regenerate!(NIWNGenerator(Int, 2), ([1, 2, 3], params, true), outputs, conditions, trace)
    @test trace[1] == t1
    @test trace[2] == t2
    @test trace[3] == t3
    @test values[1] == t1
    @test values[2] == t2
    @test values[3] == t3

    # check score is P(t1 | t2) * P(t3 | t1, t2)
    state = NIWNState(2)
    incorporate!(state, trace[2])
    expected_score = 0.
    for t in [trace[1], trace[3]]
        expected_score += predictive_logp(t, state, params)
        incorporate!(state, t)
    end
    @test isapprox(score, expected_score)
end
test_regenerate_outputs_conditions()

function test_simulate_outputs_conditions()
    # simulate with outputs and conditions
    params = NIWParams([0.1, 0.2], 0.1, 4, eye(2))
    trace = NIWNTrace(Int, 2)
    t2 = [1.1, 2.2]
    trace[2] = t2
    outputs = AddressTrie(1, 3)
    conditions = AddressTrie(2)
    (score, values) = simulate!(NIWNGenerator(Int, 2), ([1, 2, 3], params, true), outputs, conditions, trace)
    @test trace[2] == t2
    @test values[1] == trace[1]
    @test values[2] == t2
    @test values[3] == trace[3]

    # check score is P(t1 | t2) * P(t3 | t1, t2)
    state = NIWNState(2)
    incorporate!(state, trace[2])
    expected_score = 0.
    for t in [trace[1], trace[3]]
        expected_score += predictive_logp(t, state, params)
        incorporate!(state, t)
    end
    @test isapprox(score, expected_score)
end
test_simulate_outputs_conditions()

end
