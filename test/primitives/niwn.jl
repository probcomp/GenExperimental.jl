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
