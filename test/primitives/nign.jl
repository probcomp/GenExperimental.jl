@testset "NIGN" begin

    # arbitrary values
    m = 1.23
    r = 2.41
    nu = 5.44
    s = 4.2
    alpha = nu/2.
    beta = s/2.
    params = NIGNParams(m, r, nu, s)

    nign = NIGNState()
    actual = joint_log_density(nign, params)
    @test abs(actual) < 1e-15

    # test posterior
    # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    x1 = -4.1
    x2 = 2.22
    incorporate!(nign, x1)
    incorporate!(nign, x2)
    actual = posterior_params(nign, params)
    # m, r, s, nu
    avg = mean([x1, x2])
    sum_sq = sum(([x1, x2] - avg) .* ([x1,x2] - avg))
    n = 2
    m_expect = (r * m + n * avg) / (r + n)
    r_expect = r + n
    alpha_expect = alpha + n/2.
    beta_expect = beta + 0.5 * sum_sq  + r*n*((avg-m)*(avg-m))/(2*(r+n))
    nu_expect = alpha_expect*2.
    s_expect = beta_expect*2.
    @test isapprox(actual.m, m_expect)
    @test isapprox(actual.r, r_expect)
    @test isapprox(actual.s, s_expect)
    @test isapprox(actual.nu, nu_expect)

    # should go back to empty again
    unincorporate!(nign, x1)
    unincorporate!(nign, x2)
    actual = joint_log_density(nign, params)
    @test abs(actual) < 1e-15

    @testset "draw from NIGN" begin
        params = NIGNParams(1., 2., 3., 4.)

        # the initial NIGN state
        state = NIGNState()
        x1 = 2.4
        x2 = 5.4
        incorporate!(nign, x1)
        incorporate!(nign, x2)

        # the value being drawn from the NIGN state
        x3 = -0.4

        # check that the density of regenerate matches the value computed using
        # joint_log_density. these take different code paths.

        # compute the expected value using joint_log_density
        log_density_before = joint_log_density(nign, params)
        incorporate!(nign, x3)
        expected = joint_log_density(nign, params) - log_density_before
        unincorporate!(nign, x3)

        actual = logpdf(draw_nign, x3, nign, params)
        @test isapprox(actual, expected)
    
    end

end

