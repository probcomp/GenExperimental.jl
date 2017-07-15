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
    actual = logpdf(nign, params)
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
    actual = logpdf(nign, params)
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
        # logpdf. these take different code paths.

        # compute the expected value using logpdf
        log_density_before = logpdf(nign, params)
        incorporate!(nign, x3)
        expected = logpdf(nign, params) - log_density_before
        unincorporate!(nign, x3)

        actual = logpdf(draw_nign, x3, nign, params)
        @test isapprox(actual, expected)
    
    end

    @testset "NIGN generator" begin
        params = NIGNParams(1., 2., 3., 4.)
        trace = NIGNJointTrace()

        # test that the correct values were generated and that the 
        # the constrained values were not modified
        n = 10
        constrain!(trace, 5, 5.3)
        a5 = value(trace, 5)
        constrain!(trace, 9, 6.2)
        a9 = value(trace, 9)
        score = generate!(NIGNJointGenerator(), (n, params), trace)
        for i=1:n
            @test hasvalue(trace, i)
        end
        @test !hasvalue(trace, n+1)
        @test value(trace, 5) == a5
        @test value(trace, 9) == a9

        # test that the score is for the contsrained values
        state = NIGNState()
        incorporate!(state, a5)
        incorporate!(state, a9)
        expected_score = logpdf(state, params)
        @test isapprox(score, expected_score)

        # generate again and check that the score hasn't changed
        # (this checks that the sufficient statistics were correctly reverted)
        score = generate!(NIGNJointGenerator(), (n, params), trace)
        @test isapprox(score, expected_score)

    end


end

