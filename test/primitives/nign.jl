@testset "NIGN" begin

    @testset "NIGN sufficient statistics" begin
        # arbitrary values for parameters
        m = 1.23
        r = 2.41
        nu = 5.44
        s = 4.2
        alpha = nu/2.
        beta = s/2.
        params = NIGNParams(m, r, nu, s)
    
        # test without data
        nign = NIGNState()
        actual = log_joint_density(nign, params)
        @test abs(actual) < 1e-15
    
        # test posterior_params
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        x1 = -4.1
        x2 = 2.22
        nign = NIGNState()
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
    
        # test incorporate and unincorporate being inverses
        nign = NIGNState()
        incorporate!(nign, 0.1)
        incorporate!(nign, 0.2)
        unincorporate!(nign, 0.1)
        unincorporate!(nign, 0.2)
        actual = log_joint_density(nign, params)
        @test abs(actual) < 1e-15
    end

    @testset "NIGN generator" begin
        params = NIGNParams(1., 2., 3., 4.)
        trace = NIGNTrace(Int)

        # simulate without any outputs or conditions 
        n = 10
        (score, values) = simulate!(NIGNGenerator(Int), (1:n, params), AddressTrie(), AddressTrie(), trace)
        for i=1:n
            @test haskey(trace, i)
            @test trace[i] == values[i]
        end
        @test !haskey(trace, n+1)
        @test score == 0.

        # regenerate with outputs and conditions
        # p(1.0, 3.4 | 2.2; params)
        trace = NIGNTrace(Int)
        t1 = 1.0
        trace[1] = t1
        t2 = 2.2
        trace[2] = t2
        t3 = 3.4
        trace[3] = t3
        outputs = AddressTrie(1, 3)
        conditions = AddressTrie(2)
        (score, values) = regenerate!(NIGNGenerator(Int), ([1, 2, 3], params), outputs, conditions, trace)
        @test trace[1] == t1
        @test trace[2] == t2
        @test trace[3] == t3
        @test values[1] == t1
        @test values[2] == t2
        @test values[3] == t3
    
        # TODO check score against a reference implementation
        @test score != 0.

        # simulate with outputs and conditions
        # p(., .  | 2.2; params)
        trace = NIGNTrace(Int)
        t2 = 2.2
        trace[2] = t2
        outputs = AddressTrie(1, 3)
        conditions = AddressTrie(2)
        (score, values) = simulate!(NIGNGenerator(Int), ([1, 2, 3], params), outputs, conditions, trace)
        @test trace[2] == t2
        @test values[1] == trace[1]
        @test values[2] == t2
        @test values[3] == trace[3]

        # TODO check score against a reference implementation
        @test score != 0.

    end
end
