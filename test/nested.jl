@testset "nested inference generator" begin

	none = AddressTrie()


    ### a single random variable ###

    g = nested(normal, normal, Dict([() => ()]))
    t = AtomicTrace(Float64)

    # score = log(p(x) / q(x))
    (score, x) = simulate!(g, ((1., 2.), (3., 4.)), none, none, t)
	@test isapprox(score, logpdf(normal, x, 1., 2.) - logpdf(normal, x, 3., 4.))

    # score = log(p(x))
    # (the address () will become a condition for q, so q has no effect)
    (score, x) = simulate!(g, ((1., 2.), (3., 4.)), AddressTrie(()), none, t)
    @test isapprox(score, logpdf(normal, x, 1., 2.))
    @test t[()] == x

    # score = log(p(x) / q(x))
    (score, x) = regenerate!(g, ((1., 2.), (3., 4.)), none, none, t)
	@test isapprox(score, logpdf(normal, x, 1., 2.) - logpdf(normal, x, 3., 4.))

    # score = log(p(x))
    # (the address () will become a condition for q, so q has no effect)
	t[()] = 1.2
    (score, x) = regenerate!(g, ((1., 2.), (3., 4.)), AddressTrie(()), none, t)
    @test t[()] == 1.2
    @test t[()] == x
    @test isapprox(score, logpdf(normal, x, 1., 2.))


    ### a simple model with an importance distribution ###

    p = @program () begin
        mu = @g(normal(0., 1.), "mu")
        @g(normal(mu, 2.), "x")
    end
    g = nested(p, normal, Dict(["mu" => ()]))
    t = DictTrace()
    x = 1.123
	t["x"] = x
    (score, g_retval) = regenerate!(g, ((), (3., 4.)), AddressTrie("x"), none, t)

    # return value is the return value of p and the value in the trace didn't change
    @test g_retval == x
	@test t["x"] == x

    # score = log(p(mu, x) / q(mu)) (for mu ~ q(.))
    expected_score = (
        logpdf(normal, t["mu"], 0., 1.) +
        logpdf(normal, x, t["mu"], 2.) -
        logpdf(normal, t["mu"], 3., 4.))
    @test isapprox(score, expected_score)

    (score, g_retval) = simulate!(g, ((), (3., 4.)), AddressTrie("x"), none, t)

	# the return value is newly sampled
	@test t["x"] != x
	@test t["x"] == g_retval

    # score = log(p(mu, x) / q(mu)) for (mu ~ p(.))
    expected_score = (
        logpdf(normal, t["mu"], 0., 1.) +
        logpdf(normal, t["x"], t["mu"], 2.) -
        logpdf(normal, t["mu"], 3., 4.))
    @test isapprox(score, expected_score)
end
