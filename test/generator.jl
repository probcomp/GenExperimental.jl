@testset "paired generator combinator" begin

    # a single random variable
    g = compose(normal, normal, Dict([() => ((), Float64)]))
    t = AtomicTrace(Float64)
    (score, x) = generate!(g, ((1., 2.), (3., 4.)), t)
    @test isapprox(score, logpdf(normal, x, 1., 2.) - logpdf(normal, x, 3., 4.))
    @test value(t) == x
    @test mode(t) == Gen.record

    # shorthand for the above, when q is an AtomicGenerator
    # the third argument is the address in p's trace
    g = compose(normal, normal, ()) 
    t = AtomicTrace(Float64)
    (score, x) = generate!(g, ((1., 2.), (3., 4.)), t)
    @test isapprox(score, logpdf(normal, x, 1., 2.) - logpdf(normal, x, 3., 4.))
    @test value(t) == x
    @test mode(t) == Gen.record

    # a simple model with an importance distribution
    p = @program () begin
        mu = @g(normal(0., 1.), "mu")
        @g(normal(mu, 2.), "x")
    end
    g = compose(p, normal, ("mu",))
    t = ProgramTrace()
    x = 1.123
    constrain!(t, "x", x)
    (score, g_retval) = generate!(g, ((), (3., 4.)), t)
    # return value is the return value of p
    @test g_retval == x
    # should not be constrained afterwards
    @test mode(t, "mu") == Gen.record
    # score = log(p(mu, x) / q(x)) for mu ~ p(mu)
    expected_score = (
        logpdf(normal, t["mu"], 0., 1.) +
        logpdf(normal, x, t["mu"], 2.) -
        logpdf(normal, t["mu"], 3., 4.))
    @test isapprox(score, expected_score)

end
