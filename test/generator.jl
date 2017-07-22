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

@testset "encapsulate a probabilistic program" begin

    inner_generator = @program () begin
        mu = @g(normal(0, 1), "mu")
        @g(normal(mu, 1), "x")
    end

    generator = encapsulate(inner_generator, Dict([("x",) => Float64]))

    # record
    t = AtomicTrace(Dict)
    (score, val) = generate!(generator, (), t)
    @test val[("x",)] == value(t)[("x",)]

    # constrain
    # NOTE: it's not obvious how to test the score, since we don't have access to mu
    t = AtomicTrace(Dict)
    constrain!(t, (), Dict([("x",) => 0.123]))
    (score, val) = generate!(generator, (), t)
    @test val[("x",)] == 0.123
    @test value(t)[("x",)] == 0.123
    @test score != 0.

    # propose 
    # NOTE: it's not obvious how to test the score, since we don't have access to mu
    t = AtomicTrace(Dict)
    propose!(t, (), Dict)
    (score, val) = generate!(generator, (), t)
    @test score != 0.
    @test haskey(value(t, ()), ("x",))
    @test haskey(val, ("x",))
end


@testset "replicated atomic generator" begin
    run_count = 0
    inner_generator = encapsulate(
        (@program () begin
            run_count += 1
            mu = @g(normal(0, 1), "mu")
            @g(normal(mu, 1), "x")
        end),
        Dict([("x",) => Float64]))
    generator = replicate(inner_generator, 4)

    # record
    t = AtomicTrace(Dict)
    (score, val) = generate!(generator, (), t)
    @test val[("x",)] == value(t)[("x",)]
    # NOTE: if we don't require the score, there is no need to run more than once
    @test run_count == 1

    # constrain
    # NOTE: it's not obvious how to test the score, since we don't have access to mu
    run_count = 0
    t = AtomicTrace(Dict)
    constrain!(t, (), Dict([("x",) => 0.123]))
    (score, val) = generate!(generator, (), t)
    @test val[("x",)] == 0.123
    @test value(t)[("x",)] == 0.123
    @test score != 0.
    @test run_count == 4

    # propose 
    # NOTE: it's not obvious how to test the score, since we don't have access to mu
    run_count = 0
    t = AtomicTrace(Dict)
    propose!(t, (), Dict)
    (score, val) = generate!(generator, (), t)
    @test score != 0.
    @test haskey(value(t, ()), ("x",))
    @test haskey(val, ("x",))
    @test run_count == 4

end

