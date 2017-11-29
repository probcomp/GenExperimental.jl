const none = ()

@testset "program tagging syntaxes" begin

    # use the `bar = function ()` definition syntax (this was once breaking)
    bar = @probabilistic function (mu::Float64)
        @g(normal(mu, 1), "y")
        "something"
    end

    # use the `@probabilistic function foo()` definition syntax
    @probabilistic function foo()

        # untraced primitive generator invocatoin with syntax sugar
        x1 = normal(0, 1)

        # traced primitive generator invocation with syntax sugar
        x4 = @g(normal(0, 1), "a")

        # untraced generator invocation is identical to function invocation
        x5 = bar(0.5)
        @test x5 == "something"

        # non-primitive generator invocation
        @alias("b", "y", "c")
        x7 = @g(bar(0.5), "b")

        # expres
        nothing
    end

    (_, _, trace) = simulate(foo, (), none)
    @test haskey(trace, "a")
    @test haskey(trace, "c")
    @test !haskey(trace, "b")
    @test !haskey(trace, "y")
end

@testset "program definition syntaxes" begin

    # anonymous, no arguments
    foo1 = @probabilistic function () nothing end
    @test simulate(foo1, (), none)[1:2] == (0., nothing)

    # anonymous, single typed argument
    foo2 = @probabilistic function (x::Int) x end
    @test simulate(foo2, (1,), none)[1:2] == (0., 1)

    # anonymous, single untyped argument
    foo3 = @probabilistic function (x) x end
    @test simulate(foo3, (1,), none)[1:2] == (0., 1)

    # anonymous, multiple argument with one untyped
    foo4 = @probabilistic function (x::Int, y) x, y end
    @test simulate(foo4, (1, 2), none)[1:2] == (0., (1, 2))

    # anonymous, no arguments
    @probabilistic function bar1() nothing end
    @test simulate(bar1, (), none)[1:2] == (0., nothing)

    # anonymous, single typed argument
    @probabilistic function bar2(x::Int) x end
    @test simulate(bar2, (1,), none)[1:2]  == (0., 1)

    # anonymous, single untyped argument
    @probabilistic function bar3(x) x end
    @test simulate(bar3, (1,), none)[1:2]  == (0., 1)

    # anonymous, multiple argument with one untyped
    @probabilistic function bar4(x::Int, y) x, y end
    @test simulate(bar4, (1, 2), none)[1:2]  == (0., (1, 2))
end

@testset "lexical scope" begin
    x = 123

    # using one definition syntax
    foo = @probabilistic function () x end
    @test simulate(foo, (), none)[2] == x

    # using the other definition syntax
    @probabilistic function bar() x end
    @test simulate(bar, (), none)[2] == x
end

@testset "assess! and simulate" begin
    foo = @probabilistic function () @g(normal(0, 1), "x") end

    # outputs: none
    # assess! and simulate have the same behavior in this case
    # score is 0., and the value is overwritten by assess!
    t = FlatDictTrace()
    t["x"] = 2.3
    score, val = assess!(foo, (), none, t)
    @test score == 0.
    @test val != 2.3
    @test t["x"] == val
    score, val, t = simulate(foo, (), none)
    @test score == 0.
    @test val != 2.3
    @test t["x"] == val

    # outputs: "x"
    # assess!
    # score is is the log-density, and does not overwrite the value
    t["x"] = 2.3
    score, val = assess!(foo, (), ("x",), t)
    @test score == logpdf(Normal(), 2.3, 0, 1)
    @test val == 2.3
    @test t["x"] == val

    # outputs: "x"
    # simulate
    # score is is the log-density, and overwrites the value
    score, val, trace = simulate(foo, (), ("x",))
    @test val != 2.3
    @test trace["x"] == val
    @test score == logpdf(Normal(), val, 0, 1)
end

@testset "proposing from atomic trace" begin
    score, val, trace = simulate(normal, (0, 1), (ADDR_OUTPUT,))
    @test score == logpdf(normal, val, 0, 1)
    @test val == trace[ADDR_OUTPUT]
end

@testset "proposing from program trace" begin
    foo = @probabilistic function () @g(normal(0, 1), "x") end
    score, val, trace = simulate(foo, (), ("x",))
    @test score == logpdf(Normal(), val, 0, 1)
    @test val == trace["x"]
end

@testset "delete!" begin

    foo = @probabilistic function () @g(normal(0, 1), "x") end

    # deleting a value a program trace
    (_, _, trace) = simulate(foo, (), none)
    delete!(trace, "x")
    @test !haskey(trace, "x")

    # deleting the value from an atomic trace
    (_, _, trace) = simulate(normal, (0, 1), none)
    @test haskey(trace, ADDR_OUTPUT)
    delete!(trace, ADDR_OUTPUT)
    @test !haskey(trace, ADDR_OUTPUT)
end

@testset "higher order probabilistic program" begin
    foo = @probabilistic function () 
        mu = @g(normal(0, 10), "mu")
        std = @g(Gen.gamma(1., 1.), "std")

        # return a probabilistic program
        (@probabilistic function ()
            @g(normal(mu, std), "x")
        end)
    end
    trace = FlatDictTrace()
    trace["foo-mu"] = 4.
    trace["foo-std"] = 1.
    trace["a-x"] = 4.5
    trace["b-x"] = 4.3
    trace["c-x"] = 4.2
    outputs = ("foo-mu", "foo-std", "a-x", "b-x" ,"c-x")
    (score, value) = assess!((@probabilistic function () 
        @alias("foo", "mu", "foo-mu")
        @alias("foo", "std", "foo-std")
        @alias("a", "x", "a-x")
        @alias("b", "x", "b-x")
        @alias("c", "x", "c-x")
        sampler = @g(foo(), "foo")
        x1 = @g(sampler(), "a")
        x2 = @g(sampler(), "b")
        x3 = @g(sampler(), "c")
        (x1, x2, x3)
    end), (), outputs, trace)
    @test value == (4.5, 4.3, 4.2)
    expected_score = 0.
    expected_score += logpdf(Normal(), 4., 0., 10.)
    expected_score += logpdf(Gamma(), 1., 1., 1.)
    expected_score += logpdf(Normal(), 4.5, 4.0, 1)
    expected_score += logpdf(Normal(), 4.3, 4.0, 1)
    expected_score += logpdf(Normal(), 4.2, 4.0, 1)
    @test isapprox(expected_score, score)
end

@testset "scores" begin

    model = @probabilistic function () 
        cloudy = @g(flip(0.3), "cloudy")
        sprinkler = @g(flip(cloudy ? 0.1 : 0.4), "sprinkler")
        rain = @g(flip(cloudy ? 0.8 : 0.2), "rain")
        wetgrass = @g(flip(
            if sprinkler
                rain ? 0.99 : 0.9
            else
                rain ? 0.9 : 0.01
            end), "wetgrass")
    end

    # the score for assess! is the sum of output scores
    t = FlatDictTrace()
    t["cloudy"] = true
    t["sprinkler"] = true
    t["rain"] = true
    t["wetgrass"] = true
    (score, _) = assess!(model, (), ("cloudy", "sprinkler", "rain", "wetgrass"), t)
    expected_score = log(0.3) + log(0.1) + log(0.8) + log(0.99)
    @test isapprox(score, expected_score)

    # an address that is not in the output set is not scored
    t = FlatDictTrace()
    t["sprinkler"] = true
    t["rain"] = true
    t["wetgrass"] = true
    (score, _) = assess!(model, (), ("sprinkler", "rain", "wetgrass"), t)
    sprinkler_score = t["cloudy"] ? log(0.1) : log(0.4)
    rain_score = t["cloudy"] ? log(0.8) : log(0.2)
    wetgrass_score = log(0.99)
    expected_score = sprinkler_score + rain_score + wetgrass_score
    @test isapprox(score, expected_score)

    # the score for simulate is the sum of the output scores
    (score, _, t) = simulate(model, (), ("sprinkler",))
    expected_score = if t["cloudy"]
        t["sprinkler"] ? log(0.1) : log(0.9)
    else
        t["sprinkler"] ? log(0.4) : log(0.6)
    end
    @test isapprox(score, expected_score)

    # scoring works with sub-traces
    toplevel = @probabilistic function () 
        @alias("sub", "cloudy", "sub-cloudy")
        @alias("sub", "sprinkler", "sub-sprinkler")
        @alias("sub", "rain", "sub-rain")
        @alias("sub", "wetgrass", "sub-wetgrass")
        @g(model(), "sub")
    end
    trace = FlatDictTrace()
    outputs = ("sub-cloudy", "sub-sprinkler", "sub-rain", "sub-wetgrass")
    for addr in ["sub-cloudy", "sub-sprinkler", "sub-rain", "sub-wetgrass"]
        trace[addr] = true
    end
    (score, _) = assess!(toplevel, (), outputs, trace)
    expected_score = log(0.3) + log(0.1) + log(0.8) + log(0.99)
    @test isapprox(score, expected_score)
end


@testset "autodiff on trace score" begin
    foo = @probabilistic function (mu, log_std) 
        @g(normal(mu, exp(log_std)), "x")
    end

    trace = FlatDictTrace()
    x = 2.1
    trace["x"] = x
	tape1 = Tape()
	mu1 = GenScalar(0., tape1)
    logstd1 = GenScalar(0., tape1)
    (score, _) = assess!(foo, (mu1, logstd1), ("x",), trace)
	backprop(score)

    tape2 = Tape()
	mu2 = GenScalar(0., tape2)
	logstd2 = GenScalar(0., tape2)
    backprop(logpdf(normal, x, mu2, exp(logstd2)))
    @test isapprox(partial(mu1), partial(mu2))
    @test isapprox(partial(logstd1), partial(logstd2))
end

@testset "aliases" begin

    @probabilistic function inner() 
        @g(flip(0.1), "b")
    end

    @probabilistic function f() 
        @alias("a", "b", "exposed")
        @g(inner(), "a")
    end

    # simulate without any outputs
    (_, value, trace) = simulate(f, (), ())
    @test !haskey(trace, "a")
    @test !haskey(trace, "b")
    @test trace["exposed"] == value

    # simulate with alias as an output
    (score, value, trace) = simulate(f, (), ("exposed",))
    @test !haskey(trace, "a")
    @test !haskey(trace, "b")
    @test trace["exposed"] == value
    @test isapprox(score, value ? log(0.1) : log(0.9))

    # regenerate with alias as an output
    trace = FlatDictTrace()
    trace["exposed"] = true
    (score, value) = assess!(f, (), ("exposed",), trace)
    @test trace["exposed"] == value
    @test value == true
    @test isapprox(score, log(0.1))
end
