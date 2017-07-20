
@testset "program tagging syntaxes" begin

    # use the `bar = @program` definition syntax (this was once breaking)
    @program bar(mu::Float64) begin
        @g(normal(mu, 1), "y")
        "something"
    end

    # use the `@program foo` definition syntax
    @program foo() begin

        # untraced primitive generator invocatoin with syntax sugar
        x1 = normal(0, 1)

        # traced primitive generator invocation
        x2 = tag_generator(Normal(), (0, 1), 2)

        # traced primitive generator invocation with syntax sugar
        x4 = @g(normal(0, 1), 3)

        # untraced generator invocation is identical to function invocation
        x5 = bar(0.5)
        @test x5 == "something"

        # traced non-primitive generator invocation
        x6 = tag_generator(bar, (0.5,), 6)

        # traced non-primitive generator invocation with syntax sugar
        x7 = @g(bar(0.5), 7)

        # tracing a generator invocation as an expression evaluation
        x8 = @e(bar(0.5), 8)

        # tracing an expression as an expression evaluation
        x9 = @e("asdf_$x8", 9)

        # expres
        nothing
    end

    t = ProgramTrace()
    generate!(foo, (), t)
    @test haskey(t, 2)
    @test haskey(t, 3)
    @test haskey(t, 6)
    @test haskey(t, 7)
    @test haskey(t, 8)
    @test haskey(t, 9)
end

@testset "program definition syntaxes" begin

    # anonymous, no arguments
    foo1 = @program () begin nothing end
    @test generate!(foo1, (), ProgramTrace()) == (0., nothing)

    # anonymous, single typed argument
    foo2 = @program (x::Int) begin x end
    @test generate!(foo2, (1,), ProgramTrace()) == (0., 1)

    # anonymous, single untyped argument
    foo3 = @program (x) begin x end
    @test generate!(foo3, (1,), ProgramTrace()) == (0., 1)

    # anonymous, multiple argument with one untyped
    foo4 = @program (x::Int, y) begin x, y end
    @test generate!(foo4, (1, 2), ProgramTrace()) == (0., (1, 2))

    # anonymous, no arguments
    @program bar1() begin nothing end
    @test generate!(bar1, (), ProgramTrace()) == (0., nothing)

    # anonymous, single typed argument
    @program bar2(x::Int) begin x end
    @test generate!(bar2, (1,), ProgramTrace()) == (0., 1)

    # anonymous, single untyped argument
    @program bar3(x) begin x end
    @test generate!(bar3, (1,), ProgramTrace()) == (0., 1)

    # anonymous, multiple argument with one untyped
    @program bar4(x::Int, y) begin x, y end
    @test generate!(bar4, (1, 2), ProgramTrace()) == (0., (1, 2))
end

@testset "lexical scope" begin
    x = 123

    # using one definition syntax
    foo = @program () begin x end
    @test @generate!(foo(), ProgramTrace())[2] == x

    # using the other definition syntax
    @program bar() begin x end
    @test @generate!(bar(), ProgramTrace())[2] == x
end

@testset "program generate! syntaxes" begin

    foo = @program (x) begin x end
    
    # lower-level synyntax
    @test generate!(foo, ("asdf",), ProgramTrace()) == (0., "asdf")

    # the syntax sugar
    @test @generate!(foo("asdf"), ProgramTrace()) == (0., "asdf")
end

@testset "constraining trace" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()
    constrain!(t, "x", 2.3)
    score, val = @generate!(foo(), t)
    @test score == logpdf(Normal(), 2.3, 0, 1)
    @test val == 2.3
end

@testset "intervening on primitive generator invocation" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()
    intervene!(t, "x", 2.3)
    score, val = @generate!(foo(), t)
    @test score == 0.
    @test val == 2.3
end

@testset "intervening on probabilistic program invocation" begin
    bar = @program () begin @g(normal(0, 1), "y") end
    foo = @program () begin @g(bar(), "x") end
    t = ProgramTrace()
    # by default, interevne! will place an AtomicTrace at address "x", which will cause
    # an error during generate! because the generator uses ProgramTraces.
    set_subtrace!(t, "x", ProgramTrace())
    intervene!(t, "x", "fixed")
    score, val = @generate!(foo(), t)
    @test score == 0.
    @test val == "fixed"
end

@testset "tagging arbitrary expressions" begin
    foo = @program () begin
        
        # a generator invocation labelled as an expression
        x = @e(normal(0, 1), "x")

        # an expression
        y = @e(x + 1, "y")

        # a generator invocation labelled as an expression
        # using low-level syntax
        z = tag_expression(normal(0, 1), "z")

        # an expression
        # using low-level syntax
        w = tag_expression(x + 1, "w")

        (x, y, z, w)
    end
    t = ProgramTrace()
    
    # test that values are recorded
    score, val = @generate!(foo(), t)
    @test haskey(t, "x")
    @test haskey(t, "y")
    @test haskey(t, "z")
    @test haskey(t, "w")
    @test score == 0.
    @test val == (t["x"], t["y"], t["z"], t["w"])
    @test t["w"] == t["x"] + 1

    # test intervention
    intervene!(t, "x", 2.)
    intervene!(t, "y", 3.)
    intervene!(t, "z", 2.)
    intervene!(t, "w", 3.)
    score, val = @generate!(foo(), t)
    @test t["x"] == 2.
    @test t["y"] == 3.
    @test t["z"] == 2.
    @test t["w"] == 3.
    @test score == 0.
    @test val == (2., 3., 2., 3.)
end

@testset "proposing from trace" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()
    propose!(t, "x", Float64)
    score, val = @generate!(foo(), t)
    @test score == logpdf(Normal(), val, 0, 1)
end

@testset "different syntaxes for retrieving value from a trace" begin
    bar = @program () begin @g(flip(0.5), "x") end
    foo = @program () begin
        x = @g(bar(), "bar")
        y = @g(normal(0, 1), "y")
    end

    t = ProgramTrace()
    score, val = @generate!(foo(), t)

    # test a top-level address
    @test t[("y",)] == value(t, "y")
    @test t["y"] == value(t, "y")
    @test value(t, "y") == value(t, ("y",))

    # test a hierarchical address
    @test t[("bar", "x")] == value(t, ("bar", "x"))
    @test t["bar", "x"] == value(t, ("bar", "x"))
end

@testset "delete!" begin

    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()

    # deleting a constraint
    # the subtrace remains, but it is no longer constrained
    # what happens to the value is undefined (its up to the trace type)
    constrain!(t, "x", 1.1)
    score, _ = @generate!(foo(), t)
    @test value(t, "x") == 1.1
    @test score != 0.
    delete!(t, "x")
    @test haskey(t, "x")
    score, _ = @generate!(foo(), t)
    @test score == 0.
    @test value(t, "x") != 1.1
    
    # deleting an intervention
    # the subtrace remains, but it is no longer constrained
    # what happens to the value is undefined (its up to the trace type)
    intervene!(t, "x", 1.1)
    @generate!(foo(), t)
    @test value(t, "x") == 1.1
    delete!(t, "x")
    @test haskey(t, "x")
    @generate!(foo(), t)
    @test value(t, "x") != 1.1

    # deleting a proposal
    # the subtrace remains, but it is no longer constrained
    # what happens to the value is undefined (its up to the trace type
    propose!(t, "x", Float64)
    score, _ = @generate!(foo(), t)
    @test score != 0.
    delete!(t, "x")
    @test haskey(t, "x")
    score, _ = @generate!(foo(), t)
    @test score == 0.
end

@testset "higher order probabilistic program" begin
    foo = @program () begin
        mu = @g(normal(0, 10), "mu")
        std = @g(Gen.gamma(1., 1.), "std")

        # return a probabilistic program
        @program((), begin
            @g(normal(mu, std), "x")
        end)
    end
    t = ProgramTrace()
    for addr in ["foo", 1, 2, 3]
        set_subtrace!(t, addr, ProgramTrace())
    end
    constrain!(t, ("foo", "mu"), 4.)
    constrain!(t, ("foo", "std"), 1.)
    constrain!(t, (1, "x"), 4.5)
    constrain!(t, (2, "x"), 4.3)
    constrain!(t, (3, "x"), 4.2)

    (score, value) = generate!((@program () begin
        sampler = @g(foo(), "foo")
        x1 = @g(sampler(), 1)
        x2 = @g(sampler(), 2)
        x3 = @g(sampler(), 3)
        (x1, x2, x3)
    end), (), t)
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

    model = @program () begin
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

    # the score is the sum of constrained scores
    t = ProgramTrace()
    constrain!(t, "cloudy", true)
    constrain!(t, "sprinkler", true)
    constrain!(t, "rain", true)
    constrain!(t, "wetgrass", true)
    score, _ = @generate!(model(), t)
    expected_score = log(0.3) + log(0.1) + log(0.8) + log(0.99)
    @test isapprox(score, expected_score)

    # an unconstrained choice is not scored
    t = ProgramTrace()
    constrain!(t, "sprinkler", true)
    constrain!(t, "rain", true)
    constrain!(t, "wetgrass", true)
    score, _ = @generate!(model(), t)
    sprinkler_score = t["cloudy"] ? log(0.1) : log(0.4)
    rain_score = t["cloudy"] ? log(0.8) : log(0.2)
    wetgrass_score = log(0.99)
    expected_score = sprinkler_score + rain_score + wetgrass_score
    @test isapprox(score, expected_score)

    # a proposed choice is scored
    t = ProgramTrace()
    propose!(t, "sprinkler", Bool)
    score, _ = @generate!(model(), t)
    expected_score = if t["cloudy"]
        t["sprinkler"] ? log(0.1) : log(0.9)
    else
        t["sprinkler"] ? log(0.4) : log(0.6)
    end
    @test isapprox(score, expected_score)

    # scoring works with sub-traces
    toplevel = @program () begin
        @g(model(), "sub")
    end
    t = ProgramTrace()
    constrain!(t, "cloudy", true)
    constrain!(t, "sprinkler", true)
    constrain!(t, "rain", true)
    constrain!(t, "wetgrass", true)
    toplevel_trace = ProgramTrace()
    set_subtrace!(toplevel_trace, "sub", t)
    score, _ = @generate!(toplevel(), toplevel_trace)
    expected_score = log(0.3) + log(0.1) + log(0.8) + log(0.99)
    @test isapprox(score, expected_score)
end

