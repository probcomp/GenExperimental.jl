
none = AddressTrie()

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

        # traced primitive generator invocation with syntax sugar
        x4 = @g(normal(0, 1), 3)

        # untraced generator invocation is identical to function invocation
        x5 = bar(0.5)
        @test x5 == "something"

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
    simulate!(foo, (), none, none, t)
    @test haskey(t, 3)
    @test haskey(t, 7)
    @test haskey(t, 8)
    @test haskey(t, 9)
end

@testset "program definition syntaxes" begin

    # anonymous, no arguments
    foo1 = @program () begin nothing end
    @test simulate!(foo1, (), none, none, ProgramTrace()) == (0., nothing)

    # anonymous, single typed argument
    foo2 = @program (x::Int) begin x end
    @test simulate!(foo2, (1,), none, none, ProgramTrace()) == (0., 1)

    # anonymous, single untyped argument
    foo3 = @program (x) begin x end
    @test simulate!(foo3, (1,), none, none, ProgramTrace()) == (0., 1)

    # anonymous, multiple argument with one untyped
    foo4 = @program (x::Int, y) begin x, y end
    @test simulate!(foo4, (1, 2), none, none, ProgramTrace()) == (0., (1, 2))

    # anonymous, no arguments
    @program bar1() begin nothing end
    @test simulate!(bar1, (), none, none, ProgramTrace()) == (0., nothing)

    # anonymous, single typed argument
    @program bar2(x::Int) begin x end
    @test simulate!(bar2, (1,), none, none, ProgramTrace()) == (0., 1)

    # anonymous, single untyped argument
    @program bar3(x) begin x end
    @test simulate!(bar3, (1,), none, none, ProgramTrace()) == (0., 1)

    # anonymous, multiple argument with one untyped
    @program bar4(x::Int, y) begin x, y end
    @test simulate!(bar4, (1, 2), none, none, ProgramTrace()) == (0., (1, 2))
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
    @test mode(t, "x") == Gen.constrain
end

@testset "intervening on primitive generator invocation" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()
    intervene!(t, "x", 2.3)
    score, val = @generate!(foo(), t)
    @test score == 0.
    @test val == 2.3
    @test mode(t, "x") == Gen.intervene
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
    @test mode(t, "x") == Gen.intervene
end

@testset "tagging arbitrary expressions" begin
    foo = @program () begin
        
        # a generator invocation labelled as an expression
        x = @e(normal(0, 1), "x")

        # an expression
        y = @e(x + 1, "y")

        (x, y)
    end
    t = ProgramTrace()
    
    # test that values are recorded
    score, val = @generate!(foo(), t)
    @test haskey(t, "x")
    @test haskey(t, "y")
    @test score == 0.
    @test val == (t["x"], t["y"])
    @test t["y"] == t["x"] + 1

    # test intervention
    intervene!(t, "x", 2.)
    intervene!(t, "y", 3.)
    score, val = @generate!(foo(), t)
    @test t["x"] == 2.
    @test t["y"] == 3.
    @test score == 0.
    @test val == (2., 3.)
end

# TODO test haskey
# haskey does NOT check if there is a subtrace or not
# it checks if there is a value at a particular atomic address

@testset "proposing from atomic trace" begin
    t = AtomicTrace(Float64)
    propose!(t, (), Float64)
    score, val = @generate!(normal(0, 1), t)
    @test score == logpdf(normal, val, 0, 1)
end

@testset "proposing from program trace" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()
    propose!(t, "x", Float64)
    score, val = @generate!(foo(), t)
    @test score == logpdf(Normal(), val, 0, 1)
    @test mode(t, "x") == Gen.propose
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

@testset "haskey" begin

    # program trace
    # NOTE: haskey returns false even if there is a subtrace at that address
    # haskey indicates whether an atomic value is present at an address
    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()
    @test !haskey(t, "x")
    set_subtrace!(t, "x", AtomicTrace(Float64))
    @test !haskey(t, "x")
    constrain!(t, "x", 1.2)
    @test haskey(t, "x")

    # atomic trace
    t = AtomicTrace(Float64)
    @test !haskey(t, ())
    constrain!(t, (), 1.2)
    @test haskey(t, ())
end



@testset "delete!" begin

    foo = @program () begin @g(normal(0, 1), "x") end

    # deleting a subtrace from a program trace
    t = ProgramTrace()
    @generate!(foo(), t)
    delete!(t, "x")
    @test !haskey(t, "x")

    # deleting the value from an atomic trace
    t = AtomicTrace(Float64)
    @generate!(normal(0, 1), t)
    @test haskey(t, ())
    delete!(t, ())
    @test !haskey(t, ())
end


@testset "release!" begin

    foo = @program () begin @g(normal(0, 1), "x") end
    t = ProgramTrace()

    # releasing a constraint
    # the subtrace remains, but it is no longer constrained
    # what happens to the value is undefined (its up to the trace type)
    constrain!(t, "x", 1.1)
    score, _ = @generate!(foo(), t)
    @test value(t, "x") == 1.1
    @test score != 0.
    release!(t, "x")
    @test haskey(t, "x")
    score, _ = @generate!(foo(), t)
    @test score == 0.
    @test value(t, "x") != 1.1
    @test mode(t, "x") == Gen.record
    
    # releasing an intervention
    # the subtrace remains, but it is no longer constrained
    # what happens to the value is undefined (its up to the trace type)
    intervene!(t, "x", 1.1)
    @generate!(foo(), t)
    @test value(t, "x") == 1.1
    release!(t, "x")
    @test haskey(t, "x")
    @generate!(foo(), t)
    @test value(t, "x") != 1.1
    @test mode(t, "x") == Gen.record

    # releasing a proposal
    # the subtrace remains, but it is no longer constrained
    # what happens to the value is undefined (its up to the trace type
    propose!(t, "x", Float64)
    score, _ = @generate!(foo(), t)
    @test score != 0.
    release!(t, "x")
    @test haskey(t, "x")
    score, _ = @generate!(foo(), t)
    @test score == 0.
    @test mode(t, "x") == Gen.record
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

@testset "aliasing" begin

    sub = @program () begin

        # a generator invocation
        a = @g(normal(0, 1), "a")

        # an expression
        b = @e(a + 1, "b")
    end

    foo = @program () begin
        # 2 -> 1/a
        # 3 -> 1/b
        @alias(2, (1, "a"))
        @alias(3, (1, "b"))
        x = @g(sub(), 1)
    end

    # retrieve recorded value at alias
    t = ProgramTrace()
    @generate!(foo(), t)
    @test haskey(t, (1, "a"))
    @test haskey(t, (1, "b"))
    @test haskey(t, 2)
    @test haskey(t, 3)
    @test t[1, "a"] == t[2]
    @test t[1, "b"] == t[3]
    
    # constrain an alias
    t = ProgramTrace()
    @generate!(foo(), t)
    constrain!(t, 2, 2.2)
    @test t[2] == 2.2
    # NOTE: until we have run generate! again, the alias is just a separate subtrace
    @test t[1, "a"] != t[2]
    score, _ = @generate!(foo(), t)
    @test t[2] == 2.2
    @test t[1, "a"] == t[2]
    @test score == logpdf(normal, 2.2, 0, 1)

    # intervene on an alias
    t = ProgramTrace()
    @generate!(foo(), t)
    intervene!(t, 2, 2.2)
    intervene!(t, 3, 3.3)
    @test t[2] == 2.2
    @test t[3] == 3.3
    # NOTE: until we have run generate! again, the alias is just a separate subtrace
    @test t[1, "a"] != t[2]
    @test t[1, "b"] != t[3]
    score, _ = @generate!(foo(), t)
    @test t[2] == 2.2
    @test t[1, "a"] == t[2]
    @test t[3] == 3.3
    @test t[1, "b"] == t[3]
    @test score == 0.

    # propose an alias
    t = ProgramTrace()
    propose!(t, 2, Float64)
    score, _ = @generate!(foo(), t)
    @test t[1, "a"] == t[2]
    @test score == logpdf(normal, t[2], 0, 1)

end

@testset "autodiff on trace score" begin
    
    foo = @program () begin
        # NOTE: these are local because the test is run in a local scope
        # so they would otherwise overwrite the values outside of this function
        local mu = @e(0., "mu")
        local log_std = @e(0., "logstd")
        @g(normal(mu, exp(log_std)), "x")
    end

    t = ProgramTrace()
    x = 2.1
    mu = 0.
    std = 1.
    constrain!(t, "x", x)
    parametrize!(t, "mu", mu)
    parametrize!(t, "logstd", log(std))
    @generate!(foo(), t)

    tape = Tape()
    gen_mu = makeGenValue(mu, tape)
    gen_std = makeGenValue(std, tape)
    backprop(logpdf(normal, x, gen_mu, gen_std))
    @test isapprox(partial(t["mu"]), partial(gen_mu))
    @test isapprox(partial(t["logstd"]), partial(gen_std))
end
