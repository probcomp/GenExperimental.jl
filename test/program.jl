
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

    t = DictTrace()
    simulate!(foo, (), none, none, t)
    @test haskey(t, 3)
    @test haskey(t, 7)
    @test haskey(t, 8)
    @test haskey(t, 9)
end

@testset "program definition syntaxes" begin

    # anonymous, no arguments
    foo1 = @program () begin nothing end
    @test simulate!(foo1, (), none, none, DictTrace()) == (0., nothing)

    # anonymous, single typed argument
    foo2 = @program (x::Int) begin x end
    @test simulate!(foo2, (1,), none, none, DictTrace()) == (0., 1)

    # anonymous, single untyped argument
    foo3 = @program (x) begin x end
    @test simulate!(foo3, (1,), none, none, DictTrace()) == (0., 1)

    # anonymous, multiple argument with one untyped
    foo4 = @program (x::Int, y) begin x, y end
    @test simulate!(foo4, (1, 2), none, none, DictTrace()) == (0., (1, 2))

    # anonymous, no arguments
    @program bar1() begin nothing end
    @test simulate!(bar1, (), none, none, DictTrace()) == (0., nothing)

    # anonymous, single typed argument
    @program bar2(x::Int) begin x end
    @test simulate!(bar2, (1,), none, none, DictTrace()) == (0., 1)

    # anonymous, single untyped argument
    @program bar3(x) begin x end
    @test simulate!(bar3, (1,), none, none, DictTrace()) == (0., 1)

    # anonymous, multiple argument with one untyped
    @program bar4(x::Int, y) begin x, y end
    @test simulate!(bar4, (1, 2), none, none, DictTrace()) == (0., (1, 2))
end

@testset "lexical scope" begin
    x = 123

    # using one definition syntax
    foo = @program () begin x end
    @test simulate!(foo, (), none, none, DictTrace())[2] == x

    # using the other definition syntax
    @program bar() begin x end
    @test simulate!(bar, (), none, none, DictTrace())[2] == x
end

@testset "regenerate! and simulate!" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = DictTrace()

    # outputs: none
    # conditions: none
    # regenerate! and simulate! have the same behavior in this case
    # score is 0., and the value is overwritten
    t["x"] = 2.3
    score, val = regenerate!(foo, (), none, none, t)
    @test score == 0.
    @test val != 2.3
    @test t["x"] == val
    score, val = simulate!(foo, (), none, none, t)
    @test score == 0.
    @test val != 2.3
    @test t["x"] == val

    # outputs: none
    # conditions: ()
    # regenerate! and simulate! have the same behavior in this case
    # score is 0., and the return value is the given value
    t[()] = 2.3
    score, val = regenerate!(foo, (), none, AddressTrie(()), t)
    @test score == 0.
    @test val == 2.3
    t[()] = 2.3
    score, val = simulate!(foo, (), none, AddressTrie(()), t)
    @test score == 0.
    @test val == 2.3

    # NOTE: outputs: () and conditions: none is not a valid query for a ProbabilisticProgram

    # outputs: "x"
    # conditions: none
    # regenerate!
    # score is is the log-density, and does not overwrite the value
    t["x"] = 2.3
    score, val = regenerate!(foo, (), AddressTrie("x"), none, t)
    @test score == logpdf(Normal(), 2.3, 0, 1)
    @test val == 2.3
    @test t["x"] == val

    # outputs: "x"
    # conditions: none
    # simulate!
    # score is is the log-density, and overwrites the value
    t["x"] = 2.3
    score, val = simulate!(foo, (), AddressTrie("x"), none, t)
    @test val != 2.3
    @test t["x"] == val
    @test score == logpdf(Normal(), val, 0, 1)

    # outputs: none
    # conditions: "x"
    # regenerate! and simulate! have the same behavior in this case
    # score is zero, and the value is taken from the trace
    t["x"] = 2.3
    score, val = regenerate!(foo, (), none, AddressTrie("x"), t)
    @test score == 0.
    @test val == 2.3
    @test t["x"] == val
    score, val = simulate!(foo, (), none, AddressTrie("x"), t)
    @test score == 0.
    @test val == 2.3
    @test t["x"] == val
end

@testset "tagging arbitrary expressions" begin
    foo = @program () begin
        
        # a generator invocation labelled as an expression
        x = @e(normal(0, 1), "x")

        # an expression
        y = @e(x + 1, "y")

        (x, y)
    end
    t = DictTrace()
    
    # test that values are recorded
    score, val = simulate!(foo, (), none, none, t)
    @test haskey(t, "x")
    @test haskey(t, "y")
    @test score == 0.
    @test val == (t["x"], t["y"])
    @test t["y"] == t["x"] + 1

    # test conditioning on arbitrary expression
    t["x"] = 2.
    t["y"] = 3.
    conditions = AddressTrie()
    push!(conditions, "x")
    push!(conditions, "y")
    score, val = simulate!(foo, (), none, conditions, t)
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
    score, val = simulate!(normal, (0, 1), AddressTrie(()), none, t)
    @test score == logpdf(normal, val, 0, 1)
    @test val == t[()]
end

@testset "proposing from program trace" begin
    foo = @program () begin @g(normal(0, 1), "x") end
    t = DictTrace()
    score, val = simulate!(foo, (), AddressTrie("x"), none, t)
    @test score == logpdf(Normal(), val, 0, 1)
    @test val == t["x"]
end

@testset "different syntaxes for retrieving value from a trace" begin
    bar = @program () begin @g(flip(0.5), "x") end
    foo = @program () begin
        x = @g(bar(), "bar")
        y = @g(normal(0, 1), "y")
    end

    t = DictTrace()
    score, val = simulate!(foo, (), none, none, t)

    # test a top-level address
    @test t[("y",)] == t["y"]

    # test a hierarchical address
    @test t[("bar", "x")] == get_subtrace(t, "bar")["x"]
end

@testset "haskey" begin

    # program trace
    # NOTE: haskey returns false even if there is a subtrace at that address
    # haskey indicates whether an atomic value is present at an address
    foo = @program () begin @g(normal(0, 1), "x") end
    t = DictTrace()
    @test !haskey(t, "x")
    set_subtrace!(t, "x", AtomicTrace(Float64))
    @test !haskey(t, "x")
    t["x"] = 1.2
    @test haskey(t, "x")

    # atomic trace
    t = AtomicTrace(Float64)
    @test !haskey(t, ())
    t[()] = 1.2
    @test haskey(t, ())
end

@testset "delete!" begin

    foo = @program () begin @g(normal(0, 1), "x") end

    # deleting a value a program trace
    t = DictTrace()
    simulate!(foo, (), none, none, t)
    delete!(t, "x")
    @test !haskey(t, "x")

    # deleing the output value from a program trace
    @test haskey(t, ())
    delete!(t, ())
    @test !haskey(t, ())

    # deleting the value from an atomic trace
    t = AtomicTrace(Float64)
    simulate!(normal, (0, 1), none, none, t)
    @test haskey(t, ())
    delete!(t, ())
    @test !haskey(t, ())
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
    t = DictTrace()
    for addr in ["foo", 1, 2, 3]
        set_subtrace!(t, addr, DictTrace())
    end
    t[("foo", "mu")] = 4.
    t[("foo", "std")] = 1.
    t[(1, "x")] = 4.5
    t[(2, "x")] = 4.3
    t[(3, "x")] = 4.2
    outputs = AddressTrie(
        ("foo", "mu"),
        ("foo", "std"),
        (1, "x"),
        (2, "x"),
        (3, "x"))

    (score, value) = regenerate!((@program () begin
        sampler = @g(foo(), "foo")
        x1 = @g(sampler(), 1)
        x2 = @g(sampler(), 2)
        x3 = @g(sampler(), 3)
        (x1, x2, x3)
    end), (), outputs, none, t)
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

    # the score for regenerate! is the sum of output scores
    t = DictTrace()
    t["cloudy"] = true
    t["sprinkler"] = true
    t["rain"] = true
    t["wetgrass"] = true
    score, _ = regenerate!(model, (), AddressTrie("cloudy", "sprinkler", "rain", "wetgrass"),
                           none, t)
    expected_score = log(0.3) + log(0.1) + log(0.8) + log(0.99)
    @test isapprox(score, expected_score)

    # an address that is not in the output set is not scored
    t = DictTrace()
    t["sprinkler"] = true
    t["rain"] = true
    t["wetgrass"] = true
    score, _ = regenerate!(model, (), AddressTrie("sprinkler", "rain", "wetgrass"), none, t)
    sprinkler_score = t["cloudy"] ? log(0.1) : log(0.4)
    rain_score = t["cloudy"] ? log(0.8) : log(0.2)
    wetgrass_score = log(0.99)
    expected_score = sprinkler_score + rain_score + wetgrass_score
    @test isapprox(score, expected_score)

    # the score for simulate! is the sum of the output scores
    t = DictTrace()
    score, _ = simulate!(model, (), AddressTrie("sprinkler"), none, t)
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
    t = DictTrace()
    outputs = AddressTrie()
    for addr in ["cloudy", "sprinkler", "rain", "wetgrass"]
        t[addr] = true
        push!(outputs, ("sub", addr))
    end
    toplevel_trace = DictTrace()
    set_subtrace!(toplevel_trace, "sub", t)
    score, _ = regenerate!(toplevel, (), outputs, none, toplevel_trace)
    expected_score = log(0.3) + log(0.1) + log(0.8) + log(0.99)
    @test isapprox(score, expected_score)
end


@testset "autodiff on trace score" begin
    
    foo = @program () begin
        # NOTE: these are local because the test is run in a local scope
        # so they would otherwise overwrite the values outside of this function
        local mu = @e(0., "mu")
        local log_std = @e(0., "logstd")
        @g(normal(mu, exp(log_std)), "x")
    end

    trace = DictTrace()
	tape1 = Tape()
    x = 2.1
	trace["x"] = x
	trace["mu"] = GenScalar(0., tape1)
	trace["logstd"] = GenScalar(0., tape1)
	conditions = AddressTrie()
	push!(conditions, "mu")
	push!(conditions, "logstd")
    (score, _) = regenerate!(foo, (), AddressTrie("x"), conditions, trace)
	backprop(score)

    tape2 = Tape()
	mu = GenScalar(0., tape2)
	logstd = GenScalar(0., tape2)
    backprop(logpdf(normal, x, mu, exp(logstd)))
    @test isapprox(partial(trace["mu"]), partial(mu))
    @test isapprox(partial(trace["logstd"]), partial(logstd))
end

@testset "aliases" begin

    @program inner() begin
        @g(flip(0.1), "b")
    end

    @program f() begin
        @alias(("a", "b"), "exposed")
        @g(inner(), "a")
    end

    # simulate without any outputs or conditions
    trace = DictTrace()
    simulate!(f, (), AddressTrie(), AddressTrie(), trace)
    @test haskey(trace, "exposed")
    @test haskey(trace, ("a", "b"))
    @test trace["exposed"] == trace[("a", "b")]

    # simulate with alias as an output
    trace = DictTrace()
    score, val = simulate!(f, (), AddressTrie("exposed"), AddressTrie(), trace)
    @test haskey(trace, "exposed")
    @test haskey(trace, ("a", "b"))
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test isapprox(score, val ? log(0.1) : log(0.9))

    # regenerate with alias as an output
    trace = DictTrace()
    trace["exposed"] = true
    score, val = regenerate!(f, (), AddressTrie("exposed"), AddressTrie(), trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test val == true
    @test isapprox(score, log(0.1))

    # simulate with actual address as output
    trace = DictTrace()
    score, val = simulate!(f, (), AddressTrie(("a", "b")), AddressTrie(), trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test isapprox(score, val ? log(0.1) : log(0.9))

    # regenerate with actual address as an output
    trace = DictTrace()
    set_subtrace!(trace, "a", DictTrace())
    trace[("a", "b")] = true
    score, val = regenerate!(f, (), AddressTrie(("a", "b")), AddressTrie(), trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test val == true
    @test isapprox(score, log(0.1))

    # simulate with alias as a condition
    trace = DictTrace()
    trace["exposed"] = true
    _, val = simulate!(f, (), AddressTrie(), AddressTrie("exposed"), trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test val == true

    # regenerate with alias as a condition
    trace = DictTrace()
    trace["exposed"] = true
    _, val = regenerate!(f, (), AddressTrie(), AddressTrie("exposed"),  trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test val == true

    # simulate with actual address as a condition
    trace = DictTrace()
    set_subtrace!(trace, "a", DictTrace())
    trace[("a", "b")] = true
    _, val = simulate!(f, (), AddressTrie(), AddressTrie(("a", "b")), trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test val == true

    # regenerate with actual address as a condition
    trace = DictTrace()
    set_subtrace!(trace, "a", DictTrace())
    trace[("a", "b")] = true
    _, val = regenerate!(f, (), AddressTrie(), AddressTrie(("a", "b")), trace)
    @test trace["exposed"] == trace[("a", "b")]
    @test val == trace["exposed"]
    @test val == true


end

@testset "per-generator scores" begin

    model = @program () begin
        cloudy = @g(flip(0.3), "cloudy")
        sprinkler = @g(flip(cloudy ? 0.1 : 0.4), "sprinkler")
    end

    t = DictTrace()
    t["cloudy"] = true
    t["sprinkler"] = true
    regenerate!(model, (), AddressTrie("cloudy", "sprinkler"), none, t)
    @test isapprox(get_score(t, "cloudy"), log(0.3))
    @test isapprox(get_score(t, "sprinkler"), log(0.1))

    t = DictTrace()
    regenerate!(model, (), none, none, t)
    @test isapprox(get_score(t, "cloudy"), t["cloudy"] ? log(0.3) : log(1-0.3))
    if t["cloudy"]
        @test isapprox(get_score(t, "sprinkler"), t["sprinkler"] ? log(0.1) : log(1-0.1))
    else
        @test isapprox(get_score(t, "sprinkler"), t["sprinkler"] ? log(0.4) : log(1-0.4))
    end

    t = DictTrace()
    simulate!(model, (), none, none, t)
    @test isapprox(get_score(t, "cloudy"), t["cloudy"] ? log(0.3) : log(1-0.3))
    if t["cloudy"]
        @test isapprox(get_score(t, "sprinkler"), t["sprinkler"] ? log(0.1) : log(1-0.1))
    else
        @test isapprox(get_score(t, "sprinkler"), t["sprinkler"] ? log(0.4) : log(1-0.4))
    end
end
