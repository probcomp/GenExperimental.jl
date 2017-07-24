@testset "SIR crash test" begin

    num_runs = 0
    model = @program () begin
        num_runs += 1
        std = @g(Gen.gamma(1., 1.), "std")
        mu = @g(normal(0., 1.), "mu")
        @g(normal(mu, 2.), "x")
    end

    # propose mu from the proposal program
    # propose std using resimulation
    sir = SIRGenerator(model, model, Dict(["mu" => ("mu", Float64)]), Set(["std"]))

    constraints = ProgramTrace()
    x_constraint = 1.123
    constrain!(constraints, "x", x_constraint)

    function check_output(val::ProgramTrace)
        # the latents are constrained
        @test haskey(val, "mu")
        @test mode(val, "mu") == Gen.constrain
        @test haskey(val, "std")
        @test mode(val, "std") == Gen.constrain

        # the data remains constrained
        @test haskey(val, "x")
        @test mode(val, "x") == Gen.constrain
        @test val["x"] == x_constraint
    end

    # record
    sir_trace = AtomicTrace(ProgramTrace)
    num_runs = 0
    (score, val) = generate!(sir, (10, (), (), constraints), sir_trace)
    # NOTE: we run it once as the model, once as proposal per particle, and one
    # extra run for the model joint score
    @test num_runs == 21
    @test score == 0.
    check_output(val)
    check_output(value(sir_trace))

    # propose
    sir_trace = AtomicTrace(ProgramTrace)
    propose!(sir_trace, (), ProgramTrace)
    num_runs = 0
    (score, val) = generate!(sir, (10, (), (), constraints), sir_trace)
    @test num_runs == 21
    @test score != 0.
    check_output(val)
    check_output(value(sir_trace))

    # constrain
    sir_trace = AtomicTrace(ProgramTrace)
    constrained_trace = ProgramTrace()
    constrain!(constrained_trace, "x", x_constraint)
    constrain!(constrained_trace, "mu", 0.123)
    constrain!(constrained_trace, "std", 0.321)
    constrain!(sir_trace, (), constrained_trace)
    num_runs = 0
    (score, val) = generate!(sir, (10, (), (), constraints), sir_trace)
    @test num_runs == 21
    @test score != 0.
    check_output(val)
    check_output(value(sir_trace))
    @test val["mu"] == 0.123
    @test val["std"] == 0.321
end

@testset "SIR score for N=1 special case" begin

    # the score should just be the log proposal density

    model = @program () begin
        mu = @g(normal(0., 1.), "mu")
        @g(normal(mu, 2.), "x")
    end

    proposal = @program () begin
        @g(normal(1., 1.), "foo")
    end

    sir = SIRGenerator(model, proposal, Dict(["foo" => ("mu", Float64)]))

    # observations
    constraints = ProgramTrace()
    x_constraint = 1.123
    constrain!(constraints, "x", x_constraint)

    # propose
    sir_trace = AtomicTrace(ProgramTrace)
    propose!(sir_trace, (), ProgramTrace)
    (score, val) = generate!(sir, (1, (), (), constraints), sir_trace)
    @test isapprox(score, logpdf(normal, val["mu"], 1., 1.))

    # constrain
    sir_trace = AtomicTrace(ProgramTrace)
    constrained_trace = ProgramTrace()
    constrain!(constrained_trace, "x", x_constraint)
    constrain!(constrained_trace, "mu", 0.123)
    constrain!(sir_trace, (), constrained_trace)
    (score, val) = generate!(sir, (1, (), (), constraints), sir_trace)
    @test val["mu"] == 0.123
    @test isapprox(score, logpdf(normal, val["mu"], 1., 1.))

end













