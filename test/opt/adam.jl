struct ADAMTestObjective end

@testset "adam optimizer" begin

    # a very simple convex objective function
    optimum = [0.5, 0.1]
    function Gen.fgrad_estimate(::ADAMTestObjective, params::Vector{Float64})
        A = [0.5 0; 0 1.0]
        f = -(params - optimum)' * A * (params - optimum)
        grad = -(A' + A) * (params - optimum)
        (f, grad)
    end
    init_params = ones(2)
    adam_params = ADAMParams(0.001, 0.9, 0.999, 1e-8)
    opt = ADAMOptimizer(ADAMTestObjective(), adam_params, 10, false)
    result = optimize(opt, init_params, 10000)
    val, _ = fgrad_estimate(ADAMTestObjective(), result.params)
    @test isapprox(val, 0., atol=1e-16)
    @test isapprox(result.params, optimum)
end
