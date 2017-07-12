@testset "uniform" begin

        @test isapprox(regenerate(UniformContinuous(), 1., 0., 2.), log(0.5))
        @test regenerate(UniformContinuous(), -1., 0., 2.) == -Inf
end
