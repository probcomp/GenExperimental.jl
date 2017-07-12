@testset "uniform_discrete" begin

        x = 5
        lower = 1
        upper = 10
        @test isapprox(regenerate(UniformDiscrete(), x, lower, upper), log(1./upper))
end
