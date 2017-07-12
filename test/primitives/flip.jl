@testset "flip" begin

        @test isapprox(regenerate(Flip(), true, 0.1), log(0.1))
        @test isapprox(regenerate(Flip(), false, 0.1), log(0.9))
end
