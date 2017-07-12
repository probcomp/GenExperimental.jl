@testset "nil" begin

        @test isapprox(regenerate(Nil(), Nil()), log(1.0))
        @test regenerate(Nil(), 5) == -Inf
end
