@testset "gamma" begin

        # test against Distributions.Gamma since we re-implement the density
        # function ourselves for purposes of AD
        import Distributions

        x = 0.1
        k = 0.2
        s = 0.3
        @test isapprox(regenerate(Gen.Gamma(), x, k, s), Distributions.logpdf(Distributions.Gamma(k, s), x))
end
