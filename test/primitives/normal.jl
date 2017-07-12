@testset "normal" begin

        # test against Distributions.Normal since we re-implement the density
        # function ourselves for purposes of AD
        import Distributions

        x = 0.1
        mu = 0.2
        std = 0.3
        @test isapprox(regenerate(Normal(), x, mu, std), Distributions.logpdf(Distributions.Normal(mu, std), x))
end
