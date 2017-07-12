@testset "mvnormal" begin

        # NOTE: as long as regenerate is implemented using
        # Distributions.MultivariateNormal, this test is not useflu.  However,
        # when regenerate is re-implemented for use with AD
        # (https://github.com/probcomp/Gen.jl/issues/53) this test will become
        # useful.

        import Distributions
        x = [1., 1.]
        mu = [1.0, 1.0]
        std = [0.1 0.0; 0.0 0.1]
        @test isapprox(regenerate(MultivariateNormal(), x, mu, std), 
                       Distributions.logpdf(Distributions.MultivariateNormal(mu, std), x))
end
