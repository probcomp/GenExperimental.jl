import Distributions

@testset "flip" begin
    @test isapprox(logpdf(Flip(), true, 0.1), log(0.1))
    @test isapprox(logpdf(Flip(), false, 0.1), log(0.9))
end

@testset "beta" begin

    # test against Distributions.Beta since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    a = 0.2
    b = 0.3
    @test isapprox(logpdf(Beta(), x, a, b), Distributions.logpdf(Distributions.Beta(a, b), x))
end

@testset "gamma" begin

    # test against Distributions.Gamma since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    k = 0.2
    s = 0.3
    @test isapprox(logpdf(Gamma(), x, k, s), Distributions.logpdf(Distributions.Gamma(k, s), x))
end

@testset "inv_gamma" begin

    # test against Distributions.Gamma since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    shape = 0.2
    scale = 0.3
    @test isapprox(logpdf(InverseGamma(), x, shape, scale),
                   Distributions.logpdf(Distributions.InverseGamma(shape, scale), x))
end

@testset "normal" begin

    # test against Distributions.Normal since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    mu = 0.2
    std = 0.3
    @test isapprox(logpdf(Normal(), x, mu, std), Distributions.logpdf(Distributions.Normal(mu, std), x))
end

@testset "cauchy" begin

    # test against Distributions.Cauchy since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    loc = 0.2
    scale = 0.3
    @test isapprox(logpdf(Cauchy(), x, loc, scale), Distributions.logpdf(Distributions.Cauchy(loc, scale), x))
end

@testset "mvnormal" begin

    # NOTE: as long as regenerate is implemented using
    # Distributions.MultivariateNormal, this test is not useflu.  However,
    # when regenerate is re-implemented for use with AD
    # (https://github.com/probcomp/Gen.jl/issues/53) this test will become
    # useful.

    x = [1., 1.]
    mu = [1.0, 1.0]
    std = [0.1 0.0; 0.0 0.1]
    @test isapprox(logpdf(MultivariateNormal(), x, mu, std), 
                   Distributions.logpdf(Distributions.MultivariateNormal(mu, std), x))
end

@testset "uniform" begin
    @test isapprox(logpdf(UniformContinuous(), 1., 0., 2.), log(0.5))
    @test logpdf(UniformContinuous(), -1., 0., 2.) == -Inf
end

@testset "uniform_discrete" begin
    x = 5
    lower = 1
    upper = 10
    @test isapprox(logpdf(UniformDiscrete(), x, lower, upper), log(1./upper))
end

@testset "categorical log space" begin
    x = 3
    probs = [0.2, 0.3, 0.5]
    unnormalized = probs * 5
    log_unnormalized = log.(unnormalized)
    @test isapprox(logpdf(CategoricalLog(), x, log_unnormalized), log(probs[x]))
end

@testset "nil" begin
    @test isapprox(logpdf(Nil(), Nil()), log(1.0))
    @test logpdf(Nil(), 5) == -Inf
end


