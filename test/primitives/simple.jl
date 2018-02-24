import Distributions

@testset "flip" begin
    @test isapprox(logpdf(Flip(), true, 0.1), log(0.1))
    @test isapprox(logpdf(Flip(), false, 0.1), log(0.9))
end

@testset "logflip" begin
    @test isapprox(logpdf(LogFlip(), true, log(0.1)), log(0.1))
    @test isapprox(logpdf(LogFlip(), false, log(0.1)), log(0.9))
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

@testset "poisson" begin

    # test against Distributions.Poisson since we re-implement the density
    # function ourselves for purposes of AD
    x = 4
    lambda = 0.6
    @test isapprox(logpdf(Poisson(), x, lambda), Distributions.logpdf(Distributions.Poisson(lambda), x))
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

@testset "delta" begin
    # floating point
    @test logpdf(delta_float64, 4.123, 4.124) == -Inf
    @test logpdf(delta_float64, 4.123, 4.123) == 0.
    @test rand(delta_float64, 4.123) == 4.123

    # boolean
    @test logpdf(delta_bool, true, true) == 0.
    @test logpdf(delta_bool, true, false) == -Inf
    @test logpdf(delta_bool, false, true) == -Inf
    @test logpdf(delta_bool, false, false) == 0.
    @test rand(delta_bool, true) == true
    @test rand(delta_bool, false) == false

    # int
    @test logpdf(delta_int, 5, 5) == 0.
    @test logpdf(delta_int, 5, 6) == -Inf
    @test rand(delta_int, 5) == 5
end

