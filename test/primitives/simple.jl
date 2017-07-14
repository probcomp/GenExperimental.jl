import Distributions

@testset "flip" begin
    @test isapprox(logpdf(flip, true, 0.1), log(0.1))
    @test isapprox(logpdf(flip, false, 0.1), log(0.9))
end

@testset "gamma" begin

    # test against Distributions.Gamma since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    k = 0.2
    s = 0.3
    @test isapprox(logpdf(Gen.gamma, x, k, s), Distributions.logpdf(Distributions.Gamma(k, s), x))
end

@testset "normal" begin

    # test against Distributions.Normal since we re-implement the density
    # function ourselves for purposes of AD
    x = 0.1
    mu = 0.2
    std = 0.3
    @test isapprox(logpdf(normal, x, mu, std), Distributions.logpdf(Distributions.Normal(mu, std), x))
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
    @test isapprox(logpdf(mvnormal, x, mu, std), 
                   Distributions.logpdf(Distributions.MultivariateNormal(mu, std), x))
end

@testset "uniform" begin
    @test isapprox(logpdf(uniform, 1., 0., 2.), log(0.5))
    @test logpdf(uniform, -1., 0., 2.) == -Inf
end

@testset "uniform_discrete" begin
    x = 5
    lower = 1
    upper = 10
    @test isapprox(logpdf(uniform_discrete, x, lower, upper), log(1./upper))
end

@testset "nil" begin
    @test isapprox(logpdf(nil, nil), log(1.0))
    @test logpdf(nil, 5) == -Inf
end
