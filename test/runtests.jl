using Gen
using Distributions
using Base.Test

include("ad.jl")

@testset "primitives" begin

    # bernoulli
    p = 0.1
    @test isapprox(flip_regenerate(true, p), logpdf(Bernoulli(p), true))

    # normal
    x = 0.1
    mu = 0.2
    std = 0.3
    @test isapprox(normal_regenerate(x, mu, std), logpdf(Normal(mu, std), x))

    # gamma
    x = 0.1
    k = 0.2
    s = 0.3
    @test isapprox(gamma_regenerate(x, k, s), logpdf(Gamma(k, s), x))

end

nothing
