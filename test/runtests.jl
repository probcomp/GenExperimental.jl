using Gen
using Distributions
using Base.Test

include("ad.jl")

@testset "primitive modules" begin

    # bernoulli
    p = 0.1
    @test isapprox(regenerate(Gen.Flip(), true, p), logpdf(Bernoulli(p), true))

    # normal
    x = 0.1
    mu = 0.2
    std = 0.3
    @test isapprox(regenerate(Gen.Normal(), x, mu, std), logpdf(Normal(mu, std), x))

    # gamma
    x = 0.1
    k = 0.2
    s = 0.3
    @test isapprox(regenerate(Gen.Gamma(), x, k, s), logpdf(Gamma(k, s), x))

end

nothing
