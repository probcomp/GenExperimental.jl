import Distributions

# k = shape, s = scale

struct Gamma <: Gen.Module{Float64} end

function regenerate{M,N,O}(::Gamma, x::M, k::N, s::O)
    (k - 1.0) * log(x) - (x / s) - k * log(s) - lgamma(k)
end

function simulate{M,N}(gamma::Gamma, k::M, s::N)
    x = rand(Distributions.Gamma(k, s))
    (x, regenerate(gamma, x, k, s))
end

register_module(:gamma, Gamma())

gamma{M,N}(k::M, s::N) = simulate(Gamma(), k, s)[1]
export gamma
