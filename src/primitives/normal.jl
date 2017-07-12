import Distributions

struct Normal <: Gen.Module{Float64} end

function regenerate{M,N,O}(::Normal, x::M, mu::N, std::O)
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function simulate{M,N}(normal::Normal, mu::M, std::N)
    x = rand(Distributions.Normal(concrete(mu), concrete(std)))
    (x, regenerate(normal, x, mu, std))
end

register_module(:normal, Normal())

normal{M,N}(mu::M, std::N) = simulate(Normal(), mu, std)[1]
export normal
