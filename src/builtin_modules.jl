import Distributions

# flip ---------------------------------------------------

struct Flip <: Gen.Module{Bool} end

function regenerate{T}(::Flip, x::Bool, p::T)
    x ? log(p) : log(1.0 - p)
end

function simulate{T}(flip::Flip, p::T)
    x = rand() < p
    (x, regenerate(flip, x, p))
end

register_module(:flip, Flip())

flip{T}(p::T) = simulate(Flip(), p)[1]
export flip


# uniform -------------------------------------------------

struct Uniform <: Gen.Module{Float64} end

function regenerate(::Uniform, x::Float64, lower::Real, upper::Real)
    x < lower || x > upper ? -Inf : -log(upper - lower)
end

function simulate(::Uniform, lower::Real, upper::Real)
    x = rand() * (upper - lower) + lower
    (x, -log(upper - lower))
end

register_module(:uniform, Uniform())

uniform(lower::Real, upper::Real) = simulate(Uniform(), lower, upper)[1]
export uniform


# impossible ---------------------------------------------

struct Nil <: Gen.Module{Float64} end

function regenerate{T}(::Nil, x::T)
    x == Nil() ? 0.0 : -Inf
end

function simulate(::Nil)
    Nil(), 0.0
end

register_module(:nil, Nil())

nil() = simulate(Nil())[1]
export nil


# normal -------------------------------------------------

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


# gamma --------------------------------------------------
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
