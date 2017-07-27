import Distributions

"""
Bernoulli generator, with singleton `flip`

    generate!(flip, (prob,), trace::AtomicTrace{Bool})

    flip(prob)

Score is differentiable with respect to `prob`
"""
struct Flip <: AssessableAtomicGenerator{Bool} end

simulate{T}(::Flip, prob::T) = rand() < prob
logpdf{T}(::Flip, value::Bool, prob::T) = value ? log(prob) : log(1.0 - prob)

register_primitive(:flip, Flip)


"""
Gamma generator, with singleton `gamma`.

    generate!(gamma, (k, S), trace::AtomicTrace{Float64})

    gamma(k, S)

`k` is the shape parameter, `s` is the scale parameter.

Score is differentiable with respect to `k` and `s` and output

NOTE: `gamma` conflicts with `Base.gamma`. Use `Gen.gamma`
"""
struct Gamma <: AssessableAtomicGenerator{Float64} end

# k = shape, s = scale
function logpdf{M,N,O}(::Gamma, x::M, k::N, s::O)
    (k - 1.0) * log(x) - (x / s) - k * log(s) - lgamma(k)
end

function simulate{M,N}(gamma::Gamma, k::M, s::N)
    rand(Distributions.Gamma(k, s))
end

register_primitive(:gamma, Gamma)


"""
Univariate normal generator, with singleton `normal`.

    generate!(normal, (mu, std), trace::AtomicTrace{Float64})

    normal(mu, std)

Score is differentiable with respect to `mu` and `std` and output
"""
struct Normal <: AssessableAtomicGenerator{Float64} end

function logpdf{M,N,O}(::Normal, x::M, mu::N, std::O)
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function simulate{M,N}(normal::Normal, mu::M, std::N)
    rand(Distributions.Normal(concrete(mu), concrete(std)))
end

register_primitive(:normal, Normal)


"""
Multivariate normal generator, with singleton `mvnormal`.

    generate!(normal, (mu::Vector{Float64}, sigma::Matrix{Float64}), trace::AtomicTrace{Vector{Float64}})

    mvnormal(mu, sigma)

Score is not (yet) differentiable with respect to `mu` and `std` and output [#53](https://github.com/probcomp/Gen.jl/issues/53).
"""
struct MultivariateNormal <: AssessableAtomicGenerator{Vector{Float64}} end

function logpdf(::MultivariateNormal, x::Vector{Float64}, mu::Vector{Float64}, std::Matrix{Float64})
    d = Distributions.MvNormal(concrete(mu), concrete(std)) 
    Distributions.logpdf(d, x)
end

function simulate(::MultivariateNormal, mu::Vector{Float64}, std::Matrix{Float64})
    rand(Distributions.MvNormal(concrete(mu), concrete(std)))
end

register_primitive(:mvnormal, MultivariateNormal)


"""
Uniform continuous generator, with singleton `uniform`.

    generate!(uniform, (lower::Real, upper::Real), trace::AtomicTrace{Float64})

    uniform(lower, upper)

Score is not differentiable.
"""
struct UniformContinuous <: AssessableAtomicGenerator{Float64} end

function logpdf(::UniformContinuous, x::Float64, lower::Real, upper::Real)
    x < lower || x > upper ? -Inf : -log(upper - lower)
end

function simulate(::UniformContinuous, lower::Real, upper::Real)
    rand() * (upper - lower) + lower
end

register_primitive(:uniform, UniformContinuous)


"""
Uniform discrete generator, with singleton `uniform_discrete`.

    generate!(uniform_discrete, (lower::Int, upper::Int), trace::AtomicTrace{Int})

    uniform_discrete(lower, upper)

Return value is uniformly distributed integer in the set `[lower, lower + 1, ..., upper]` (i.e. upper limit is inclusive).

Score is not differentiable.
"""
struct UniformDiscrete <: AssessableAtomicGenerator{Int64} end

function logpdf(::UniformDiscrete, x::Int64, lower::Int, upper::Int)
    d = Distributions.DiscreteUniform(concrete(lower), concrete(upper)) 
    Distributions.logpdf(d, x)
end

function simulate(::UniformDiscrete, lower::Int, upper::Int)
    rand(Distributions.DiscreteUniform(concrete(lower), concrete(upper)))
end

register_primitive(:uniform_discrete, UniformDiscrete)


"""
Categorical generator with log-space probability vector argument, with singleton `categoriceal_log`.

    generate!(categorical_log, (scores::Vector{Float64}), trace::AtomicTrace{Int})

    categorical_log(scores)

Return value is sampled from categorical distribution on the set `[1, ..., length(scores)]`, where `scores` is the possibly unnormalized vector of log-probabilities.

Score is not (yet) differentiable [#65](https://github.com/probcomp/Gen.jl/issues/65)
"""
struct CategoricalLog <: AssessableAtomicGenerator{Int64} end

function logpdf(::CategoricalLog, x::Int64, scores::Vector{Float64})
    (scores - logsumexp(scores))[x]
end

function simulate(::CategoricalLog, scores::Vector{Float64})
    probs = exp.(scores - logsumexp(scores))
    rand(Distributions.Categorical(probs))
end

register_primitive(:categorical_log, CategoricalLog)


"""
Degenerate generator  with singleton `nil`.

    generate!(nil, (), trace::AtomicTrace{Nil})

    nil()

Sample from a degenerate distribution which places all mass on the singleton value `Nil()`.

Score is not differentiable.
"""

struct Nil <: AssessableAtomicGenerator{Float64} end

logpdf{T}(::Nil, x::T) = x == Nil() ? 0.0 : -Inf

simulate(::Nil) = Nil()

register_primitive(:nil, Nil)
