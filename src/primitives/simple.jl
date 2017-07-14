import Distributions

####################
# Flip (Bernoulli) #
####################

struct Flip <: AssessableAtomicGenerator{Bool} end

simulate{T}(::Flip, prob::T) = rand() < prob
logpdf{T}(::Flip, value::Bool, prob::T) = value ? log(prob) : log(1.0 - prob)

register_primitive(:flip, Flip)


#########
# Gamma #
#########

struct Gamma <: AssessableAtomicGenerator{Float64} end

# k = shape, s = scale
function logpdf{M,N,O}(::Gamma, x::M, k::N, s::O)
    (k - 1.0) * log(x) - (x / s) - k * log(s) - lgamma(k)
end

function simulate{M,N}(gamma::Gamma, k::M, s::N)
    rand(Distributions.Gamma(k, s))
end

register_primitive(:gamma, Gamma)


##########
# Normal #
##########

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


#######################
# Multivariate normal #
#######################

struct MultivariateNormal <: AssessableAtomicGenerator{Vector{Float64}} end

function logpdf(::MultivariateNormal, x::Vector{Float64}, mu::Vector{Float64}, std::Matrix{Float64})
    d = Distributions.MvNormal(concrete(mu), concrete(std)) 
    Distributions.logpdf(d, x)
end

function simulate(::MultivariateNormal, mu::Vector{Float64}, std::Matrix{Float64})
    rand(Distributions.MvNormal(concrete(mu), concrete(std)))
end

register_primitive(:mvnormal, MultivariateNormal)


######################
# Uniform continuous #
######################

struct UniformContinuous <: AssessableAtomicGenerator{Float64} end

function logpdf(::UniformContinuous, x::Float64, lower::Real, upper::Real)
    x < lower || x > upper ? -Inf : -log(upper - lower)
end

function simulate(::UniformContinuous, lower::Real, upper::Real)
    rand() * (upper - lower) + lower
end

register_primitive(:uniform, UniformContinuous)


####################
# Uniform discrete #
####################

# uniform discrete on the range [lower, lower + 1, ..., upper]
# upper is inclusive

struct UniformDiscrete <: AssessableAtomicGenerator{Int64} end

import Distributions

function logpdf(::UniformDiscrete, x::Int64, lower::Int, upper::Int)
    d = Distributions.DiscreteUniform(concrete(lower), concrete(upper)) 
    Distributions.logpdf(d, x)
end

function simulate(::UniformDiscrete, lower::Int, upper::Int)
    rand(Distributions.DiscreteUniform(concrete(lower), concrete(upper)))
end

register_primitive(:uniform_discrete, UniformDiscrete)


##########################################
# Nil (degenerate discrete distribution) #
##########################################

struct Nil <: AssessableAtomicGenerator{Float64} end

logpdf{T}(::Nil, x::T) = x == Nil() ? 0.0 : -Inf

simulate(::Nil) = Nil()

register_primitive(:nil, Nil)
