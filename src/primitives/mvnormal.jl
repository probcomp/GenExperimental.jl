import Distributions

struct MultivariateNormal <: Gen.Module{Vector{Float64}} end

function regenerate(::MultivariateNormal, x::Vector{Float64}, mu::Vector{Float64}, std::Matrix{Float64})
    d = Distributions.MvNormal(concrete(mu), concrete(std)) 
    Distributions.logpdf(d, x)
end

function simulate(mvnormal::MultivariateNormal, mu::Vector{Float64}, std::Matrix{Float64})
    x = rand(Distributions.MvNormal(concrete(mu), concrete(std)))
    (x, regenerate(mvnormal, x, mu, std))
end

register_module(:mvnormal, MultivariateNormal())

mvnormal(mu::Vector{Float64}, std::Matrix{Float64}) = simulate(MultivariateNormal(), mu, std)[1]
export mvnormal
export MultivariateNormal
