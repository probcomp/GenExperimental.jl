# samples a uniform discrete on the range [lower, lower + 1, ..., upper]
# upper is inclusive

struct UniformDiscrete <: Gen.Module{Int64} end

import Distributions

function regenerate(::UniformDiscrete, x::Int64, lower::Int, upper::Int)
    d = Distributions.DiscreteUniform(concrete(lower), concrete(upper)) 
    Distributions.logpdf(d, x)
end

function simulate(duniform::UniformDiscrete, lower::Int, upper::Int)
    x = rand(Distributions.DiscreteUniform(concrete(lower), concrete(upper)))
    (x, regenerate(duniform, x, lower, upper))
end

register_module(:uniform_discrete, UniformDiscrete())

uniform_discrete(lower::Int, upper::Int) = simulate(UniformDiscrete(), lower, upper)[1]
export uniform_discrete
export UniformDiscrete
