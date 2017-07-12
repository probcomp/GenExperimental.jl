struct UniformContinuous <: Gen.Module{Float64} end

function regenerate(::UniformContinuous, x::Float64, lower::Real, upper::Real)
    x < lower || x > upper ? -Inf : -log(upper - lower)
end

function simulate(::UniformContinuous, lower::Real, upper::Real)
    x = rand() * (upper - lower) + lower
    (x, -log(upper - lower))
end

register_module(:uniform, UniformContinuous())

uniform(lower::Real, upper::Real) = simulate(UniformContinuous(), lower, upper)[1]
export uniform
export UniformContinuous
