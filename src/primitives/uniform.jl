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
