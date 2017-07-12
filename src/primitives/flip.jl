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
export Flip
