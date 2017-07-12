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
export Nil
