include("../src/ad/types.jl")
include("../src/ad/operators.jl")

using Base.Test

dx = 1e-6

function finite_difference(f, val::Float64)
    x_pos = val + dx
    x_neg = val - dx
    (f(x_pos) - f(x_neg)) / (2. * dx)
end

function finite_difference(f, val::Vector{Float64})
    grad = zeros(val)
    for i=1:length(val)
        e_vec = zeros(val)
        e_vec[i] = 1.
        x_pos = val + dx * e_vec
        x_neg = val - dx * e_vec
        grad[i] = (f(x_pos) - f(x_neg)) / (2. * dx)
    end
    grad
end

function finite_difference(f, val::Matrix{Float64})
    rows, cols = size(val)
    grad = zeros(val)
    for row=1:rows
        for col=1:cols
            e_vec = zeros(val)
            e_vec[row, col] = 1.
            x_pos = val + dx * e_vec
            x_neg = val - dx * e_vec
            grad[row, col] = (f(x_pos) - f(x_neg)) / (2. * dx)
        end
    end
    grad
end


function adtest(f, a_val, b_val)
    tape = Tape()
    a = makeGenValue(a_val, tape)
    b = makeGenValue(b_val, tape)
    c = f(a, b)
    backprop(c)
    @test isapprox(partial(a), finite_difference((x) -> f(x, b_val), a_val))
    @test isapprox(partial(b), finite_difference((x) -> f(a_val, x), b_val))
    @test f(datum(a), datum(b)) == datum(f(a, b))
end

@testset "automatic differentiation" begin


    @testset "elementwise-divide" begin

        # scalar / scalar
        adtest(/, 1.300181, 4.131234)

        # scalar ./ vector
        adtest((a, b) -> (a ./ b)[1], 1.300181, [1.4123, 4.3452])
        adtest((a, b) -> (a ./ b)[2], 1.300181, [1.4123, 4.3452])

        # vector / scalar
        adtest((a, b) -> (a ./ b)[1], [1.4123, 4.3452], 1.300181)
        adtest((a, b) -> (a ./ b)[2], [1.4123, 4.3452], 1.300181)

        # vector ./ vector
        adtest((a, b) -> (a ./ b)[1], [1.4123, 4.3452], [5.245, 0.4924])
        adtest((a, b) -> (a ./ b)[2], [1.4123, 4.3452], [5.245, 0.4924])

    end

    @testset "elementwise-multiply" begin

        # scalar * scalar
        adtest(*, 1.300181, 4.131234)

        # scalar * vector
        adtest((a, b) -> (a * b)[1], 1.300181, [1.4123, 4.3452])
        adtest((a, b) -> (a * b)[2], 1.300181, [1.4123, 4.3452])

        # vector * scalar
        adtest((a, b) -> (a * b)[1], [1.4123, 4.3452], 1.300181)
        adtest((a, b) -> (a * b)[2], [1.4123, 4.3452], 1.300181)

        # vector .* vector
        adtest((a, b) -> (a .* b)[1], [1.4123, 4.3452], [5.245, 0.4924])
        adtest((a, b) -> (a .* b)[2], [1.4123, 4.3452], [5.245, 0.4924])



    end


end


