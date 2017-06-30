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

    a_scalar = 1.300181
    b_scalar = 1.45245
    a_vector = [1.4123, 4.3452]
    b_vector = [8.3453, 0.9913]
    a_matrix = [1.4123 4.3452 40.1; 10.123 0.9314 0.11] # 2 rows, 3 columns
    b_matrix = [4.13 33.123 5.32431; 4.314 5.1341 8.09] # 2 rows, 3 columns


    @testset "elementwise-divide" begin

        # scalar / scalar
        adtest(/, a_scalar, b_scalar)

        # scalar ./ vector
        for i=1:2
            adtest((a, b) -> (a ./ b)[i], a_scalar, a_vector)
        end

        # vector / scalar
        for i=1:2
            adtest((a, b) -> (a ./ b)[i], a_vector, a_scalar)
        end

        # vector ./ vector
        for i=1:2
            adtest((a, b) -> (a ./ b)[i], a_vector, b_vector)
        end

    end

    @testset "elementwise-multiply" begin

        # scalar * scalar
        adtest(*, a_scalar, b_scalar)

        # scalar * vector
        for i=1:2
            adtest((a, b) -> (a * b)[i], a_scalar, a_vector)
        end

        # vector * scalar
        for i=1:2
            adtest((a, b) -> (a * b)[i], a_vector, a_scalar)
        end

        # vector .* vector
        for i=1:2
            adtest((a, b) -> (a .* b)[i], a_vector, b_vector)
        end

        # scalar * matrix
        for i=1:6
            adtest((a, b) -> (a * b)[i], a_scalar, a_matrix)
        end

        # matrix * scalar
        for i=1:6
            adtest((a, b) -> (a * b)[i], a_matrix, a_scalar)
        end

        # matrix .* matrix
        for i=1:6
            adtest((a, b) -> (a .* b)[i], a_matrix, b_matrix)
        end
    end


end


