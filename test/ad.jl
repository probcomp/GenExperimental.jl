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

function adtest(f, a_val)
    tape = Tape()
    a = makeGenValue(a_val, tape)
    c = f(a)
    backprop(c)
    @test isapprox(partial(a), finite_difference(f, a_val))
    @test f(datum(a)) == datum(f(a))
end


@testset "automatic differentiation" begin

    a_scalar = 1.300181
    b_scalar = 1.45245
    a_vector = [1.4123, 4.3452]
    b_vector = [8.3453, 0.9913]
    a_matrix = [1.4123 4.3452 40.1; 10.123 0.9314 0.11] # 2 rows, 3 columns
    b_matrix = [4.13 33.123 5.32431; 4.314 5.1341 8.09] # 2 rows, 3 columns

    @testset "unary plus" begin

        # +scalar
        adtest(+, a_scalar)

        # +vector
        for i=1:length(a_vector)
            adtest((x) -> (+x)[i], a_vector)
        end

        # +matrix
        for i=1:length(a_matrix)
            adtest((x) -> (+x)[i], a_matrix)
        end
    end

    @testset "unary minus" begin

        # -scalar
        adtest(-, a_scalar)

        # -vector
        for i=1:length(a_vector)
            adtest((x) -> (-x)[i], a_vector)
        end

        # -matrix
        for i=1:length(a_matrix)
            adtest((x) -> (-x)[i], a_matrix)
        end
    end

    @testset "add" begin

        # scalar + scalar
        adtest(+, a_scalar, b_scalar)

        # scalar + vector
        for i=1:length(a_scalar)
            adtest((a, b) -> (a + b)[i], a_scalar, a_vector)
        end

        # vector + scalar
        for i=1:length(a_scalar)
            adtest((a, b) -> (a + b)[i], a_vector, a_scalar)
        end

        # vector + vector
        for i=1:length(a_scalar)
            adtest((a, b) -> (a + b)[i], a_vector, b_vector)
        end

        # scalar + matrix
        # TODO not implemented yet

        # matrix + scalar
        # TODO not implemented yet

        # matrix .+ vector (broadcast)
        # TODO not implemented yet

        # vector .+ matrix (broadcast)
        # TODO not implemented yet

        # matrix + matrix
        # TODO not implemented yet
    end

    @testset "subtract" begin

        # scalar - scalar
        adtest(+, a_scalar, b_scalar)

        # scalar - vector
        for i=1:length(a_vector)
            adtest((a, b) -> (a - b)[i], a_scalar, a_vector)
        end

        # vector - scalar
        for i=1:length(a_vector)
            adtest((a, b) -> (a - b)[i], a_vector, a_scalar)
        end

        # vector - vector
        for i=1:length(a_vector)
            adtest((a, b) -> (a - b)[i], a_vector, b_vector)
        end

        # scalar - matrix
        # TODO not implemented yet

        # matrix - scalar
        # TODO not implemented yet

        # matrix .- vector (broadcast)
        # TODO not implemented yet

        # vector .- matrix (broadcast)
        # TODO not implemented yet

        # matrix - matrix
        # TODO not implemented yet
    end

    @testset "divide" begin

        # scalar / scalar
        adtest(/, a_scalar, b_scalar)

        # scalar ./ vector
        for i=1:length(a_vector)
            adtest((a, b) -> (a ./ b)[i], a_scalar, a_vector)
        end

        # vector / scalar
        for i=1:length(a_vector)
            adtest((a, b) -> (a ./ b)[i], a_vector, a_scalar)
        end

        # vector ./ vector
        for i=1:length(a_vector)
            adtest((a, b) -> (a ./ b)[i], a_vector, b_vector)
        end

        # scalar ./ matrix
        # TODO not implemented yet

        # matrix / scalar
        # TODO not implemented yet

        # matrix ./ vector (broadcast)
        # TODO not implemented yet

        # vector ./ matrix (broadcast)
        # TODO not implemented yet
    end

    @testset "elementwise-multiply" begin

        # scalar * scalar
        adtest(*, a_scalar, b_scalar)

        # scalar * vector
        for i=1:length(a_vector)
            adtest((a, b) -> (a * b)[i], a_scalar, a_vector)
        end

        # vector * scalar
        for i=1:length(a_vector)
            adtest((a, b) -> (a * b)[i], a_vector, a_scalar)
        end

        # vector .* vector
        for i=1:length(a_vector)
            adtest((a, b) -> (a .* b)[i], a_vector, b_vector)
        end

        # scalar * matrix
        for i=1:length(a_matrix)
            adtest((a, b) -> (a * b)[i], a_scalar, a_matrix)
        end

        # scalar * matrix, cartesian index into result matrix
        for row=1:size(a_matrix)[1]
            for col=1:size(a_matrix)[2]
                adtest((a, b) -> (a * b)[row, col], a_scalar, a_matrix)
            end
        end

        # matrix * scalar, linear index into result matrix
        for i=1:length(a_matrix)
            adtest((a, b) -> (a * b)[i], a_matrix, a_scalar)
        end

        # matrix * scalar, cartesian index into result matrix
        for row=1:size(a_matrix)[1]
            for col=1:size(a_matrix)[2]
                adtest((a, b) -> (a * b)[row, col], a_matrix, a_scalar)
            end
        end

        # matrix .* matrix, linear index into result matrix
        for i=1:length(a_matrix)
            adtest((a, b) -> (a .* b)[i], a_matrix, b_matrix)
        end

        # matrix .* matrix, cartesian index into result matrix
        for row=1:size(a_matrix)[1]
            for col=1:size(a_matrix)[2]
                adtest((a, b) -> (a .* b)[row, col], a_matrix, b_matrix)
            end
        end

        # matrix .* vector (broadcast)
        # TODO not implemented yet

        # vector .* matrix (broadcast)
        # TODO not implemented yet
    end

    @testset "matrix-multiply" begin

        # matrix * matrix, linear index into result matrix 
        for i=1:length(a_matrix)
            adtest((a, b) -> (a * b)[i], a_matrix', a_matrix)
        end

        # matrix * matrix, cartesian index into result matrix
        for row=1:size(a_matrix)[1]
            for col=1:size(a_matrix)[2]
                adtest((a, b) -> (a * b)[row, col], a_matrix', a_matrix)
            end
        end

        # matrix * vector 
        for i=1:length(a_matrix' * a_vector)
            adtest((a, b) -> (a * b)[i], a_matrix', a_vector)
        end
    end

    @testset "exp" begin
        
        # exp(scalar)
        adtest(exp, a_scalar)

        # exp(vector)
        for i=1:length(a_vector)
            adtest((x) -> exp.(x)[i], a_vector)
        end

        # exp(matrix)
        for i=1:length(a_matrix)
            adtest((x) -> exp.(x)[i], a_matrix)
        end
    end

    @testset "log" begin
        
        # log(scalar)
        adtest(log, a_scalar)

        # log(vector)
        for i=1:length(a_vector)
            adtest((x) -> log.(x)[i], a_vector)
        end

        # log(matrix)
        for i=1:length(a_matrix)
            adtest((x) -> log.(x)[i], a_matrix)
        end
    end

    @testset "lgamma" begin

        import SpecialFunctions.lgamma
        import SpecialFunctions.digamma
        
        # lgamma(scalar)
        adtest(lgamma, a_scalar)

        # lgamma(vector)
        for i=1:length(a_vector)
            adtest((x) -> lgamma.(x)[i], a_vector)
        end

        # lgamma(matrix)
        for i=1:length(a_matrix)
            adtest((x) -> lgamma.(x)[i], a_matrix)
        end
    end

end
