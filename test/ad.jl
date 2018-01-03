
dx = 1e-6


function finite_difference(f, val::Real)
    x_pos = val + dx
    x_neg = val - dx
    (f(x_pos) - f(x_neg)) / (2. * dx)
end

ColumnOrRowVector = Union{Vector{W}, RowVector{W, Vector{W}}} where W<:Real

function finite_difference(f, val::ColumnOrRowVector{W}) where W<:Real
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

    # test that the operator behavior matches the operator on the built-in types

    # gen - gen
    tape = Tape()
    a = makeGenValue(a_val, tape)
    b = makeGenValue(b_val, tape)
    result = f(a, b)
    @test f(a_val, b_val) == concrete(result)

    # gen - concrete
    tape = Tape()
    a = makeGenValue(a_val, tape)
    result = f(a, b_val)
    @test f(a_val, b_val) == concrete(result)

    # concrete - gen 
    tape = Tape()
    b = makeGenValue(b_val, tape)
    result = f(a_val, b)
    @test f(a_val, b_val) == concrete(result)

    # test backpropagation
    # this works for scalars, vectors, and matrix results.
    # note that scalars have scalar[1] = scalar
    for i=1:length(result)
        tape = Tape()
        a = makeGenValue(a_val, tape)
        b = makeGenValue(b_val, tape)
        f_i = (x, y) -> f(x, y)[i] * 2. # the 2. is so that the result adjoint is not always 1
        result_i = f_i(a, b)
        backprop(result_i)
        @test isapprox(partial(a), finite_difference((x) -> f_i(x, b_val), a_val))
        @test isapprox(partial(b), finite_difference((x) -> f_i(a_val, x), b_val))
    end
end

function adtest(f, a_val)

    # test that the operator behavior matches the operator on the built-in types
    tape = Tape()
    a = makeGenValue(a_val, tape)
    result = f(a)
    @test f(a_val) == concrete(result)

    # test backpropagation
    # this works for scalars, vectors, and matrix results.
    # note that scalars have scalar[1] = scalar
    for i=1:length(result)
        tape = Tape()
        a = makeGenValue(a_val, tape)
        f_i = (x) -> f(x)[i] * 2. # the 2. is so that the result adjoint is not always 1
        result_i = f_i(a)
        backprop(result_i)
        @test isapprox(partial(a), finite_difference(f_i, a_val), atol=1e-16)
    end
end

@testset "automatic differentiation" begin

    a_scalar = 1.300181
    b_scalar = 1.45245
    a_vector = [1.4123, 4.3452]
    b_vector = [8.3453, 0.9913]
    a_row_vector = a_vector'
    b_row_vector = b_vector'
    a_matrix = [1.4123 2.3452 2.1; 2.123 0.9314 0.11] # 2 rows, 3 columns
    b_matrix = [4.13 33.123 5.32431; 4.314 5.1341 8.09] # 2 rows, 3 columns

    @testset "add" begin
        adtest(+, a_scalar, b_scalar)
        adtest((a, b) -> ewise(+, a, b), a_scalar, b_scalar)
        adtest((a, b) -> (a + b), a_scalar, a_vector)
        adtest((a, b) -> (a + b), a_scalar, a_row_vector)
        adtest((a, b) -> ewise(+, a, b), a_scalar, a_vector)
        adtest((a, b) -> ewise(+, a, b), a_scalar, a_row_vector)
        adtest((a, b) -> (a + b), a_vector, a_scalar)
        adtest((a, b) -> (a + b), a_row_vector, a_scalar)
        adtest((a, b) -> ewise(+, a, b), a_vector, a_scalar)
        adtest((a, b) -> ewise(+, a, b), a_row_vector, a_scalar)
        adtest((a, b) -> (a + b), a_vector, b_vector)
        adtest((a, b) -> ewise(+, a, b), a_vector, b_vector)
        adtest((a, b) -> (a + b), a_row_vector, b_row_vector)
        adtest((a, b) -> ewise(+, a, b), a_row_vector, b_row_vector)

        # column vector .+ row vector
        # TODO not implemented yet
        #adtest((a, b) -> ewise(+, a, b), a_vector, b_row_vector)

        # row vector .+ column vector
        # TODO not implemented yet
        #adtest((a, b) -> ewise(+, a, b), a_row_vector, b_vector)

        # scalar + matrix
        # TODO not implemented yet
        #adtest((a, b) -> (a + b), a_scalar, a_matrix)

        # matrix + scalar
        # TODO not implemented yet
        #adtest((a, b) -> (a + b), a_matrix, a_scalar)

        # scalar .+ matrix
        # TODO not implemented yet
        #adtest((a, b) -> ewise(+, a, b), a_scalar, a_matrix)

        # matrix .+ scalar
        # TODO not implemented yet
        #adtest((a, b) -> ewise(+, a, b), a_matrix, a_scalar)

        # matrix + matrix
        # TODO not implemented yet
        #adtest((a, b) -> (a + b), a_matrix, a_scalar)

        # matrix .+ matrix
        # TODO not implemented yet
        #adtest((a, b) -> ewise(+, a, b), a_matrix, a_scalar)

        # matrix .+ vector (broadcast)
        # TODO not implemented yet

        # vector .+ matrix (broadcast)
        # TODO not implemented yet

        # TODO test cartesian indexing into a matrix
    end

    @testset "subtract" begin
        adtest(-, a_scalar, b_scalar)
        adtest((a, b) -> ewise(-, a, b), a_scalar, b_scalar)
        adtest((a, b) -> (a - b), a_scalar, a_vector)
        adtest((a, b) -> (a - b), a_scalar, a_row_vector)
        adtest((a, b) -> ewise(-, a, b), a_scalar, a_vector)
        adtest((a, b) -> ewise(-, a, b), a_scalar, a_row_vector)
        adtest((a, b) -> (a - b), a_vector, a_scalar)
        adtest((a, b) -> (a - b), a_row_vector, a_scalar)
        adtest((a, b) -> ewise(-, a, b), a_vector, a_scalar)
        adtest((a, b) -> ewise(-, a, b), a_row_vector, a_scalar)
        adtest((a, b) -> (a - b), a_vector, b_vector)
        adtest((a, b) -> ewise(-, a, b), a_vector, b_vector)
        adtest((a, b) -> (a - b), a_row_vector, b_row_vector)
        adtest((a, b) -> ewise(-, a, b), a_row_vector, b_row_vector)

        # column vector .- row vector
        # TODO not implemented yet
        #adtest((a, b) -> ewise(-, a, b), a_vector, b_row_vector)

        # row vector .- column vector
        # TODO not implemented yet
        #adtest((a, b) -> ewise(-, a, b), a_row_vector, b_vector)

        # scalar - matrix
        # TODO not implemented yet
        #adtest((a, b) -> (a - b), a_scalar, a_matrix)

        # matrix - scalar
        # TODO not implemented yet
        #adtest((a, b) -> (a - b), a_matrix, a_scalar)

        # scalar .- matrix
        # TODO not implemented yet
        #adtest((a, b) -> ewise(-, a, b), a_scalar, a_matrix)

        # matrix .- scalar
        # TODO not implemented yet
        #adtest((a, b) -> ewise(-, a, b), a_matrix, a_scalar)

        # matrix - matrix
        # TODO not implemented yet
        #adtest((a, b) -> (a - b), a_matrix, a_scalar)

        # matrix .- matrix
        # TODO not implemented yet
        #adtest((a, b) -> ewise(-, a, b), a_matrix, a_scalar)

        # matrix .- vector (broadcast)
        # TODO not implemented yet

        # vector .- matrix (broadcast)
        # TODO not implemented yet

        # TODO test cartesian indexing into a matrix
    end

    @testset "divide" begin
        adtest(/, a_scalar, b_scalar)
        adtest((a, b) -> ewise(/, a, b), a_scalar, b_scalar)
        adtest((a, b) -> ewise(/, a, b), a_scalar, a_vector)
        adtest((a, b) -> ewise(/, a, b), a_scalar, a_row_vector)
        adtest((a, b) -> (a / b), a_vector, a_scalar)
        adtest((a, b) -> (a / b), a_row_vector, a_scalar)
        adtest((a, b) -> ewise(/, a, b), a_vector, a_scalar)
        adtest((a, b) -> ewise(/, a, b), a_row_vector, a_scalar)
        adtest((a, b) -> ewise(/, a, b), a_vector, b_vector)
        adtest((a, b) -> ewise(/, a, b), a_row_vector, b_row_vector)

        # column vector ./ row vector
        # TODO not implemented yet
        #adtest((a, b) -> ewise(/, a, b), a_vector, b_row_vector)

        # row vector ./ column vector
        # TODO not implemented yet
        #adtest((a, b) -> ewise(/, a, b), a_row_vector, b_vector)

        adtest((a, b) -> ewise(/, a, b), a_scalar, a_matrix)
        adtest((a, b) -> (a / b), a_matrix, a_scalar)
        adtest((a, b) -> ewise(/, a, b), a_matrix, a_scalar)
        adtest((a, b) -> ewise(/, a, b), a_matrix, a_scalar)

        # matrix ./ vector (broadcast)
        # TODO not implemented yet

        # vector ./ matrix (broadcast)
        # TODO not implemented yet

        # TODO test cartesian indexing into a matrix
    end

    @testset "elementwise-multiply" begin
        adtest(*, a_scalar, b_scalar)
        adtest((a, b) -> ewise(*, a, b), a_scalar, b_scalar)
        adtest((a, b) -> (a * b), a_scalar, a_vector)
        adtest((a, b) -> (a * b), a_scalar, a_row_vector)
        adtest((a, b) -> ewise(*, a, b), a_scalar, a_vector)
        adtest((a, b) -> ewise(*, a, b), a_scalar, a_row_vector)
        adtest((a, b) -> (a * b), a_vector, a_scalar)
        adtest((a, b) -> (a * b), a_row_vector, a_scalar)
        adtest((a, b) -> ewise(*, a, b), a_vector, a_scalar)
        adtest((a, b) -> ewise(*, a, b), a_row_vector, a_scalar)
        adtest((a, b) -> ewise(*, a, b), a_vector, b_vector)
        adtest((a, b) -> ewise(*, a, b), a_row_vector, b_row_vector)
        adtest((a, b) -> ewise(*, a, b), a_vector, b_row_vector)
        adtest((a, b) -> ewise(*, a, b), a_row_vector, b_vector)
        adtest((a, b) -> (a * b), a_scalar, a_matrix)
        adtest((a, b) -> (a * b), a_matrix, a_scalar)
        adtest((a, b) -> ewise(*, a, b), a_matrix, a_scalar)

        # matrix .* vector (broadcast)
        # TODO not implemented yet

        # vector .* matrix (broadcast)
        # TODO not implemented yet

        # TODO test cartesian indexing into a matrix
    end

    @testset "matrix-multiply" begin
        adtest((a, b) -> (a * b), a_matrix', a_matrix)
        adtest((a, b) -> (a * b), a_matrix', a_vector)
        #adtest((a, b) -> (a * b), a_row_vector, a_matrix) # TODO
        adtest((a, b) -> (a * b), a_vector, b_row_vector)
        adtest((a, b) -> (a * b), a_row_vector, b_vector)
    end

    @testset "transpose" begin
        adtest((a) -> a', a_scalar)
        adtest((a) -> a', a_matrix)
        adtest((a) -> a', a_vector)
        adtest((a) -> a', a_row_vector)
    end

    @testset "unary plus" begin
        adtest(+, a_scalar)
        adtest(+, a_vector)
        adtest(+, a_row_vector)
        adtest(+, a_matrix)
    end

    @testset "unary minus" begin
        adtest(-, a_scalar)
        adtest(-, a_vector)
        adtest(-, a_row_vector)
        adtest(-, a_matrix)
    end

    @testset "exp" begin
        adtest(exp, a_scalar)
        adtest((a) -> ewise(exp, a), a_vector)
        adtest((a) -> ewise(exp, a), a_row_vector)
        adtest((a) -> ewise(exp, a), a_matrix)
    end

    @testset "log" begin
        adtest(log, a_scalar)
        adtest((a) -> ewise(log, a), a_vector)
        adtest((a) -> ewise(log, a), a_row_vector)
        adtest((a) -> ewise(log, a), a_matrix)
    end

    @testset "log1p" begin
        adtest(log1p, a_scalar)
        adtest((a) -> ewise(log1p, a), a_vector)
        adtest((a) -> ewise(log1p, a), a_row_vector)
        adtest((a) -> ewise(log1p, a), a_matrix)
    end

    @testset "lgamma" begin
        adtest(lgamma, a_scalar)
        adtest((a) -> ewise(lgamma, a), a_vector)
        adtest((a) -> ewise(lgamma, a), a_row_vector)
        adtest((a) -> ewise(lgamma, a), a_matrix)
    end

    @testset "sigmoid" begin
        adtest(sigmoid, a_scalar)
        adtest((a) -> ewise(sigmoid, a), a_vector)
        adtest((a) -> ewise(sigmoid, a), a_row_vector)
        adtest((a) -> ewise(sigmoid, a), a_matrix)
    end

    @testset "sum" begin
        adtest(sum, a_scalar)
        adtest(sum, a_vector)
        adtest(sum, a_row_vector)
        adtest(sum, a_matrix)
    end

    @testset "prod" begin
        adtest(prod, a_scalar)
        adtest(prod, a_vector)
        adtest(prod, a_row_vector)
        adtest(prod, a_matrix)
    end

    @testset "logsumexp" begin

        # test when argument is a GenColumnVector
        adtest(logsumexp, a_vector)

        # test when the argument is a mixed Vector{Any}
        tape = Tape()
        a = [GenScalar(1.0, tape), 2.5, GenScalar(2.3, tape)]
        result = logsumexp(a)
        @test logsumexp(map(concrete, a)) == concrete(result)

        tape = Tape()
        a = [GenScalar(1.0, tape), 2.5, GenScalar(2.3, tape)]
        f = (x) -> logsumexp(x) * 2. # the 2. is so that the result adjoint is not always 1
        result = f(a)
        backprop(result)
        expected = finite_difference(f, map(concrete, a))
        @test isapprox(partial(a[1]), expected[1])
        @test isapprox(partial(a[3]), expected[3])
    end

    @testset "range index into column vector" begin
        col_vector = [1., 2., 3.]
        range_index_sum(v) = (prod(v[2:3]) / sum(v[2:3]))
        adtest(range_index_sum, col_vector)
    end

    @testset "range index into row vector" begin
        row_vector = [1., 2., 3.]'
        range_index_sum(v) = (prod(v[2:3]) / sum(v[2:3]))
        adtest(range_index_sum, row_vector)
    end

    @testset "zero grad" begin
        tape = Tape()
        a = GenScalar(1., tape)
        b = GenColumnVector(ones(2), tape)
        c = GenRowVector(transpose(ones(2)), tape)
        d = GenMatrix(ones(2, 2), tape)
        backprop(sum((b * (c * a)) * d))
        partial_a = partial(a)
        @test partial_a != 0.
        zero_grad!(tape)
        @test partial(a) == 0.
        @test partial(b) == zeros(2)
        @test partial(c) == transpose(zeros(2))
        @test partial(d) == zeros(2, 2)
        backprop(sum((b * (c * a)) * d))
        @test partial(a) == partial_a
    end


end
