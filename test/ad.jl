include("../src/ad/types.jl")
include("../src/ad/operators.jl")

using Base.Test

dx = 1e-6


function finite_difference(f, val::Real)
    x_pos = val + dx
    x_neg = val - dx
    (f(x_pos) - f(x_neg)) / (2. * dx)
end

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
    tape = Tape()
    a = makeGenValue(a_val, tape)
    b = makeGenValue(b_val, tape)
    result = f(a, b)
    @test f(a_val, b_val) == datum(result)

    # test backpropagation
    # this works for scalars, vectors, and matrix results.
    # note that scalars have scalar[1] = scalar
    for i=1:length(result)
        tape = Tape()
        a = makeGenValue(a_val, tape)
        b = makeGenValue(b_val, tape)
        f_i = (x, y) -> f(x, y)[i]
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
    @test f(a_val) == datum(result)

    # test backpropagation
    # this works for scalars, vectors, and matrix results.
    # note that scalars have scalar[1] = scalar
    for i=1:length(result)
        tape = Tape()
        a = makeGenValue(a_val, tape)
        f_i = (x) -> f(x)[i]
        result_i = f_i(a)
        backprop(result_i)
        @test isapprox(partial(a), finite_difference(f_i, a_val))
    end
end

@testset "automatic differentiation" begin

    a_scalar = 1.300181
    b_scalar = 1.45245
    a_vector = [1.4123, 4.3452]
    b_vector = [8.3453, 0.9913]
    a_row_vector = a_vector'
    b_row_vector = b_vector'
    a_matrix = [1.4123 4.3452 40.1; 10.123 0.9314 0.11] # 2 rows, 3 columns
    b_matrix = [4.13 33.123 5.32431; 4.314 5.1341 8.09] # 2 rows, 3 columns

    #@testset "unary plus" begin
#
        ## + scalar
        #adtest(+, a_scalar)
#
        ## + column vector
        #resulttest(+, a_vector)
        #for i=1:length(a_vector)
            #adtest((x) -> (+x)[i], a_vector)
        #end
#
        ## + row vector
        #resulttest(+, a_row_vector)
        #for i=1:length(a_row_vector)
            #adtest((x) -> (+x)[i], a_row_vector)
        #end
#
        ## + matrix
        #resulttest(+, a_matrix)
        #for i=1:length(a_matrix)
            #adtest((x) -> (+x)[i], a_matrix)
        #end
    #end
#
    #@testset "unary minus" begin
#
        ## - scalar
        #adtest(-, a_scalar)
#
        ## - column vector
        #resulttest(-, a_vector)
        #for i=1:length(a_vector)
            #adtest((x) -> (-x)[i], a_vector)
        #end
#
        ## - row vector
        #resulttest(-, a_row_vector)
        #for i=1:length(a_row_vector)
            #adtest((x) -> (-x)[i], a_row_vector)
        #end
#
        ## - matrix
        #resulttest(-, a_matrix)
        #for i=1:length(a_matrix)
            #adtest((x) -> (-x)[i], a_matrix)
        #end
    #end
#
    #@testset "add" begin
#
        ## scalar + scalar
        #adtest(+, a_scalar, b_scalar)
#
        ## scalar .+ scalar
        #adtest((x, y) -> broadcast(+, x, y), a_scalar, b_scalar)
#
        ## scalar + column vector
        #resulttest(+, a_scalar, a_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a + b)[i], a_scalar, a_vector)
        #end
#
        ## scalar + row vector
        #resulttest(+, a_scalar, a_row_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a + b)[i], a_scalar, a_row_vector)
        #end
#
        ## scalar .+ column vector
        #resulttest((x, y) -> broadcast(+, x, y), a_scalar, a_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a .+ b)[i], a_scalar, a_vector)
        #end
#
        ## scalar + row vector
        #resulttest((x, y) -> broadcast(+, x, y), a_scalar, a_row_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a .+ b)[i], a_scalar, a_row_vector)
        #end
#
        ## column vector + scalar
        #resulttest(+, a_vector, a_scalar)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a + b)[i], a_vector, a_scalar)
        #end
#
        ## row vector + scalar
        #resulttest(+, a_row_vector, a_scalar)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a + b)[i], a_row_vector, a_scalar)
        #end
#
        ## column vector .+ scalar
        #resulttest((x, y) -> broadcast(+, x, y), a_vector, a_scalar)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a .+ b)[i], a_vector, a_scalar)
        #end
#
        ## row vector .+ scalar
        #resulttest((x, y) -> broadcast(+, x, y), a_row_vector, a_scalar)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a .+ b)[i], a_row_vector, a_scalar)
        #end
#
        ## column vector + column vector
        #resulttest(+, a_vector, b_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a + b)[i], a_vector, b_vector)
        #end
#
        ## row vector + row vector
        #resulttest(+, a_row_vector, b_row_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a + b)[i], a_row_vector, b_row_vector)
        #end
#
        ## column vector .+ column vector
        #resulttest((x, y) -> broadcast(+, x, y), a_vector, b_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a .+ b)[i], a_vector, b_vector)
        #end
#
        ## row vector .+ row vector
        #resulttest((x, y) -> broadcast(+, x, y), a_row_vector, b_row_vector)
        #for i=1:length(a_scalar)
            #adtest((a, b) -> (a .+ b)[i], a_row_vector, b_row_vector)
        #end
#
        ## row vector .+ column vector (broadcast)
        ## TODO not implemented yet
#
        ## column vector .+ row vector (broadcast)
        ## TODO not implemented yet
#
        ## scalar + matrix
        ## TODO not implemented yet
#
        ## matrix + scalar
        ## TODO not implemented yet
#
        ## matrix .+ vector (broadcast)
        ## TODO not implemented yet
#
        ## vector .+ matrix (broadcast)
        ## TODO not implemented yet
#
        ## matrix + matrix
        ## TODO not implemented yet
    #end
#
    #@testset "subtract" begin
#
        ## scalar - scalar
        #adtest(-, a_scalar, b_scalar)
#
        ## scalar - column vector
        #resulttest(-, a_vector)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a - b)[i], a_scalar, a_vector)
        #end
#
        ## scalar - row vector
        #resulttest(-, a_scalar, a_row_vector)
        #for i=1:length(a_row_vector)
            #adtest((a, b) -> (a - b)[i], a_scalar, a_row_vector)
        #end
#
        ## column vector - scalar
        #resulttest(-, a_vector, a_scalar)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a - b)[i], a_vector, a_scalar)
        #end
#
        ## row vector - scalar
        #resulttest(-, a_row_vector, a_scalar)
        #for i=1:length(a_row_vector)
            #adtest((a, b) -> (a - b)[i], a_row_vector, a_scalar)
        #end
#
        ## column vector - column vector
        #resulttest(-, a_vector, b_vector)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a - b)[i], a_vector, b_vector)
        #end
#
        ## row vector - row vector
        #resulttest(-, a_row_vector, b_row_vector)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a - b)[i], a_row_vector, b_row_vector)
        #end
#
        ## row vector .- column vector (broadcast)
        ## TODO not implemented yet
#
        ## column vector .- row vector (broadcast)
        ## TODO not implemented yet
#
        ## scalar - matrix
        ## TODO not implemented yet
#
        ## matrix - scalar
        ## TODO not implemented yet
#
        ## matrix .- vector (broadcast)
        ## TODO not implemented yet
#
        ## vector .- matrix (broadcast)
        ## TODO not implemented yet
#
        ## matrix - matrix
        ## TODO not implemented yet
    #end
#
    #@testset "divide" begin
#
        ## scalar / scalar
        #adtest(/, a_scalar, b_scalar)
#
        ## scalar ./ scalar
        #adtest((x, y) -> broadcast(/, x, y), a_scalar, b_scalar)
#
        ## scalar ./ column vector
        #resulttest((x, y) -> broadcast(/, x, y), a_scalar, a_vector)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a ./ b)[i], a_scalar, a_vector)
        #end
#
        ## scalar ./ row vector
        #resulttest((x, y) -> broadcast(/, x, y), a_scalar, a_row_vector)
        #for i=1:length(a_row_vector)
            #adtest((a, b) -> (a ./ b)[i], a_scalar, a_row_vector)
        #end
#
        ## column vector / scalar
        #resulttest(/, a_vector, a_scalar)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a / b)[i], a_vector, a_scalar)
        #end
#
        ## row vector / scalar
        #resulttest(/, a_row_vector, a_scalar)
        #for i=1:length(a_row_vector)
            #adtest((a, b) -> (a / b)[i], a_row_vector, a_scalar)
        #end
#
        ## column vector ./ scalar
        #resulttest((x, y) -> broadcast(/, x, y), a_vector, a_scalar)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a ./ b)[i], a_vector, a_scalar)
        #end
#
        ## row vector ./ scalar
        #resulttest((x, y) -> broadcast(/, x, y), a_row_vector, a_scalar)
        #for i=1:length(a_row_vector)
            #adtest((a, b) -> (a ./ b)[i], a_row_vector, a_scalar)
        #end
#
        ## column vector ./ column vector
        #resulttest((x, y) -> broadcast(/, x, y), a_vector, b_vector)
        #for i=1:length(a_vector)
            #adtest((a, b) -> (a ./ b)[i], a_vector, b_vector)
        #end
#
        ## row vector ./ row vector
        #resulttest((x, y) -> broadcast(/, x, y), a_row_vector, b_row_vector)
        #for i=1:length(a_row_vector)
            #adtest((a, b) -> (a ./ b)[i], a_row_vector, b_row_vector)
        #end
        #
        ## row vector ./ column vector (broadcast)
        ## TODO not implemented yet
#
        ## column vector ./ row vector (broadcast)
        ## TODO not implemented yet
#
        ## scalar ./ matrix
        ## TODO not implemented yet
#
        ## matrix / scalar
        ## TODO not implemented yet
#
        ## matrix ./ vector (broadcast)
        ## TODO not implemented yet
#
        ## vector ./ matrix (broadcast)
        ## TODO not implemented yet
    #end

    @testset "elementwise-multiply" begin

        # scalar * scalar
        adtest(*, a_scalar, b_scalar)

        # scalar .* scalar
        adtest((a, b) -> a .* b, a_scalar, b_scalar)

        # scalar * column vector
        adtest((a, b) -> (a * b), a_scalar, a_vector)

        # scalar * row vector
        adtest((a, b) -> (a * b), a_scalar, a_row_vector)

        # scalar .* column vector
        adtest((a, b) -> (a .* b), a_scalar, a_vector)

        # scalar .* row vector
        adtest((a, b) -> (a .* b), a_scalar, a_row_vector)

        # column vector * scalar
        adtest((a, b) -> (a * b), a_vector, a_scalar)

        # row vector * scalar
        adtest((a, b) -> (a * b), a_row_vector, a_scalar)

        # column vector .* scalar
        adtest((a, b) -> (a .* b), a_vector, a_scalar)

        # row vector .* scalar
        adtest((a, b) -> (a .* b), a_row_vector, a_scalar)

        # column vector .* column vector
        adtest((a, b) -> (a .* b), a_vector, b_vector)

        # row vector .* row vector
        adtest((a, b) -> (a .* b), a_row_vector, b_row_vector)

        # column vector .* row vector
        adtest((a, b) -> (a .* b), a_vector, b_row_vector)

        # row vector .* column vector
        adtest((a, b) -> (a .* b), a_row_vector, b_vector)

        # scalar * matrix
        adtest((a, b) -> (a * b), a_scalar, a_matrix)

        # matrix * scalar
        adtest((a, b) -> (a * b), a_matrix, a_scalar)

        # matrix .* matrix
        adtest((a, b) -> (a .* b), a_matrix, a_scalar)

        # matrix .* vector (broadcast)
        # TODO not implemented yet

        # vector .* matrix (broadcast)
        # TODO not implemented yet

        # TODO test cartesian indexing into a matrix
    end

    @testset "matrix-multiply" begin

        # matrix * matrix, linear index into result matrix 
        adtest((a, b) -> (a * b), a_matrix', a_matrix)

        # matrix * column vector 
        adtest((a, b) -> (a * b), a_matrix', a_vector)

        # row vector * matrix
        # TODO

        # column vector * row vector (= column vector .* row vector)
        # TODO

        # row vector * column_vector 
        # TODO


    end

    @testset "transpose" begin

        # transpose scalar 
        adtest((a) -> a', a_scalar)

        # transpose matrix
        adtest((a) -> a', a_matrix)

        # transpose column vector 
        adtest((a) -> a', a_vector)

        # transpose row vector 
        adtest((a) -> a', a_row_vector)
    end

    @testset "exp" begin
        
        # exp(scalar)
        adtest(exp, a_scalar)

        # exp,(column vector)
        adtest((a) -> exp.(a), a_vector)

        # exp.(row vector)
        adtest((a) -> exp.(a), a_row_vector)

        # exp.(matrix)
        adtest((a) -> exp.(a), a_matrix)
    end

    @testset "log" begin
        
        # log(scalar)
        adtest(log, a_scalar)

        # log.(column vector)
        adtest((a) -> log.(a), a_vector)

        # log.(row vector)
        adtest((a) -> log.(a), a_row_vector)

        # log.(matrix)
        adtest((a) -> log.(a), a_matrix)
    end

    @testset "lgamma" begin
        
        # lgamma(scalar)
        adtest(lgamma, a_scalar)

        # lgamma.(column vector)
        adtest((a) -> lgamma.(a), a_vector)

        # lgamma.(row vector)
        adtest((a) -> lgamma.(a), a_row_vector)

        # lgamma.(matrix)
        adtest((a) -> lgamma.(a), a_matrix)
    end


#
    #@testset "sum" begin
        ## sum(scalar)
        #adtest(sum, a_scalar)
#
        ## sum(vector)
        #adtest(sum, a_vector)
#
        ## sum(matrix)
        #adtest(sum, a_matrix)
    #end

end
