
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


@testset "automatic differentiation" begin

    @testset "basic operations" begin

        srand(1)
        a_val, b_val = rand(2)
    
        # binary plus
        tape = Tape()
        a = GenScalar(a_val, tape)
        b = GenScalar(b_val, tape)
        c = a + b
        backprop(c)
        @test isapprox(partial(a), finite_difference((x) -> x + b_val, a_val))
        @test isapprox(partial(b), finite_difference((x) -> a_val + x, b_val))
        @test concrete(a) + concrete(b) == concrete(a + b)

        # unary plus
        tape = Tape()
        a = GenScalar(a_val, tape)
        c = +a
        backprop(c)
        @test isapprox(partial(a), finite_difference((x) -> +x, a_val))
        @test concrete(+a) == concrete(a)
    
        # binary minus
        tape = Tape()
        a = GenScalar(a_val, tape)
        b = GenScalar(b_val, tape)
        c = a - b
        backprop(c)
        @test isapprox(partial(a), finite_difference((x) -> x - b_val, a_val))
        @test isapprox(partial(b), finite_difference((x) -> a_val - x, b_val))
        @test concrete(a - b) == concrete(a) - concrete(b)
    
        # unary minus
        tape = Tape()
        a = GenScalar(a_val, tape)
        c = -a
        backprop(c)
        @test isapprox(partial(a), finite_difference((x) -> -x, a_val))
        @test concrete(-a) == -concrete(a)

        # times
        tape = Tape()
        a = GenScalar(a_val, tape)
        b = GenScalar(b_val, tape)
        c = a * b
        backprop(c)
        @test isapprox(partial(a), finite_difference((x) -> x * b_val, a_val))
        @test isapprox(partial(b), finite_difference((x) -> a_val * x, b_val))
        @test concrete(a * b) == concrete(a) * concrete(b)
    
        # divide
        tape = Tape()
        a = GenScalar(a_val, tape)
        b = GenScalar(b_val, tape)
        c = a / b
        backprop(c)
        @test isapprox(partial(a), finite_difference((x) -> x / b_val, a_val))
        @test isapprox(partial(b), finite_difference((x) -> a_val / x, b_val))
        @test concrete(a / b) == concrete(a) / concrete(b)
    
        # log 
        tape = Tape()
        a = GenScalar(a_val, tape)
        c = log(a)
        backprop(c)
        @test isapprox(partial(a), finite_difference(log, a_val))
        @test concrete(log(a)) == log(concrete(a))
    
        # exp 
        tape = Tape()
        a = GenScalar(a_val, tape)
        c = exp(a)
        backprop(c)
        @test isapprox(partial(a), finite_difference(exp, a_val))
        @test concrete(exp(a)) == exp(concrete(a))

        # lgamma
        tape = Tape()
        a = GenScalar(a_val, tape)
        c = lgamma(a)
        backprop(c)
        @test isapprox(partial(a), finite_difference(lgamma, a_val))
        @test concrete(lgamma(a)) == lgamma(concrete(a))

    end

    @testset "simple expressions" begin
        
        srand(1)
        x_val, y_val, z_val = rand(3)

        # w = x + y + z
        tape = Tape()
        x = GenScalar(x_val, tape)
        y = GenScalar(y_val, tape)
        z = GenScalar(z_val, tape)
        w = x + y - z
        backprop(w)
        @test isapprox(partial(x), finite_difference((x) -> (x + y_val - z_val), x_val))
        @test isapprox(partial(y), finite_difference((y) -> (x_val + y - z_val), y_val))
        @test isapprox(partial(z), finite_difference((z) -> (x_val + y_val - z), z_val))

    end

    @testset "sigmoid function" begin
        srand(1)
        sig = (x) -> Float64(1.0) / (Float64(1.0) + exp(-x))
        tape = Tape()
        x_val = rand()
        x = GenScalar(x_val, tape)
        y = sig(x)
        backprop(y)
        @test isapprox(partial(x), finite_difference(sig, x_val))
        
    end

    @testset "operations involving matrices" begin

        # getindex
        tape = Tape()
        a_val = rand(2, 3)
        a = GenMatrix(a_val, tape)
        b = a[1, 2]
        backprop(b)
        @test isapprox(partial(a), [0. 1. 0.; 0. 0. 0.])

        # multiply and sum
        tape = Tape()
        a_val = rand(2, 3)
        b_val = rand(3, 2)
        a = GenMatrix(a_val, tape)
        b = GenMatrix(b_val, tape)
        c = sum(a * b)
        backprop(c)
        # a * b
        # (a * b)[1,1] = a[1,1]*b[1,1] + a[1,2]*b[2,1] + a[1,3]*b[3,1]
        # (a * b)[1,2] = a[1,1]*b[1,2] + a[1,2]*b[2,2] + a[1,3]*b[3,2]
        # (a * b)[2,1] = a[2,1]*b[1,1] + a[2,2]*b[2,1] + a[2,3]*b[3,1]
        # (a * b)[2,2] = a[2,1]*b[1,2] + a[2,2]*b[2,2] + a[2,3]*b[3,2]

        # deriv a[1,1] = b[1,1] + b[1,2]
        # deriv a[1,2] = b[2,1] + b[2,2]
        # deriv a[1,3] = b[3,1] + b[3,2]
        # deriv a[2,1] = b[1,1] + b[1,2]
        # deriv a[2,2] = b[2,1] + b[2,2]
        # deriv a[2,3] = b[3,1] + b[3,2]
        row = sum(b_val, 2)'
        @test isapprox(partial(a), vcat(row, row))
        
        # deriv b[1,1] = a[1,1] + a[2,1]
        # deriv b[1,2] = a[1,1] + a[2,1]
        # deriv b[2,1] = a[1,2] + a[2,2]
        # deriv b[2,2] = a[1,2] + a[2,2]
        # deriv b[3,1] = a[1,3] + a[2,3]
        # deriv b[3,2] = a[1,3] + a[2,3]
        col = sum(a_val, 1)'
        @test isapprox(partial(b), hcat(col, col))

        # elementwise op
        tape = Tape()
        a_val = rand(2, 3)
        a = GenMatrix(a_val, tape)
        b = exp(a)
        backprop(sum(b))
        @test isapprox(partial(a), exp(a_val))
    
    end

    @testset "operations involving vectors" begin

        # getindex
        tape = Tape()
        a_val = rand(2)
        a = GenVector(a_val, tape)
        f = (a) -> a[2]
        b = f(a)
        backprop(b)
        @test isapprox(partial(a), finite_difference(f, a_val))
        @test concrete(b) == f(a_val)

        # transpose
        tape = Tape()
        a_val = rand(2)
        a = GenVector(a_val, tape)
        f = (a) -> a'
        b = f(a)
        @test typeof(b) <: GenMatrix
        @test concrete(b) == f(a_val)
        backprop(b[1, 2])
        @test isapprox(partial(a), finite_difference((a) -> (a')[1,2], a_val))

        # matrix-vector multiply
        tape = Tape()
        a_val = rand(2)
        b_val = rand(3, 2)
        a = GenVector(a_val, tape)
        b = GenMatrix(b_val, tape)
        c = b * a
        @test typeof(c) <: GenVector
        @test concrete(c) == b_val * a_val
        backprop(c[1])
        # b * a
        # (b * a)[1] = b[1,1]*a[1] + b[1,2]*a[2] 
        # deriv a[1] = b[1,1]
        # deriv a[2] = b[1,2]
        @test isapprox(partial(a), finite_difference((a) -> (b_val * a)[1], a_val))
        
        # deriv b[1,1] = a[1]
        # deriv b[1,2] = a[2]
        # deriv b[2,1] = 0.
        # deriv b[2,2] = 0.
        # deriv b[3,1] = 0.
        # deriv b[3,2] = 0.
        @test isapprox(partial(b), finite_difference((b) -> (b * a_val)[1], b_val))

        # elementwise op
        tape = Tape()
        a_val = rand(2)
        a = GenVector(a_val, tape)
        b = exp(a)
        backprop(sum(b))
        @test isapprox(partial(a), finite_difference((a) -> sum(exp(a)), a_val))
    
    end


end


