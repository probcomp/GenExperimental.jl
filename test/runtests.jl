using Gen
using Distributions
using Base.Test

@testset "automatic differentiation" begin

    @testset "basic operations" begin

        # TODO test using finite differences instead of this..

        srand(1)
        a_val, b_val = rand(2)
    
        # binary plus
        tape = Tape()
        a = GenFloat(a_val, tape)
        b = GenFloat(b_val, tape)
        c = a + b
        backprop(c)
        @test isapprox(partial(a), 1.0)
        @test isapprox(partial(b), 1.0)
        @test concrete(a) + concrete(b) == concrete(a + b)

        # unary plus
        tape = Tape()
        a = GenFloat(a_val, tape)
        c = +a
        backprop(c)
        @test isapprox(partial(a), 1.0)
        @test concrete(+a) == concrete(a)
    
        # binary minus
        tape = Tape()
        a = GenFloat(a_val, tape)
        b = GenFloat(b_val, tape)
        c = a - b
        backprop(c)
        @test isapprox(partial(a), 1.0)
        @test isapprox(partial(b), -1.0)
        @test concrete(a - b) == concrete(a) - concrete(b)
    
        # unary minus
        tape = Tape()
        a = GenFloat(a_val, tape)
        c = -a
        backprop(c)
        @test isapprox(partial(a), -1.0)
        @test concrete(-a) == -concrete(a)

        # times
        tape = Tape()
        a = GenFloat(a_val, tape)
        b = GenFloat(b_val, tape)
        c = a * b
        backprop(c)
        @test isapprox(partial(a), b_val)
        @test isapprox(partial(b), a_val)
        @test concrete(a * b) == concrete(a) * concrete(b)
    
        # divide
        tape = Tape()
        a = GenFloat(a_val, tape)
        b = GenFloat(b_val, tape)
        c = a / b
        backprop(c)
        @test isapprox(partial(a), 1.0 / b_val)
        @test isapprox(partial(b), -a_val/(b_val * b_val))
        @test concrete(a / b) == concrete(a) / concrete(b)
    
        # log 
        tape = Tape()
        a = GenFloat(a_val, tape)
        c = log(a)
        backprop(c)
        @test isapprox(partial(a), 1.0 / a_val)
        @test concrete(log(a)) == log(concrete(a))
    
        # exp 
        tape = Tape()
        a = GenFloat(a_val, tape)
        c = exp(a)
        backprop(c)
        @test isapprox(partial(a), exp(a_val))
        @test concrete(exp(a)) == exp(concrete(a))

        # exp 
        tape = Tape()
        a = GenFloat(a_val, tape)
        c = lgamma(a)
        backprop(c)
        @test isapprox(partial(a), digamma(a_val))
        @test concrete(lgamma(a)) == lgamma(concrete(a))

    end

    @testset "simple expressions" begin
        
        srand(1)
        x_val, y_val, z_val = rand(3)

        # w = x + y + z
        tape = Tape()
        x = GenFloat(x_val, tape)
        y = GenFloat(y_val, tape)
        z = GenFloat(z_val, tape)
        w = x + y - z
        backprop(w)
        @test isapprox(partial(x), 1.0)
        @test isapprox(partial(y), 1.0)
        @test isapprox(partial(z), -1.0)

    end

    @testset "sigmoid function" begin
        srand(1)
        sig = (x) -> Float64(1.0) / (Float64(1.0) + exp(-x))
        tape = Tape()
        x = GenFloat(rand(), tape)
        y = sig(x)
        backprop(y)
        @test isapprox(partial(x), concrete(y * (1.0 - y)))
        
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

end

@testset "primitives" begin

    # bernoulli
    p = 0.1
    @test isapprox(flip_regenerate(true, p), logpdf(Bernoulli(p), true))

    # normal
    x = 0.1
    mu = 0.2
    std = 0.3
    @test isapprox(normal_regenerate(x, mu, std), logpdf(Normal(mu, std), x))

    # gamma
    x = 0.1
    k = 0.2
    s = 0.3
    @test isapprox(gamma_regenerate(x, k, s), logpdf(Gamma(k, s), x))

end

nothing
