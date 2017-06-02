include("ad.jl")
include("trace.jl")
using Base.Test

@testset "automatic differentiation" begin

    @testset "basic operations" begin
        srand(1)
        a_val, b_val = rand(2)
    
        # binary plus
        tape = Tape()
        a = GenNum(a_val, tape)
        b = GenNum(b_val, tape)
        c = a + b
        backprop(c)
        @test isapprox(partial(a), 1.0)
        @test isapprox(partial(b), 1.0)
        @test concrete(a) + concrete(b) == concrete(a + b)
    
        # unary plus
        tape = Tape()
        a = GenNum(a_val, tape)
        c = +a
        backprop(c)
        @test isapprox(partial(a), 1.0)
        @test concrete(+a) == concrete(a)
    
        # binary minus
        tape = Tape()
        a = GenNum(a_val, tape)
        b = GenNum(b_val, tape)
        c = a - b
        backprop(c)
        @test isapprox(partial(a), 1.0)
        @test isapprox(partial(b), -1.0)
        @test concrete(a - b) == concrete(a) - concrete(b)
    
        # unary minus
        tape = Tape()
        a = GenNum(a_val, tape)
        c = -a
        backprop(c)
        @test isapprox(partial(a), -1.0)
        @test concrete(-a) == -concrete(a)

        # times
        tape = Tape()
        a = GenNum(a_val, tape)
        b = GenNum(b_val, tape)
        c = a * b
        backprop(c)
        @test isapprox(partial(a), b_val)
        @test isapprox(partial(b), a_val)
        @test concrete(a * b) == concrete(a) * concrete(b)
    
        # divide
        tape = Tape()
        a = GenNum(a_val, tape)
        b = GenNum(b_val, tape)
        c = a / b
        backprop(c)
        @test isapprox(partial(a), 1.0 / b_val)
        @test isapprox(partial(b), -a_val/(b_val * b_val))
        @test concrete(a / b) == concrete(a) / concrete(b)
    
        # log 
        tape = Tape()
        a = GenNum(a_val, tape)
        c = log(a)
        backprop(c)
        @test isapprox(partial(a), 1.0 / a_val)
        @test concrete(log(a)) == log(concrete(a))
    
        # exp 
        tape = Tape()
        a = GenNum(a_val, tape)
        c = exp(a)
        backprop(c)
        @test isapprox(partial(a), exp(a_val))
        @test concrete(exp(a)) == exp(concrete(a))
    end

    @testset "simple expressions" begin
        
        srand(1)
        x_val, y_val, z_val = rand(3)

        # w = x + y + z
        tape = Tape()
        x = GenNum(x_val, tape)
        y = GenNum(y_val, tape)
        z = GenNum(z_val, tape)
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
        x = GenNum(rand(), tape)
        y = sig(x)
        backprop(y)
        println("x: $(concrete(x))")
        println("y: $(concrete(y))")
        println("actual: $(partial(x))")
        println("expected: $(concrete(y * (1.0 - y)))")
        @test isapprox(partial(x), concrete(y * (1.0 - y)))
        
    end

end
