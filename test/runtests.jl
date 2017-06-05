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

        # exp 
        tape = Tape()
        a = GenNum(a_val, tape)
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
        @test isapprox(partial(x), concrete(y * (1.0 - y)))
        
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

@testset "trace macros" begin
    
    # test that @constrain and @unconstrain work with @in
    trace = Trace()
    @in trace begin
        @constrain "a" 1.0
        @constrain "b" 2.0
        @unconstrain "a"
        @constrain "a" 1.5
    end
    @test trace.vals == Dict([("a", 1.5), ("b", 2.0)])

    # test that code is evaluated in the right scope
    trace = Trace()
    i = 1
    j = 2
    v = 2.0
    w = 3.0
    @in trace begin
        j = 3
        w = 4.0
        @constrain("$i", v)
        @constrain("$j", w)
    end
    @test trace.vals == Dict([("1", 2.0), ("3", 4.0)])
    
end

nothing
