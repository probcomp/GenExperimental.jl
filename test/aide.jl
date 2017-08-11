@testset "AIDE between atomic generators" begin

    srand(1)

    # small stochastic test
    expected_est = 0.5 * (2.^2) # (mu1 - mu2)^2 / (2*sigma^2) for equal-variance
    est = mean([aide(normal, (0, 1), normal, (2, 1)) for i=1:1000])
    @test isapprox(est, expected_est, atol=0.1)

    est = mean([aide(normal, (0, 1), normal, (0, 1)) for i=1:1000])
    @test isapprox(est, 0, atol=0.1)
end

@testset "AIDE between non-atomic generators" begin

    @program p() begin
        x = normal(0, 1)
        a = @g(normal(x, 1), "a")
    end

    @program q() begin
        y = normal(1, 2)
        b = @g(normal(y, 1), "b")
    end

    # small stochastic test
    # due to bias, estimates will be positive in both cases
    # TODO: compute expected value in both cases
    srand(1)
    est = mean([aide(p, (), q, (), Dict(["a" => "b"])) for i=1:1000])
    @test est > 0

    est = mean([aide(p, (), p, (), Dict(["a" => "a"])) for i=1:1000])
    @test est > 0
end
