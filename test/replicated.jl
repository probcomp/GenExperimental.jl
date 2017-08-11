@testset "replicated generator" begin

    run_count = 0

    @program inner_generator() begin
            run_count += 1
            mu = @g(normal(0, 1), "mu")
            @g(normal(mu, 1), "x")
        end

    generator = replicated(inner_generator, 4)
    t = DictTrace()

    (score, val) = simulate!(generator, (), AddressTrie(), AddressTrie(), t)
    @test val == t["x"]
    @test score == 0.
    @test run_count == 4

    run_count = 0
    (score, val) = regenerate!(generator, (), AddressTrie(), AddressTrie(), t)
    @test val == t["x"]
    @test score == 0.
    @test run_count == 4

    run_count = 0
    (score, val) = simulate!(generator, (), AddressTrie("x"), AddressTrie(), t)
    @test val == t["x"]
    @test score != 0.
    @test run_count == 4

    run_count = 0
    t["x"] = 0.123
    (score, val) = regenerate!(generator, (), AddressTrie("x"), AddressTrie(), t)
    @test val == t["x"]
    @test val == 0.123
    @test score != 0.
    @test run_count == 4
end
