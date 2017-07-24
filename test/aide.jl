@testset "AIDE" begin
    # crash test
    est = aide(normal, (0, 1), normal, (0, 1))

    # tiny extreme stochastic test
    # these distributioons are normalized, and kl divergence should be positive
    srand(1)
    est = mean([aide(normal, (0, 1), normal, (2, 1)) for i=1:100])
    @test est > 0
end
