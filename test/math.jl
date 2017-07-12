@testset "logsumexp" begin
    a = [1.1, 2.2, 3.3]
    @test isapprox(logsumexp(a), log(sum(exp.(a))))
end
