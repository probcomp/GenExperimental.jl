@testset "CRP" begin

    @testset "assignment book-keeping" begin
        crp = CRPState()

        c1 = next_new_cluster(crp)
        @test c1 == 1
        incorporate!(crp, c1)

        c2 = next_new_cluster(crp)
        @test c2 == 2
        incorporate!(crp, c2)

        incorporate!(crp, c2)
        @test counts(crp, c1) == 1
        @test counts(crp, c2) == 2
        @test next_new_cluster(crp) == c2 + 1
        @test crp.next_cluster == c2 + 2

        unincorporate!(crp, c1)
        @test !has_cluster(crp, c1)
        @test counts(crp, c2) == 2
        @test next_new_cluster(crp) == c1

        unincorporate!(crp, c2)
        @test counts(crp, c2) == 1

        unincorporate!(crp, c2)
        @test !has_cluster(crp, c2)
        @test next_new_cluster(crp) == c2

        # this is never decremented
        @test crp.next_cluster == c2 + 2

    end

    @testset "joint_log_probability" begin

        alpha = 0.25
        crp = CRPState()
        @test joint_log_probability(crp, alpha) == 0.0

        # [1]
        c1 = next_new_cluster(crp)
        incorporate!(crp, c1)
        @test joint_log_probability(crp, alpha) == 0.0
        
        # join the same table as 1: [1,2]
        incorporate!(crp, c1)
        actual = joint_log_probability(crp, alpha)
        expected = log(1. / (1. + alpha))
        @test isapprox(actual, expected)

        # join a new table: [1] [2]
        unincorporate!(crp, c1)
        c2 = next_new_cluster(crp)
        incorporate!(crp, c2)
        actual = joint_log_probability(crp, alpha)
        expected = log(alpha / (1. + alpha))
        @test isapprox(actual, expected)

        # join same table as 1: [1, 3] [2]
        incorporate!(crp, c1)
        actual = joint_log_probability(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)))
        @test isapprox(actual, expected)

        # join same table as 1 [1, 3, 4] [2]
        incorporate!(crp, c1)
        actual = joint_log_probability(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)) *
                       (2./(3.+alpha)))
        @test isapprox(actual, expected)
    end

    @testset "draw from crp" begin

        alpha = 0.25

        # [1], [2, 3]
        crp = CRPState()
        c1 = next_new_cluster(crp)
        incorporate!(crp, c1)
        c2 = next_new_cluster(crp)
        incorporate!(crp, c2)
        incorporate!(crp, c2)
        c3 = next_new_cluster(crp)

        # log(prob([1], [2, 3]))
        log_prob_before = joint_log_probability(crp, alpha)

        # test probability of selecting new cluster
        # c3: [4] | c1: [1], c2: [2, 3]
        actual = logpdf(draw_crp, c3, crp, alpha)
        incorporate!(crp, c3)
        expected_joint = joint_log_probability(crp, alpha) - log_prob_before
        unincorporate!(crp, c3)
        expected_manual = log(alpha / (alpha + 1 + 2))
        @test isapprox(actual, expected_manual)
        @test isapprox(actual, expected_joint)

        # test probability of joining previous cluster
        # c1: [1, 4] | c1: [1], c2: [2, 3]
        actual = logpdf(draw_crp, c1, crp, alpha)
        incorporate!(crp, c1)
        expected_joint = joint_log_probability(crp, alpha) - log_prob_before
        unincorporate!(crp, c1)
        expected_manual = log(1 / (alpha + 1 + 2))
        @test isapprox(actual, expected_manual)
        @test isapprox(actual, expected_joint)

    end

end
