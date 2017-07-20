@testset "CRP" begin

    @testset "assignment book-keeping" begin
        crp = CRPState()

        c1 = new_cluster(crp)
        @test c1 == 1
        incorporate!(crp, c1)

        c2 = new_cluster(crp)
        @test c2 == 2
        incorporate!(crp, c2)

        incorporate!(crp, c2)
        @test counts(crp, c1) == 1
        @test counts(crp, c2) == 2
        @test new_cluster(crp) == c2 + 1
        @test crp.next_cluster == c2 + 2

        unincorporate!(crp, c1)
        @test !has_cluster(crp, c1)
        @test counts(crp, c2) == 2
        @test new_cluster(crp) != c2

        unincorporate!(crp, c2)
        @test counts(crp, c2) == 1

        unincorporate!(crp, c2)
        @test !has_cluster(crp, c2)

        # this is never decremented
        @test crp.next_cluster == c2 + 2

    end

    @testset "logpdf" begin

        alpha = 0.25
        crp = CRPState()
        @test logpdf(crp, alpha) == 0.0

        # [1]
        c1 = new_cluster(crp)
        incorporate!(crp, c1)
        @test logpdf(crp, alpha) == 0.0
        
        # join the same table as 1: [1,2]
        incorporate!(crp, c1)
        actual = logpdf(crp, alpha)
        expected = log(1. / (1. + alpha))
        @test isapprox(actual, expected)

        # join a new table: [1] [2]
        unincorporate!(crp, c1)
        c2 = new_cluster(crp)
        incorporate!(crp, c2)
        actual = logpdf(crp, alpha)
        expected = log(alpha / (1. + alpha))
        @test isapprox(actual, expected)

        # join same table as 1: [1, 3] [2]
        incorporate!(crp, c1)
        actual = logpdf(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)))
        @test isapprox(actual, expected)

        # join same table as 1 [1, 3, 4] [2]
        incorporate!(crp, c1)
        actual = logpdf(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)) *
                       (2./(3.+alpha)))
        @test isapprox(actual, expected)
    end

    @testset "draw from crp" begin

        alpha = 0.25

        # [1], [2, 3]
        crp = CRPState()
        c1 = new_cluster(crp)
        incorporate!(crp, c1)
        c2 = new_cluster(crp)
        incorporate!(crp, c2)
        incorporate!(crp, c2)
        c3 = new_cluster(crp)

        # log(prob([1], [2, 3]))
        log_prob_before = logpdf(crp, alpha)

        # test probability of selecting new cluster
        # c3: [4] | c1: [1], c2: [2, 3]
        actual = logpdf(CRPDraw(), c3, crp, alpha)
        incorporate!(crp, c3)
        expected_joint = logpdf(crp, alpha) - log_prob_before
        unincorporate!(crp, c3)
        expected_manual = log(alpha / (alpha + 1 + 2))
        @test isapprox(actual, expected_manual)
        @test isapprox(actual, expected_joint)

        # test probability of joining previous cluster
        # c1: [1, 4] | c1: [1], c2: [2, 3]
        actual = logpdf(CRPDraw(), c1, crp, alpha)
        incorporate!(crp, c1)
        expected_joint = logpdf(crp, alpha) - log_prob_before
        unincorporate!(crp, c1)
        expected_manual = log(1 / (alpha + 1 + 2))
        @test isapprox(actual, expected_manual)
        @test isapprox(actual, expected_joint)

    end

    @testset "CRP generator" begin

        alpha = 0.25
        trace = CRPJointTrace()

        # test that the correct values were generated and that the 
        # the constrained values were not modified
        n = 10
        constrain!(trace, 5, new_cluster(trace))
        a5 = value(trace, 5)
        constrain!(trace, 9, a5)
        a9 = value(trace, 9)
        (score, values) = generate!(CRPJointGenerator(), (Set(1:n), alpha), trace)
        for i=1:n
            @test haskey(trace, i)
            @test value(trace, i) == values[i]
        end
        @test !haskey(trace, n+1)
        @test value(trace, 5) == a5
        @test value(trace, 9) == a9

        # the score is for the constrained values only
        expected_score = log(1. * (1. / (1. + alpha)))
        @test isapprox(score, expected_score)

        # generate again and check that the score hasn't changed
        # (this checks that the sufficient statistics were correctly reverted)
        (score, values) = generate!(CRPJointGenerator(), (Set(1:n), alpha), trace)
        @test isapprox(score, expected_score)
        @test values[5] == a5
        @test values[9] == a9

        # the CRP_NEW_CLUSTER constraint feature
        trace = CRPJointTrace()
        c1 =  new_cluster(trace)
        constrain!(trace, 1, CRP_NEW_CLUSTER)
        @test trace[1] == c1
        c2 =  new_cluster(trace)
        constrain!(trace, 2, CRP_NEW_CLUSTER)
        @test trace[2] == c2
    end

end
