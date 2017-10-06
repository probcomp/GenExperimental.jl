@testset "CRP" begin

    @testset "assignment book-keeping" begin
        crp = CRPState()

        c1 = new_table(crp)
        @test c1 == 1
        incorporate!(crp, c1)

        c2 = new_table(crp)
        @test c2 == 2
        incorporate!(crp, c2)

        incorporate!(crp, c2)
        @test counts(crp, c1) == 1
        @test counts(crp, c2) == 2
        @test new_table(crp) == c2 + 1
        @test crp.next_table == c2 + 2

        unincorporate!(crp, c1)
        @test !has_table(crp, c1)
        @test counts(crp, c2) == 2
        @test new_table(crp) != c2

        unincorporate!(crp, c2)
        @test counts(crp, c2) == 1

        unincorporate!(crp, c2)
        @test !has_table(crp, c2)

        # this is never decremented
        @test crp.next_table == c2 + 2

    end

    @testset "log_joint_probability" begin

        alpha = 0.25
        crp = CRPState()
        @test log_joint_probability(crp, alpha) == 0.0

        # [1]
        c1 = new_table(crp)
        incorporate!(crp, c1)
        @test log_joint_probability(crp, alpha) == 0.0
        
        # join the same table as 1: [1,2]
        incorporate!(crp, c1)
        actual = log_joint_probability(crp, alpha)
        expected = log(1. / (1. + alpha))
        @test isapprox(actual, expected)

        # join a new table: [1] [2]
        unincorporate!(crp, c1)
        c2 = new_table(crp)
        incorporate!(crp, c2)
        actual = log_joint_probability(crp, alpha)
        expected = log(alpha / (1. + alpha))
        @test isapprox(actual, expected)

        # join same table as 1: [1, 3] [2]
        incorporate!(crp, c1)
        actual = log_joint_probability(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)))
        @test isapprox(actual, expected)

        # join same table as 1 [1, 3, 4] [2]
        incorporate!(crp, c1)
        actual = log_joint_probability(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)) *
                       (2./(3.+alpha)))
        @test isapprox(actual, expected)
    end

    @testset "CRP generator" begin

        alpha = 0.25
        trace = CRPTrace(Int)

        # simulate without any outputs or conditions 
        n = 10
        (score, values) = simulate!(CRPGenerator(Int), (1:n, alpha, true), AddressTrie(), AddressTrie(), trace)
        for i=1:n
            @test haskey(trace, i)
            @test trace[i] == values[i]
        end
        @test !haskey(trace, n+1)
        @test score == 0.

        # regenerate with outputs and conditions
        trace = CRPTrace(Int)
        t1 = new_table(trace)
        trace[1] = t1
        t2 = new_table(trace)
        trace[2] = t2
        t3 = t2
        trace[3] = t3
        outputs = AddressTrie(1, 3)
        conditions = AddressTrie(2)
        (score, values) = regenerate!(CRPGenerator(Int), ([1, 2, 3], alpha, true), outputs, conditions, trace)
        @test trace[1] == t1
        @test trace[2] == t2
        @test trace[3] == t3
        @test values[1] == t1
        @test values[2] == t2
        @test values[3] == t3
        @test isapprox(score, log((alpha/(1 + alpha)) * (1 / (2 + alpha))))

        # simulate with outputs and conditions
        trace = CRPTrace(Int)
        t2 = new_table(trace)
        trace[2] = t2
        outputs = AddressTrie(1, 3)
        conditions = AddressTrie(2)
        (score, values) = simulate!(CRPGenerator(Int), ([1, 2, 3], alpha, true), outputs, conditions, trace)
        @test trace[2] == t2
        @test values[1] == trace[1]
        @test values[2] == t2
        @test values[3] == trace[3]
        if (trace[1] == t2) && (trace[3] == t2)
            # 1 and 3 are in the same cluster as 2
            @test isapprox(score, log((1/(alpha + 1)) * (2/(alpha + 2))))
        elseif (trace[1] == t2) && (trace[3] != t2)
            # 1 is in the same cluster as 2, but 3 is in a different cluster
            @test isapprox(score, log((1/(alpha + 1)) * (alpha/(alpha + 2))))
        elseif (trace[1] != t2) && (trace[3] == t2)
            # 1 is in a new cluster, and 3 is in the same cluster as 2
            @test isapprox(score, log((alpha/(alpha + 1)) * (1/(alpha + 2))))
        elseif (trace[1] != t2) && (trace[3] != t2) && (trace[3] != trace[1])
            # 1 is in a new cluster, and 3 is in a different cluster
            @test isapprox(score, log((alpha/(alpha + 1)) * (alpha/(alpha + 2))))
        elseif (trace[1] != t2) && (trace[3] == trace[1])
            # 1 is in a new cluster, and 3 is in the same cluster as 1
            @test isapprox(score, log((alpha/(alpha + 1)) * (1/(alpha + 2))))
        else
            error("unexpected state")
        end
    end

end
