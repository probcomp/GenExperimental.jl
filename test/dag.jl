@testset "Markov chain" begin

    dag = DAG(String)
    add_node!(dag, "x-1", String[])
    for i=2:10
        add_node!(dag, "x-$i", String["x-$(i-1)"])
    end

    # without a condition
    order = execution_order(dag, Set(["x-5"]), Set{String}())
    for i=1:5
        @test pop!(order) == "x-$i"
    end
    @test isempty(order)

    # with a condition
    order = execution_order(dag, Set(["x-5"]), Set(["x-2"]))
    for i=3:5
        @test pop!(order) == "x-$i"
    end
    @test isempty(order)

end

@testset "Complicated" begin

    dag = DAG(String)
    add_node!(dag, "a", String[])
    add_node!(dag, "b", String[])
    add_node!(dag, "c", String["a"])
    add_node!(dag, "d", String["b"])
    add_node!(dag, "e", String["a", "d"])
    add_node!(dag, "f", String["b"])
    add_node!(dag, "g", String["f"])
    add_node!(dag, "h", String["g", "c"])

    ordering = Dict{String,Int}()
    for (i, addr) in enumerate(execution_order(dag, Set(["h"]), Set{String}()))
        ordering[addr] = i
    end

    @test ordering["a"] < ordering["c"]
    @test ordering["b"] < ordering["f"]
    @test ordering["f"] < ordering["g"]
    @test ordering["g"] < ordering["h"]

    # not necessary to visit these
    @test !haskey(ordering, "e")
    @test !haskey(ordering, "d")

    # an empty query
    @test isempty(execution_order(dag, Set{String}(), Set{String}()))

    # a query for a node with no parents
    order = execution_order(dag, Set(["a"]), Set{String}())
    @test pop!(order) == "a"
    @test isempty(order)

    # with a condition that causes a to not be visited
    order = execution_order(dag, Set(["h"]), Set(["c"]))
    @test pop!(order) == "b"
    @test pop!(order) == "f"
    @test pop!(order) == "g"
    @test pop!(order) == "h"
    @test isempty(order)
    
end
