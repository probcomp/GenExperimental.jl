import DataStructures.Stack
import DataStructures.PriorityQueue
import DataStructures.enqueue!
import DataStructures.dequeue!

mutable struct DAG
    next::Int
    priorities::Dict{String,Int}
    parents::Dict{String,Vector{String}}
end

DAG() = DAG(0, Dict{String,Int}(), Dict{String,Vector{String}}())

function add_node!(dag::DAG, addr::String, parents::Vector{String})
    if haskey(dag.priorities, addr)
        error("Node $addr already exists")
    end
    for parent_addr in parents
        if !haskey(dag.priorities, parent_addr)
            error("Node $parent_addr does not exist")
        end
    end
    dag.priorities[addr] = dag.next
    dag.parents[addr] = parents
    dag.next += 1
end

function execution_order(dag::DAG, request::Set{String})
    pq = PriorityQueue(String, Int, Base.Order.Reverse)
    for addr in request
        if !haskey(dag.priorities, addr)
            error("Node $addr does not exist")
        end
        enqueue!(pq, addr, dag.priorities[addr])
    end
    order = Stack(String)
    while !isempty(pq)

        # return highest-priority (latest in the ordering)
        addr = dequeue!(pq) 

        for parent_addr in dag.parents[addr]
            if !haskey(pq, parent_addr) # TODO also check if its a condition
                enqueue!(pq, parent_addr, dag.priorities[parent_addr])
            end
        end

        push!(order, addr)
    end
    return order
end


using Base.Test

@testset "Markov chain" begin

    dag = DAG()
    add_node!(dag, "x-1", String[])
    for i=2:10
        add_node!(dag, "x-$i", String["x-$(i-1)"])
    end
    order = execution_order(dag, Set(["x-5"]))
    for i=1:5
        @test pop!(order) == "x-$i"
    end
    @test isempty(order)
end

@testset "Complicated" begin

    dag = DAG()
    add_node!(dag, "a", String[])
    add_node!(dag, "b", String[])
    add_node!(dag, "c", String["a"])
    add_node!(dag, "d", String["b"])
    add_node!(dag, "e", String["a", "d"])
    add_node!(dag, "f", String["b"])
    add_node!(dag, "g", String["f"])
    add_node!(dag, "h", String["g", "c"])

    ordering = Dict{String,Int}()
    for (i, addr) in enumerate(execution_order(dag, Set(["h"])))
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
    @test isempty(execution_order(dag, Set{String}()))

    # a query for a node with no parents
    order = execution_order(dag, Set(["a"]))
    @test pop!(order) == "a"
    @test isempty(order)
    
end
