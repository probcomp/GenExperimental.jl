import DataStructures.Stack
import DataStructures.PriorityQueue
import DataStructures.enqueue!
import DataStructures.dequeue!

"""
`T` is the type of the address (e.g. String)
"""
mutable struct DAG{T}
    next::Int
    priorities::Dict{T,Int}
    parents::Dict{T,Vector{T}}
end

DAG(::Type{T}) where {T} = DAG(0, Dict{T,Int}(), Dict{T,Vector{T}}())

function add_node!(dag::DAG{T}, addr::T, parents::Vector{T}) where {T}
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

function execution_order(dag::DAG{T}, outputs, conditions) where {T}
    pq = PriorityQueue{T,Int}(Base.Order.Reverse)
    for addr in outputs 
        if !haskey(dag.priorities, addr)
            error("Node $addr does not exist")
        end
        if addr in conditions
            error("Node $addr cannot be a condition and a output")
        end
        enqueue!(pq, addr, dag.priorities[addr])
    end
    for addr in conditions
        if !haskey(dag.priorities, addr)
            error("Node $addr does not exist")
        end
    end
    order = Stack(String)
    while !isempty(pq)

        # return highest-priority (latest in the ordering)
        addr = dequeue!(pq) 

        for parent_addr in dag.parents[addr]
            if !haskey(pq, parent_addr) && !(parent_addr in conditions)
                enqueue!(pq, parent_addr, dag.priorities[parent_addr])
            end
        end

        push!(order, addr)
    end
    return order
end

export DAG
export add_node!
export execution_order
