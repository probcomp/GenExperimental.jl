abstract type AbstractOperator end
abstract type GenValue{T} end

mutable struct Tape
    nums::Array{GenValue, 1}
    function Tape()
        new(Array{GenValue, 1}())
    end
end
nums(t::Tape) = t.nums


# scalar

mutable struct GenScalar{S<:Real, T<:AbstractOperator} <: GenValue{T}
    datum::S
    adj::Float64
    tape::Tape
    tapeIdx::Int
    op::T
end

function GenScalar{S<:Real, T<:AbstractOperator}(datum::S, tape::Tape, op::T)
    ns = nums(tape)
    num = GenScalar{S,T}(datum, 0., tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

GenScalar{S}(datum::S, tape::Tape) = GenScalar(datum, tape, Input())

# vector

mutable struct GenVector{S<:Real, T<:AbstractOperator} <: GenValue{T}
    # NOTE: the datum is intended to be immutable (there should be no setindex operator)
    datum::Vector{S}
    adj::Vector{Float64}
    tape::Tape
    tapeIdx::Int
    op::T
end

function GenVector{S<:Real, T<:AbstractOperator}(datum::Vector{S}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenVector{S,T}(datum, zeros(Float64, size(datum)), tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

GenVector{S}(datum::S, tape::Tape) = GenVector(datum, tape, Input())

# GenValueconstructors

GenValue(datum::Real, tape::Tape) = GenScalar(datum, tape, Input())
GenValue(datum::Real, tape::Tape, op::AbstractOperator) = GenScalar(datum, tape, op)
GenValue(datum::Vector{U}, tape::Tape) where U <: Real = GenVector(datum, tape, Input())
GenValue(datum::Vector{U}, tape::Tape, op::AbstractOperator) where U <: Real = GenVector(datum, tape, op)
GenValue(datum::Vector{U}, tape::Tape, op::AbstractOperator) where U <: Real = GenVector(datum, tape, op)


macro generate_binary_operators(op, opType)
    eval(quote
            struct $(opType){T,U} <: AbstractOperator
                left::T
                right::U
            end

            function ($op)(l::L, r::R) where L <: GenValue where R <: GenValue
                GenValue(($op)(datum(l), datum(r)), l.tape, ($opType)(l, r))
            end
            
            function ($op)(ldatum::L, r::R) where L where R <: GenValue
                l = GenValue(ldatum, r.tape)
                GenValue((/)(ldatum, datum(r)), l.tape, ($opType)(l, r))
            end
            
            function ($op)(l::L, rdatum::R) where L <: GenValue where R
                r = GenValue(rdatum, l.tape)
                GenValue(($op)(datum(l), rdatum), l.tape, ($opType)(l, r))
            end
            
            function broadcast(::typeof($op), l::L, r::R) where L <: GenValue where R <: GenValue
                GenValue(broadcast($op, datum(l), datum(r)), l.tape, ($opType)(l, r))
            end
            
            function broadcast(::typeof($op), ldatum::L, r::R) where L where R <: GenValue
                l = GenValue(ldatum, r.tape)
                GenValue(broadcast($op, ldatum, datum(r)), l.tape, ($opType)(l, r))
            end
            
            function broadcast(::typeof($op), l::L, rdatum::R) where L <: GenValue where R
                r = GenValue(rdatum, l.tape)
                GenValue(broadcast($op, datum(l), rdatum), l.tape, ($opType)(l, r))
            end
        end)
end


# show

import Base.show

function show(io::IO, num::GenScalar)
    print(io, "GenScalar(datum=$(num.datum), idx=$(num.tapeIdx))")
end

function show(io::IO, num::GenVector)
    print(io, "GenVector(datum=$(num.datum), idx=$(num.tapeIdx))")
end


# common

datum(x::Real) = x
datum(x::Vector{Real}) = x
datum(x::GenValue) = x.datum

function check_tapes(a::GenValue, b::GenValue)
    if a.tape != b.tape
        error("$a and $b use different tapes")
    end
end


# operators

struct Input <: AbstractOperator end
propagate{T}(op::Input, datum::T, adj::T) = nothing # no-op


# divide

import Base./
import Base.broadcast

struct Divide{T,U} <: AbstractOperator
    left::T
    right::U
end

@generate_binary_operators(/, Divide)

function propagate{T<:GenScalar,U<:GenScalar}(op::Divide{T,U}, datum::Real, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end

function propagate{T<:GenVector,U<:GenScalar,W<:Real}(op::Divide{T,U}, datum::Vector{W}, 
                                                      adj::Vector{Float64})
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

function propagate{T<:GenScalar,U<:GenVector,W<:Real}(op::Divide{T,U}, datum::Vector{W},
                                                      adj::Vector{Float64})
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end



# element-wise multiplication (matrix multiply is separate)

import Base.*

struct Multiply{T,U} <: AbstractOperator
    left::T
    right::U
end

@generate_binary_operators(*, Multiply)


t = Tape()

x = GenScalar(2.0, t) / GenScalar(1.0, t)
println(x)

x = 3.0 / GenScalar(1.0, t)
println(x)

x = GenScalar(3.0, t) / 2
println(x)

x = GenVector([1.0, 3.0], t) ./ GenVector([2.0, 2.0], t)
println(x)

x = GenScalar(3, t) ./ GenVector([2.0, 2.0], t)
println(x)

x = GenVector([1.0, 3.0], t) / GenScalar(2, t)
println(x)

x = GenVector([1.0, 3.0], t) / 2
println(x)

println("calling with broadcast")
x = broadcast(/, 2., GenVector([1.0, 3.0], t))
println(x)

x = [2.0, 2.0] ./ GenVector([1.0, 3.0], t)
println(x)

# backward pass

function backprop(a::GenScalar)
    a.adj = 1.0 # this is the root node
    ns = nums(a.tape)
    for i=a.tapeIdx:-1:1
        propagate(ns[i].op, ns[i].datum, ns[i].adj)
    end
end

partial{T <: GenValue}(a::T) = a.adj # partial derivative


# test
using Base.Test

dx = 1e-6

function finite_difference(f, val::Float64)
    x_pos = val + dx
    x_neg = val - dx
    (f(x_pos) - f(x_neg)) / (2. * dx)
end

a_val = 1.1
b_val = 2.1
tape = Tape()
a = GenScalar(a_val, tape)
b = GenScalar(b_val, tape)
c = a / b
backprop(c)
@test isapprox(partial(a), finite_difference((x) -> x / b_val, a_val))
@test isapprox(partial(b), finite_difference((x) -> a_val / x, b_val))
@test datum(a / b) == datum(a) / datum(b)

# scalar / scalar
# scalar / vector (broadcast)
# scalar / matrix (broadcast)


# vector / vector (the same size)
# matrix / matrix (the same size)

# vector / scalar
# matrix / scalar

# matrix / vector (broadcast the vector)
# vector / matrix (broadcast the vector)

# also for row vectors?
