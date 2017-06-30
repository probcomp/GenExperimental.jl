
# tape for reverse-mode auto-differentiation

mutable struct Tape
    nums::Array{Any, 1}
    function Tape()
        new(Array{Any, 1}())
    end
end
nums(t::Tape) = t.nums


# each numeric struct records the operation by which it was produced, and its
# operands

abstract type AbstractOperator end

struct Input <: AbstractOperator end
propagate{T}(op::Input, datum::T, adj::T) = nothing # no-op


# scalar numeric type

mutable struct GenScalar{S<:Real, T<:AbstractOperator}
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


# vector numeric type

mutable struct GenVector{S<:Real, T<:AbstractOperator} <: AbstractArray{GenScalar{S}, 1}
    # NOTE: the datum is immutable (there is no setindex! operator)
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

GenVector{S<:Real}(datum::Vector{S}, tape::Tape) = GenVector(datum, tape, Input())

Base.size(v::GenVector) = size(v.datum)
Base.IndexStyle(v::GenVector) = IndexLinear()


# matrix numeric type

mutable struct GenMatrix{S<:Real, T<:AbstractOperator} <: AbstractArray{GenScalar{S}, 2}
    # NOTE: the datum is immutable (there is not setindex! operator)
    datum::Matrix{S}
    adj::Matrix{Float64}
    tape::Tape
    tapeIdx::Int
    op::T
end

function GenMatrix{S<:Real, T<:AbstractOperator}(datum::Matrix{S}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenMatrix{S,T}(datum, zeros(datum), tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

GenMatrix{S<:Real}(datum::Matrix{S}, tape::Tape) = GenMatrix(datum, tape, Input())

Base.size(v::GenMatrix) = size(v.datum)
Base.IndexStyle(v::GenMatrix) = IndexLinear()


# show

import Base.show

function show(io::IO, num::GenScalar)
    print(io, "GenScalar(datum=$(num.datum), idx=$(num.tapeIdx))")
end

function show(io::IO, num::GenVector)
    print(io, "GenVector(datum=$(num.datum), idx=$(num.tapeIdx))")
end

function show(io::IO, num::GenMatrix)
    print(io, "GenMatrix(datum=$(num.datum), idx=$(num.tapeIdx))")
end


# union type

GenValue = Union{GenScalar, GenVector, GenMatrix}


# indexing into a vector gives a scalar

import Base.getindex

struct GetVectorIndex <: AbstractOperator
    arg::GenVector
    i::Int
end

Base.getindex(arg::GenVector, i::Int) = GenScalar(arg.datum[i], arg.tape, GetVectorIndex(arg, i))

function propagate(op::GetVectorIndex, datum::Real, adj::Float64)
    op.arg.adj[op.i] += adj
end

# indexing into a matrix gives a scalar

struct GetMatrixIndex <: AbstractOperator
    arg::GenMatrix
    i::Int
end

Base.getindex(arg::GenMatrix, i::Int) = GenScalar(arg.datum[i], arg.tape, GetMatrixIndex(arg, i))

function propagate(op::GetMatrixIndex, datum::Real, adj::Float64)
    op.arg.adj[op.i] += adj
end


# getting concrete values from the numeric struct

datum(x::Real) = x
datum(x::Vector{Real}) = x
datum(x::GenValue) = x.datum


# check that two numeric structs are located on the same tape

function check_tapes(a::GenValue, b::GenValue)
    if a.tape != b.tape
        error("$a and $b use different tapes")
    end
end

# backward pass

function backprop(a::GenValue)
    a.adj = 1.0 # this is the root node
    ns = nums(a.tape)
    for i=a.tapeIdx:-1:1
        propagate(ns[i].op, ns[i].datum, ns[i].adj)
    end
end

partial{T <: GenValue}(a::T) = a.adj # partial derivative

# exports
export Tape
export GenScalar
export GenVector
export GenMatrix
export datum
export backprop
export partial
