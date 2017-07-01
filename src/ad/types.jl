
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


# column vector numeric type

mutable struct GenColumnVector{S<:Real, T<:AbstractOperator} <: AbstractArray{GenScalar{S}, 1}
    # NOTE: the datum is immutable (there is no setindex! operator)
    datum::Vector{S}
    adj::Vector{Float64}
    tape::Tape
    tapeIdx::Int
    op::T
end

function GenColumnVector{S<:Real, T<:AbstractOperator}(datum::Vector{S}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenColumnVector{S,T}(datum, zeros(Float64, size(datum)), tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

GenColumnVector{S<:Real}(datum::Vector{S}, tape::Tape) = GenColumnVector(datum, tape, Input())

Base.size(v::GenColumnVector) = size(v.datum)
Base.IndexStyle(v::GenColumnVector) = IndexLinear()


# row vector numeric type

mutable struct GenRowVector{S<:Real, T<:AbstractOperator} <: AbstractArray{GenScalar{S}, 1}
    # NOTE: the datum is immutable (there is no setindex! operator)
    datum::RowVector{S,Vector{S}}
    adj::RowVector{Float64, Vector{Float64}}
    tape::Tape
    tapeIdx::Int
    op::T
end

function GenRowVector{S<:Real, T<:AbstractOperator}(datum::RowVector{S,Vector{S}}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenRowVector{S,T}(datum, transpose(zeros(Float64, length(datum))), tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

GenRowVector{S<:Real}(datum::Vector{S}, tape::Tape) = GenRowVector(datum, tape, Input())

Base.size(v::GenRowVector) = size(v.datum)
Base.IndexStyle(v::GenRowVector) = IndexLinear()


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

function show(io::IO, num::GenColumnVector)
    print(io, "GenColumnVector(datum=$(num.datum), idx=$(num.tapeIdx))")
end

function show(io::IO, num::GenMatrix)
    print(io, "GenMatrix(datum=$(num.datum), idx=$(num.tapeIdx))")
end


# other types

GenValue = Union{GenScalar, GenColumnVector, GenRowVector, GenMatrix}
GenVector = Union{GenColumnVector, GenRowVector}
ColumnOrRowVector{W} = Union{Vector{W}, RowVector{W, Vector{W}}}

ConcreteValue = Union{
    Real,
    Vector{W} where W<:Real,
    RowVector{W,Vector{W}} where W<:Real,
    Matrix{W} where W<:Real
}



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
datum(x::ColumnOrRowVector{W}) where W<:Real = x
datum(x::Matrix{W}) where W<:Real = x
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
export GenColumnVector
export GenRowVector
export GenMatrix
export datum
export backprop
export partial
