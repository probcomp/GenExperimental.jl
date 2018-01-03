
# tape for reverse-mode auto-differentiation

mutable struct Tape
    nums::Array{Any, 1}
    function Tape()
        new(Array{Any, 1}())
    end
end
nums(t::Tape) = t.nums

function zero_grad!(tape::Tape)
    for num in tape.nums
        zero_grad!(num)
    end
end


# each numeric struct records the operation by which it was produced, and its
# operands

abstract type AbstractOperator end

struct Input <: AbstractOperator end
propagate(op::Input, datum::T, adj::U) where {T,U} = nothing # no-op


#######################
# Scalar numeric type #
#######################

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

Base.getindex(v::GenScalar, i::Int) =  i == 1 ? v : error("Invalid index $i into scalar")

zero_grad!(v::GenScalar) = (v.adj = 0.)


##############################
# Column vector numeric type #
##############################

mutable struct GenColumnVector{S<:Real, T<:AbstractOperator}# <: AbstractArray{GenScalar{S}, 1}
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

Base.IndexStyle(v::GenColumnVector) = IndexLinear()

zero_grad!(v::GenColumnVector) = (v.adj = zeros(Float64, size(v.datum)))


###########################
# Row vector numeric type #
###########################

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

GenRowVector{S<:Real}(datum::RowVector{S}, tape::Tape) = GenRowVector(datum, tape, Input())

Base.IndexStyle(v::GenRowVector) = IndexLinear()

zero_grad!(v::GenRowVector) = (v.adj = transpose(zeros(Float64, length(v.datum))))


#######################
# Matrix numeric type #
#######################

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

Base.IndexStyle(v::GenMatrix) = IndexLinear()

zero_grad!(v::GenMatrix) = (v.adj = zeros(v.datum))


#######################################
# Methods common to all numeric types #
#######################################

GenValue = Union{GenScalar, GenColumnVector, GenRowVector, GenMatrix}

Base.size(v::GenValue) = size(v.datum)

Base.length(v::GenValue) = length(v.datum)


###############
# Other types #
###############

GenVector = Union{GenColumnVector, GenRowVector}

ConcreteScalar = Real
ConcreteColumnVector = Vector{W} where W<:Real
ConcreteRowVector = RowVector{W, Vector{W}} where W<:Real
ConcreteMatrix = Matrix{W} where W<:Real

ConcreteValue = Union{
    ConcreteScalar,
    ConcreteColumnVector,
    ConcreteRowVector,
    ConcreteMatrix
}

# element-wise operations on concrete values are equivalent to broadcast
ewise(f, a::ConcreteValue) = broadcast(f, a)
ewise(f, a::ConcreteValue, b::ConcreteValue) = broadcast(f, a, b)


# create Gen value from a concrete value

makeGenValue(datum::ConcreteScalar, tape::Tape) = GenScalar(datum, tape, Input())
makeGenValue(datum::ConcreteScalar, tape::Tape, op::AbstractOperator) = GenScalar(datum, tape, op)

makeGenValue(datum::ConcreteColumnVector, tape::Tape) = GenColumnVector(datum, tape, Input())
makeGenValue(datum::ConcreteColumnVector, tape::Tape, op::AbstractOperator) = GenColumnVector(datum, tape, op)

makeGenValue(datum::ConcreteRowVector, tape::Tape) = GenRowVector(datum, tape, Input())
makeGenValue(datum::ConcreteRowVector, tape::Tape, op::AbstractOperator) = GenRowVector(datum, tape, op)

makeGenValue(datum::ConcreteMatrix, tape::Tape) = GenMatrix(datum, tape, Input())
makeGenValue(datum::ConcreteMatrix, tape::Tape, op::AbstractOperator) = GenMatrix(datum, tape, op)

# adjoints always use Float64

ScalarAdjoint = Float64
ColumnVectorAdjoint = ConcreteColumnVector{Float64}
RowVectorAdjoint = ConcreteRowVector{Float64}
MatrixAdjoint = ConcreteMatrix{Float64}

# show
import Base.show
show(io::IO, num::GenValue) = show(io, num.datum)

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

# range-indexing into a (column,row) vector gives a (column,row) vector
# T is either GenColumnVector or GenRowVector
# U is either Vector{Float64} (for GenColumnVector) or RowVector{Float64} (for GenRowVector)

struct GetVectorUnitRange{T} <: AbstractOperator
    arg::T
    range::UnitRange{Int}
end

Base.getindex(arg::GenColumnVector, range::UnitRange{Int}) = GenColumnVector(arg.datum[range], arg.tape, GetVectorUnitRange(arg, range))


Base.getindex(arg::GenRowVector, range::UnitRange{Int}) = GenRowVector(arg.datum[range], arg.tape, GetVectorUnitRange(arg, range))

function propagate(op::GetVectorUnitRange, datum::U, adj::U) where {U}
    op.arg.adj[op.range] += adj
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

concrete(x::ConcreteValue) = x
concrete(x::GenValue) = x.datum


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
export zero_grad!
export GenValue
export makeGenValue
export ConcreteScalar
export ConcreteColumnVector
export ConcreteRowVector
export ConcreteMatrix
export ConcreteValue
export concrete
export backprop
export partial
export ewise
