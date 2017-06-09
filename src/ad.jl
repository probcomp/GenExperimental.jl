abstract AbstractOperator
abstract GenVal{T}

type Tape
    nums::Array{GenVal, 1}
    function Tape()
        new(Array{GenVal, 1}())
    end
end
nums(t::Tape) = t.nums

type GenScalar{T <: AbstractOperator} <: GenVal{T}
    # NOTE: the datum is intended to be immutable
    datum::Float64
    adj::Float64
    tape::Tape
    tapeIdx::Int
    op::T
end

type GenVector{T <: AbstractOperator} <: GenVal{T}
    # NOTE: the datum is intended to be immutable (there should be no setindex operator)
    datum::Vector{Float64}
    adj::Vector{Float64}
    tape::Tape
    tapeIdx::Int
    op::T
end

type GenMatrix{T <: AbstractOperator} <: GenVal{T}
    # NOTE: the datum is intended to be immutable (there should be no setindex operator)
    datum::Matrix{Float64}
    adj::Matrix{Float64}
    tape::Tape
    tapeIdx::Int
    op::T
end

concrete(x::GenVal) = x.datum
concrete(x::Real) = x

function show(num::GenScalar)
    println("GenScalar(datum=$(num.datum), adj=$(num.adj), idx=$(num.tapeIdx))")
end

function GenScalar{T <: AbstractOperator}(datum::Float64, tape::Tape, op::T)
    ns = nums(tape)
    num = GenScalar{T}(datum, 0.0, tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

function GenVector{T <: AbstractOperator}(datum::Vector{Float64}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenScalar{T}(datum, 0.0, tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

function GenMatrix{T <: AbstractOperator}(datum::Matrix{Float64}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenMatrix{T}(datum, zeros(datum), tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

function check_tapes(a::GenVal, b::GenVal)
    if a.tape != b.tape
        error("cannot $a and $b use different tapes")
    end
end

function GenScalar(datum::Float64, tape::Tape)
    GenScalar(datum, tape, Input())
end

function GenVector(datum::Vector{Float64}, tape::Tape)
    GenVector(datum, tape, Input())
end

function GenMatrix(datum::Matrix{Float64}, tape::Tape)
    GenMatrix(datum, tape, Input())
end

immutable Input <: AbstractOperator end
propagate{T}(op::Input, datum::T, adj::T) = nothing # no-op

macro generate_ad_binary_operator(op, opType)
    eval(quote
            immutable $(opType){T,U} <: AbstractOperator
                left::T
                right::U
            end
            function ($op)(l::GenScalar, r::GenScalar)
                check_tapes(l, r)
                GenScalar($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end

            function ($op)(l::GenScalar, r::Float64)
                rnum = GenScalar(r, l.tape, Input())
                GenScalar($(op)(l.datum, r), l.tape, ($opType)(l, rnum))
            end
            function ($op)(l::Float64, r::GenScalar)
                lnum = GenScalar(l, r.tape, Input())
                GenScalar($(op)(l, r.datum), r.tape, ($opType)(lnum, r))
            end

            function ($op)(l::GenMatrix, r::GenMatrix)
                check_tapes(l, r)
                GenMatrix($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end

            function ($op)(l::GenMatrix, r::Matrix{Float64})
                rnum = GenMatrix(r, l.tape, Input())
                GenMatrix($(op)(l.datum, r), l.tape, ($opType)(l, rnum))
            end
            function ($op)(l::Matrix{Float64}, r::GenMatrix)
                lnum = GenMatrix(l, r.tape, Input())
                GenMatrix($(op)(l, r.datum), r.tape, ($opType)(lnum, r))
            end

            function ($op)(l::GenMatrix, r::Float64)
                rnum = GenScalar(r, l.tape, Input())
                GenMatrix($(op)(l.datum, r), l.tape, ($opType)(l, rnum))
            end
            function ($op)(l::Float64, r::GenMatrix)
                lnum = GenScalar(l, r.tape, Input())
                GenMatrix($(op)(l, r.datum), r.tape, ($opType)(lnum, r))
            end

            function ($op)(l::GenMatrix, r::GenScalar)
                check_tapes(l, r)
                GenMatrix($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end
            function ($op)(l::GenScalar, r::GenMatrix)
                check_tapes(l, r)
                GenMatrix($(op)(l.datum, r.datum), r.tape, ($opType)(l, r))
            end

            function ($op){T}(l::T, r::T)
                check_tapes(l, r)
                T($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end

        end)
end

macro generate_ad_unary_operator(op, opType)
    eval(quote
            immutable ($opType){T} <: AbstractOperator
                arg::T
            end
            function ($op)(arg::GenScalar)
                GenScalar($(op)(arg.datum), arg.tape, ($opType)(arg))
            end
            function ($op)(arg::GenMatrix)
                GenMatrix($(op)(arg.datum), arg.tape, ($opType)(arg))
            end
        end)
end

# getindex
import Base.getindex
immutable GetVectorIndex <: AbstractOperator
    arg::GenVector
    i::Real
end
function getindex(arg::GenVector, i::Real)
    GenScalar(arg.datum[i], arg.tape, GetVectorIndex(arg, i))
end
function propagate(op::GetVectorIndex, datum::Float64, adj::Float64)
    op.arg.adj[op.i] += adj
end
immutable GetMatrixIndex <: AbstractOperator
    arg::GenMatrix
    i1::Real
    i2::Real
end
function getindex(arg::GenMatrix, i1::Real, i2::Real)
    GenScalar(arg.datum[i1, i2], arg.tape, GetMatrixIndex(arg, i1, i2))
end
function propagate(op::GetMatrixIndex, datum::Float64, adj::Float64)
    op.arg.adj[op.i1, op.i2] += adj
end

# TODO support more elaborate indexing schemes
# TODO support hcat, vcat

# transpose
import Base.transpose
immutable TransposeVector <: AbstractOperator
    arg::GenVector
end
function transpose(arg::GenVector)
    # transposing a vector creates a row matrix
    GenMatrix(arg.datum', arg.tape, TransposeVector(arg))
end
function propagate(op::TransposeVector, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.arg.adj += adj[:]
end
immutable TransposeMatrix <: AbstractOperator
    arg::GenMatrix
end
function transpose(arg::GenMatrix)
    GenMatrix(arg.datum', arg.tape, TransposeMatrix(arg))
end
function propagate(op::TransposeMatrix, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.arg.adj += adj'
end

# sum
import Base.sum
immutable VectorSum <: AbstractOperator
    arg::GenVector
end
function sum(arg::GenVector)
    GenScalar(sum(arg.datum), arg.tape, VectorSum(arg))
end
function propagate(op::VectorSum, datum::Float64, adj::Float64)
    op.arg.adj += adj
end
immutable MatrixSum <: AbstractOperator
    arg::GenMatrix
end
function sum(arg::GenMatrix)
    GenScalar(sum(arg.datum), arg.tape, MatrixSum(arg))
end
function propagate(op::MatrixSum, datum::Float64, adj::Float64)
    op.arg.adj += adj
end

# TODO handle broadcasting

# +
import Base.+
@generate_ad_binary_operator(+, Plus)
function propagate{T<:GenScalar,U<:GenScalar}(op::Plus{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj
    op.right.adj += adj
end
function propagate{T<:GenMatrix,U<:GenScalar}(op::Plus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj
    op.right.adj += sum(adj)
end
function propagate{T<:GenScalar,U<:GenMatrix}(op::Plus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += sum(adj)
    op.right.adj += adj
end
function propagate{T<:GenMatrix,U<:GenMatrix}(op::Plus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj
    op.right.adj += adj
end
function propagate{T<:GenVector,U<:GenScalar}(op::Plus{T,U}, datum::Vector{Float64}, adj::Vector{Float64})
    op.left.adj += adj
    op.right.adj += sum(adj)
end
function propagate{T<:GenScalar,U<:GenVector}(op::Plus{T,U}, datum::Vector{Float64}, adj::Vector{Float64})
    op.left.adj += sum(adj)
    op.right.adj += adj
end
function propagate{T<:GenVector,U<:GenVector}(op::Plus{T,U}, datum::Vector{Float64}, adj::Vector{Float64})
    op.left.adj += adj
    op.right.adj += adj
end


# +
import Base.+
@generate_ad_unary_operator(+, UnaryPlus)
function propagate{T}(op::UnaryPlus, datum::T, adj::T)
    op.arg.adj += adj
end

# -
import Base.-
@generate_ad_binary_operator(-, Minus)
function propagate{T<:GenScalar,U<:GenScalar}(op::Minus{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj
    op.right.adj -= adj
end
function propagate{T<:GenMatrix,U<:GenScalar}(op::Minus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj
    op.right.adj -= sum(adj)
end
function propagate{T<:GenScalar,U<:GenMatrix}(op::Minus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += sum(adj)
    op.right.adj -= adj
end
function propagate{T<:GenVector,U<:GenScalar}(op::Minus{T,U}, datum::Vector{Float64}, adj::Float64)
    op.left.adj += adj
    op.right.adj -= sum(adj)
end
function propagate{T<:GenScalar,U<:GenVector}(op::Minus{T,U}, datum::Float64, adj::Vector{Float64})
    op.left.adj += sum(adj)
    op.right.adj -= adj
end
function propagate{T<:GenVector,U<:GenVector}(op::Minus{T,U}, datum::Vector{Float64}, adj::Vector{Float64})
    op.left.adj += adj
    op.right.adj -= adj
end

# +
import Base.-
@generate_ad_unary_operator(-, UnaryMinus)
function propagate{T}(op::UnaryMinus, datum::T, adj::T)
    op.arg.adj -= adj
end


# *
import Base.*
@generate_ad_binary_operator(*, Times)
function propagate{T<:GenScalar,U<:GenScalar}(op::Times{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj * op.right.datum
    op.right.adj += adj * op.left.datum
end
function propagate{T<:GenScalar,U<:GenMatrix}(op::Times{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    # scalar * matrix
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end
function propagate{T<:GenMatrix,U<:GenScalar}(op::Times{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    # matrix * scalar 
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end
function propagate{T<:GenMatrix,U<:GenMatrix}(op::Times{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    # matrix * matrix
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end
function propagate{T<:GenMatrix,U<:GenVector}(op::Times{T,U}, datum::Vector{Float64}, adj::Vector{Float64})
    # matrix * vector 
    op.left.adj += adj * op.right.datum' # TODO
    op.right.adj += op.left.datum' * adj
end
function propagate{T<:GenVector,U<:GenScalar}(op::Times{T,U}, datum::Vector{Float64}, adj::Vector{Float64})
    # matrix * scalar
    op.left.adj += adj * op.right.datum' # TODO
    op.right.adj += op.left.datum' * adj
end


# /
import Base./
@generate_ad_binary_operator(/, Divide)
function propagate{T<:GenScalar,U<:GenScalar}(op::Divide{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end
function propagate{T<:GenMatrix,U<:GenScalar}(op::Divide{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# ./
import Base.(./)
@generate_ad_binary_operator(./, ElementwiseDivide)
function propagate{T<:GenScalar,U<:GenScalar}(op::ElementwiseDivide{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end
function propagate{T<:GenScalar,U<:GenMatrix}(op::ElementwiseDivide{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end
function propagate{T<:GenMatrix,U<:GenMatrix}(op::ElementwiseDivide{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end



# log
import Base.log
@generate_ad_unary_operator(log, Log)
function propagate{T}(op::Log, datum::T, adj::T)
    op.arg.adj += adj ./ op.arg.datum
end

# exp
import Base.exp
@generate_ad_unary_operator(exp, Exp)
function propagate{T}(op::Exp, datum::T, adj::T)
    op.arg.adj += adj .* datum
end

# lgamma 
import Base.lgamma
@generate_ad_unary_operator(lgamma, LogGamma)
function propagate{T}(op::LogGamma, datum::T, adj::T)
    op.arg.adj += adj .* digamma(op.arg.datum)
end

# backward pass
function backprop(a::GenScalar)
    a.adj = 1.0 # this is the root node
    ns = nums(a.tape)
    for i=a.tapeIdx:-1:1
        propagate(ns[i].op, ns[i].datum, ns[i].adj)
    end
end

partial{T <: GenVal}(a::T) = a.adj # partial derivative

# exports
export Tape
export show 
export GenScalar
export GenMatrix
export concrete
export backprop
export adj
export partial
