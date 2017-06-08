abstract AbstractOperator
abstract Dual{T}

type Tape
    nums::Array{Dual, 1}
    function Tape()
        new(Array{Dual, 1}())
    end
end
nums(t::Tape) = t.nums


# TODO generalize to other number types besides Float64?

type GenFloat{T <: AbstractOperator} <: Dual{T}
    # NOTE: the datum is intended to be immutable
    datum::Float64
    adj::Float64
    tape::Tape
    tapeIdx::Int # we will walk backward from this index
    op::T # stores parameters to operator, and has derivative method
end

type GenMatrix{T <: AbstractOperator} <: Dual{T}
    # NOTE: the datum is intended to be immutable (there should be no setindex operator)
    datum::Matrix{Float64}
    adj::Matrix{Float64}
    tape::Tape
    tapeIdx::Int # we will walk backward from this index
    op::T # stores parameters to operator, and has derivative method
end

concrete(x::GenFloat) = x.datum
concrete(x::Real) = x

function show(num::GenFloat)
    println("GenFloat(datum=$(num.datum), adj=$(num.adj), idx=$(num.tapeIdx))")
end

function GenFloat{T <: AbstractOperator}(datum::Float64, tape::Tape, op::T)
    ns = nums(tape)
    num = GenFloat{T}(datum, 0.0, tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

function GenMatrix{T <: AbstractOperator}(datum::Array{Float64,2}, tape::Tape, op::T)
    ns = nums(tape)
    num = GenMatrix{T}(datum, zeros(datum), tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

function check_tapes(a::Dual, b::Dual)
    if a.tape != b.tape
        error("cannot $a and $b use different tapes")
    end
end

function GenFloat(datum::Float64, tape::Tape)
    GenFloat(datum, tape, Input())
end

function GenMatrix(datum::Array{Float64,2}, tape::Tape)
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
            function ($op)(l::GenFloat, r::GenFloat)
                check_tapes(l, r)
                GenFloat($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end

            function ($op)(l::GenFloat, r::Float64)
                rnum = GenFloat(r, l.tape, Input())
                GenFloat($(op)(l.datum, r), l.tape, ($opType)(l, rnum))
            end
            function ($op)(l::Float64, r::GenFloat)
                lnum = GenFloat(l, r.tape, Input())
                GenFloat($(op)(l, r.datum), r.tape, ($opType)(lnum, r))
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
                rnum = GenFloat(r, l.tape, Input())
                GenMatrix($(op)(l.datum, r), l.tape, ($opType)(l, rnum))
            end
            function ($op)(l::Float64, r::GenMatrix)
                lnum = GenFloat(l, r.tape, Input())
                GenMatrix($(op)(l, r.datum), r.tape, ($opType)(lnum, r))
            end

            function ($op)(l::GenMatrix, r::GenFloat)
                check_tapes(l, r)
                GenMatrix($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end
            function ($op)(l::GenFloat, r::GenMatrix)
                check_tapes(l, r)
                GenMatrix($(op)(l.datum, r.datum), r.tape, ($opType)(l, r))
            end
        end)
end

macro generate_ad_unary_operator(op, opType)
    eval(quote
            immutable ($opType){T} <: AbstractOperator
                arg::T
            end
            function ($op)(arg::GenFloat)
                GenFloat($(op)(arg.datum), arg.tape, ($opType)(arg))
            end
            function ($op)(arg::GenMatrix)
                GenMatrix($(op)(arg.datum), arg.tape, ($opType)(arg))
            end
        end)
end

# getindex
import Base.getindex
immutable GetIndex <: AbstractOperator
    arg::GenMatrix
    i1::Real
    i2::Real
end
function getindex(arg::GenMatrix, i1::Real, i2::Real)
    GenFloat(arg.datum[i1, i2], arg.tape, GetIndex(arg, i1, i2))
end
function propagate(op::GetIndex, datum::Float64, adj::Float64)
    op.arg.adj[op.i1, op.i2] += adj
end

# transpose
import Base.transpose
immutable TransposeOp <: AbstractOperator
    arg::GenMatrix
end
function transpose(arg::GenMatrix)
    GenMatrix(arg.datum', arg.tape, TransposeOp(arg))
end
function propagate(op::TransposeOp, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.arg.adj += adj'
end


# sum
import Base.sum
immutable Sum <: AbstractOperator
    arg::GenMatrix
end
function sum(arg::GenMatrix)
    GenFloat(sum(arg.datum), arg.tape, Sum(arg))
end
function propagate(op::Sum, datum::Float64, adj::Float64)
    op.arg.adj += adj
end

# +
import Base.+
@generate_ad_binary_operator(+, Plus)
function propagate{T<:GenFloat,U<:GenFloat}(op::Plus{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj
    op.right.adj += adj
end
function propagate{T<:GenMatrix,U<:GenFloat}(op::Plus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj
    op.right.adj += sum(adj)
end
function propagate{T<:GenFloat,U<:GenMatrix}(op::Plus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += sum(adj)
    op.right.adj += adj
end
function propagate{T<:GenMatrix,U<:GenMatrix}(op::Plus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
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
function propagate{T<:GenFloat,U<:GenFloat}(op::Minus{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj
    op.right.adj -= adj
end
function propagate{T<:GenMatrix,U<:GenFloat}(op::Minus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj
    op.right.adj -= sum(adj)
end
function propagate{T<:GenFloat,U<:GenMatrix}(op::Minus{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += sum(adj)
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
function propagate{T<:GenFloat,U<:GenFloat}(op::Times{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj * op.right.datum
    op.right.adj += adj * op.left.datum
end
function propagate{T<:GenFloat,U<:GenMatrix}(op::Times{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    # scalar * matrix
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end
function propagate{T<:GenMatrix,U<:GenFloat}(op::Times{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    # matrix * scalar 
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end
function propagate{T<:GenMatrix,U<:GenMatrix}(op::Times{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    # matrix * matrix
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end

# /
import Base./
@generate_ad_binary_operator(/, Divide)
function propagate{T<:GenFloat,U<:GenFloat}(op::Divide{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end
function propagate{T<:GenMatrix,U<:GenFloat}(op::Divide{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# ./
import Base.(./)
@generate_ad_binary_operator(./, ElementwiseDivide)
function propagate{T<:GenFloat,U<:GenFloat}(op::ElementwiseDivide{T,U}, datum::Float64, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end
function propagate{T<:GenFloat,U<:GenMatrix}(op::ElementwiseDivide{T,U}, datum::Matrix{Float64}, adj::Matrix{Float64})
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
function backprop(a::GenFloat)
    a.adj = 1.0 # this is the root node
    ns = nums(a.tape)
    for i=a.tapeIdx:-1:1
        propagate(ns[i].op, ns[i].datum, ns[i].adj)
    end
end

partial{T <: Dual}(a::T) = a.adj # partial derivative

# exports
export Tape
export show 
export GenFloat
export GenMatrix
export concrete
export backprop
export adj
export partial
