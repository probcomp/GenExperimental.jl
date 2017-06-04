type Tape
    nums::Array
end

abstract AbstractOperator

type GenNum{T <: AbstractOperator}
    datum::Float64
    adj::Float64
    tape::Tape
    tapeIdx::Int # we will walk backward from this index
    op::T# stores parameters to operator, and has derivative method
end

concrete(x::Float64) = x
concrete(x::GenNum) = x.datum

function show(num::GenNum)
    println("GenNum(datum=$(num.datum), adj=$(num.adj), idx=$(num.tapeIdx))")
end

nums(t::Tape) = t.nums::Array{GenNum,1}
Tape() = Tape(Array{GenNum,1}())

function GenNum{T <: AbstractOperator}(datum::Float64, tape::Tape, op::T)
    ns = nums(tape)
    num = GenNum{T}(datum, 0.0, tape, length(ns) + 1, op)
    push!(ns, num)
    num
end

function check_tapes(a::GenNum, b::GenNum)
    if a.tape != b.tape
        error("cannot $a and $b use different tapes")
    end
end


# input
function GenNum(datum::Float64, tape::Tape)
    GenNum(datum, tape, Input())
end
immutable Input <: AbstractOperator end
propagate(op::Input, datum::Float64, adj::Float64) = nothing # no-op

macro generate_ad_binary_operator(op, opType)
    eval(quote
            immutable $(opType) <: AbstractOperator
                left::GenNum
                right::GenNum
            end
            function ($op)(l::GenNum, r::GenNum)
                check_tapes(l, r)
                GenNum($(op)(l.datum, r.datum), l.tape, ($opType)(l, r))
            end
            function ($op)(l::GenNum, r::Float64)
                rnum = GenNum(r, l.tape, Input())
                GenNum($(op)(l.datum, r), l.tape, ($opType)(l, rnum))
            end
            function ($op)(l::Float64, r::GenNum)
                lnum = GenNum(l, r.tape, Input())
                GenNum($(op)(l, r.datum), r.tape, ($opType)(lnum, r))
            end
        end)
end

macro generate_ad_unary_operator(op, opType)
    eval(quote
            immutable ($opType) <: AbstractOperator
                arg::GenNum
            end
            function ($op)(arg::GenNum)
                GenNum($(op)(arg.datum), arg.tape, ($opType)(arg))
            end
        end)
end

# +
import Base.+
@generate_ad_binary_operator(+, Plus)
function propagate(op::Plus, datum::Float64, adj::Float64)
    op.left.adj += adj
    op.right.adj += adj
end

# +
import Base.+
@generate_ad_unary_operator(+, UnaryPlus)
function propagate(op::UnaryPlus, datum::Float64, adj::Float64)
    op.arg.adj += adj
end

# -
import Base.-
@generate_ad_binary_operator(-, Minus)
function propagate(op::Minus, datum::Float64, adj::Float64)
    op.left.adj += adj
    op.right.adj -= adj
end

# +
import Base.-
@generate_ad_unary_operator(-, UnaryMinus)
function propagate(op::UnaryMinus, datum::Float64, adj::Float64)
    op.arg.adj -= adj
end


# *
import Base.*
@generate_ad_binary_operator(*, Times)
function propagate(op::Times, datum::Float64, adj::Float64)
    op.left.adj += adj * op.right.datum
    op.right.adj += adj * op.left.datum
end

# /
import Base./
@generate_ad_binary_operator(/, Divide)
function propagate(op::Divide, datum::Float64, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end

# log
import Base.log
@generate_ad_unary_operator(log, Log)
function propagate(op::Log, datum::Float64, adj::Float64)
    op.arg.adj += adj / op.arg.datum
end

# exp
import Base.exp
@generate_ad_unary_operator(exp, Exp)
function propagate(op::Exp, datum::Float64, adj::Float64)
    op.arg.adj += adj * datum
end

# lgamma 
import Base.lgamma
@generate_ad_unary_operator(lgamma, LogGamma)
function propagate(op::LogGamma, datum::Float64, adj::Float64)
    op.arg.adj += adj * digamma(op.arg.datum)
end


# backward pass
function backprop(a::GenNum)
    a.adj = 1.0 # this is the root node
    ns = nums(a.tape)
    for i=a.tapeIdx:-1:1
        propagate(ns[i].op, ns[i].datum, ns[i].adj)
    end
end

adj(a::GenNum) = a.adj
partial(a::GenNum) = a.adj # partial derivative

# exports
export Tape
export show 
export GenNum
export concrete
export backprop
export adj
export partial

