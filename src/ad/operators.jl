import Base.broadcast
import Base./
import Base.*

macro generate_binary_operator_type(opType)
    eval(quote
            struct $(opType){T,U} <: AbstractOperator
                left::T
                right::U
            end
    end)
end



# NOTES:
# adj is always based on Float64
# the pattern is to define the opertaors including combination of a concrete
# datum and a GenValue, and then define one backpropagation method



# GenValueconstructors

makeGenValue(datum::Real, tape::Tape) = GenScalar(datum, tape, Input())
makeGenValue(datum::Real, tape::Tape, op::AbstractOperator) = GenScalar(datum, tape, op)
makeGenValue(datum::Vector{U}, tape::Tape) where U <: Real = GenVector(datum, tape, Input())
makeGenValue(datum::Vector{U}, tape::Tape, op::AbstractOperator) where U <: Real = GenVector(datum, tape, op)
makeGenValue(datum::Vector{U}, tape::Tape, op::AbstractOperator) where U <: Real = GenVector(datum, tape, op)



# ---- element-wise division ----

@generate_binary_operator_type(ElementwiseDivide)

# scalar / scalar
function (/)(l::GenScalar, r::GenScalar)
    makeGenValue((/)(datum(l), datum(r)), l.tape, ElementwiseDivide(l, r))
end

function (/)(ldatum::Real, r::GenScalar)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((/)(ldatum, datum(r)), l.tape, ElementwiseDivide(l, r))
end

function (/)(l::GenScalar, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((/)(datum(l), rdatum), l.tape, ElementwiseDivide(l, r))
end

function propagate{T<:GenScalar,U<:GenScalar}(op::ElementwiseDivide{T,U}, datum::Real, adj::Float64)
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end

# scalar ./ vector
function broadcast(::typeof(/), l::GenScalar, r::GenVector)
    makeGenValue(datum(l) ./ datum(r), l.tape, ElementwiseDivide(l, r))
end

function broadcast(::typeof(/), ldatum::Real, r::GenVector)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum ./ datum(r), l.tape, ElementwiseDivide(l, r))
end

function broadcast(::typeof(/), l::GenScalar, rdatum::Vector{Real})
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) ./ rdatum, l.tape, ElementwiseDivide(l, r))
end

function propagate{T<:GenScalar,U<:GenVector,W<:Real}(op::ElementwiseDivide{T,U}, datum::Vector{W},
                                                      adj::Vector{Float64})
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# vector / scalar
function (/)(l::GenVector, r::GenScalar)
    makeGenValue((/)(datum(l), datum(r)), l.tape, ElementwiseDivide(l, r))
end

function (/)(ldatum::Vector{Real}, r::GenScalar)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((/)(ldatum, datum(r)), l.tape, ElementwiseDivide(l, r))
end

function (/)(l::GenVector, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((/)(datum(l), rdatum), l.tape, ElementwiseDivide(l, r))
end

function propagate{T<:GenVector,U<:GenScalar,W<:Real}(op::ElementwiseDivide{T,U}, datum::Vector{W}, 
                                                      adj::Vector{Float64})
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# vector ./ vector
function broadcast(::typeof(/), l::GenVector, r::GenVector)
    makeGenValue(datum(l) ./ datum(r), l.tape, ElementwiseDivide(l, r))
end

function broadcast(::typeof(/), ldatum::Vector{Real}, r::GenVector)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum ./ datum(r), l.tape, ElementwiseDivide(l, r))
end

function broadcast(::typeof(/), l::GenVector, rdatum::Vector{Real})
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) ./ rdatum, l.tape, ElementwiseDivide(l, r))
end

function propagate{T<:GenVector,U<:GenVector,W<:Real}(op::ElementwiseDivide{T,U}, datum::Vector{W},
                                                      adj::Vector{Float64})
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end


# ---- element-wise multiplication ----

# ---- matrix-matrix multiplication ----


# ---- matrix-vector multiplication ----


# ---- addition ----


# ---- subtraction ----


# backward pass

function backprop(a::GenValue)
    a.adj = 1.0 # this is the root node
    ns = nums(a.tape)
    for i=a.tapeIdx:-1:1
        propagate(ns[i].op, ns[i].datum, ns[i].adj)
    end
end

partial{T <: GenValue}(a::T) = a.adj # partial derivative



