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

makeGenValue(datum::Vector{W}, tape::Tape) where W<:Real = GenVector(datum, tape, Input())
makeGenValue(datum::Vector{W}, tape::Tape, op::AbstractOperator) where W<:Real = GenVector(datum, tape, op)

makeGenValue(datum::Matrix{W}, tape::Tape) where W<:Real = GenMatrix(datum, tape, Input())
makeGenValue(datum::Matrix{W}, tape::Tape, op::AbstractOperator) where W<:Real = GenMatrix(datum, tape, op)



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

function propagate(op::ElementwiseDivide{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
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

function broadcast(::typeof(/), l::GenScalar, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) ./ rdatum, l.tape, ElementwiseDivide(l, r))
end

function propagate(op::ElementwiseDivide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# vector / scalar
function (/)(l::GenVector, r::GenScalar)
    makeGenValue((/)(datum(l), datum(r)), l.tape, ElementwiseDivide(l, r))
end

function (/)(ldatum::Vector{W}, r::GenScalar) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((/)(ldatum, datum(r)), l.tape, ElementwiseDivide(l, r))
end

function (/)(l::GenVector, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((/)(datum(l), rdatum), l.tape, ElementwiseDivide(l, r))
end

function propagate(op::ElementwiseDivide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# vector ./ vector
function broadcast(::typeof(/), l::GenVector, r::GenVector)
    makeGenValue(datum(l) ./ datum(r), l.tape, ElementwiseDivide(l, r))
end

function broadcast(::typeof(/), ldatum::Vector{W}, r::GenVector) where W <: Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum ./ datum(r), l.tape, ElementwiseDivide(l, r))
end

function broadcast(::typeof(/), l::GenVector, rdatum::Vector{W}) where W <: Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) ./ rdatum, l.tape, ElementwiseDivide(l, r))
end

function propagate(op::ElementwiseDivide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenVector, W<:Real}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end


# ---- element-wise multiplication ----

@generate_binary_operator_type(ElementwiseMultiply)

# scalar * scalar
function (*)(l::GenScalar, r::GenScalar)
    makeGenValue((*)(datum(l), datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(ldatum::Real, r::GenScalar)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(l::GenScalar, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj * op.right.datum
    op.right.adj += adj * op.left.datum
end

# scalar * vector
function (*)(l::GenScalar, r::GenVector)
    makeGenValue((*)(datum(l), datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(ldatum::Real, r::GenVector)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(l::GenScalar, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end


# vector * scalar
function (*)(l::GenVector, r::GenScalar)
    makeGenValue((*)(datum(l), datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(ldatum::Vector{W}, r::GenScalar) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(l::GenVector, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# vector .* vector
function broadcast(::typeof(*), l::GenVector, r::GenVector)
    makeGenValue(datum(l) .* datum(r), l.tape, ElementwiseMultiply(l, r))
end

function broadcast(::typeof(*), ldatum::Vector{W}, r::GenVector) where W <: Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum .* datum(r), l.tape, ElementwiseMultiply(l, r))
end

function broadcast(::typeof(*), l::GenVector, rdatum::Vector{W}) where W <: Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) .* rdatum, l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenVector, W<:Real}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# scalar * matrix
function (*)(l::GenScalar, r::GenMatrix)
    makeGenValue((*)(datum(l), datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(ldatum::Real, r::GenMatrix)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(l::GenScalar, rdatum::Matrix{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenScalar, U<:GenMatrix, W<:Real}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# matrix * scalar
function (*)(l::GenMatrix, r::GenScalar)
    makeGenValue((*)(datum(l), datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(ldatum::Matrix{W}, r::GenScalar) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, ElementwiseMultiply(l, r))
end

function (*)(l::GenMatrix, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenScalar, W<:Real}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# matrix .* matrix
function broadcast(::typeof(*), l::GenMatrix, r::GenMatrix)
    makeGenValue(datum(l) .* datum(r), l.tape, ElementwiseMultiply(l, r))
end

function broadcast(::typeof(*), ldatum::Matrix{W}, r::GenMatrix) where W <: Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum .* datum(r), l.tape, ElementwiseMultiply(l, r))
end

function broadcast(::typeof(*), l::GenMatrix, rdatum::Matrix{W}) where W <: Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) .* rdatum, l.tape, ElementwiseMultiply(l, r))
end

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenMatrix, W<:Real}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# matrix .* vector (broadcast)

# vector .* matrix (broadcast)





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



