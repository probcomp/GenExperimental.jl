import Base.broadcast
import Base.+
import Base.-
import Base./
import Base.*
import Base.log
import Base.exp
import Base.lgamma


macro generate_binary_operator_type(opType)
    eval(quote
            struct $(opType){T,U} <: AbstractOperator
                left::T
                right::U
            end
    end)
end

macro generate_unary_operator_type(opType)
    eval(quote
            struct $(opType){T} <: AbstractOperator
                arg::T
            end
        end)
end

makeGenValue(datum::Real, tape::Tape) = GenScalar(datum, tape, Input())
makeGenValue(datum::Real, tape::Tape, op::AbstractOperator) = GenScalar(datum, tape, op)

makeGenValue(datum::Vector{W}, tape::Tape) where W<:Real = GenVector(datum, tape, Input())
makeGenValue(datum::Vector{W}, tape::Tape, op::AbstractOperator) where W<:Real = GenVector(datum, tape, op)

makeGenValue(datum::Matrix{W}, tape::Tape) where W<:Real = GenMatrix(datum, tape, Input())
makeGenValue(datum::Matrix{W}, tape::Tape, op::AbstractOperator) where W<:Real = GenMatrix(datum, tape, op)


# NOTES:
# adj is always based on Float64
# the pattern is to define the opertaors including combination of a concrete
# datum and a GenValue, and then define one backpropagation method


# ---- addition ----

@generate_binary_operator_type(Add)

# scalar + scalar
function (+)(l::GenScalar, r::GenScalar)
    makeGenValue((+)(datum(l), datum(r)), l.tape, Add(l, r))
end

function (+)(ldatum::Real, r::GenScalar)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((+)(ldatum, datum(r)), l.tape, Add(l, r))
end

function (+)(l::GenScalar, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((+)(datum(l), rdatum), l.tape, Add(l, r))
end

function propagate(op::Add{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj
    op.right.adj += adj
end

# scalar + vector
function (+)(l::GenScalar, r::GenVector)
    makeGenValue((+)(datum(l), datum(r)), l.tape, Add(l, r))
end

function (+)(ldatum::Real, r::GenVector)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((+)(ldatum, datum(r)), l.tape, Add(l, r))
end

function (+)(l::GenScalar, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((+)(datum(l), rdatum), l.tape, Add(l, r))
end

function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    op.left.adj += sum(adj)
    op.right.adj += adj
end

# vector + scalar
function (+)(l::GenVector, r::GenScalar)
    makeGenValue((+)(datum(l), datum(r)), l.tape, Add(l, r))
end

function (+)(ldatum::Vector{W}, r::GenScalar) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((+)(ldatum, datum(r)), l.tape, Add(l, r))
end

function (+)(l::GenVector, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((+)(datum(l), rdatum), l.tape, Add(l, r))
end

function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    op.left.adj += adj
    op.right.adj += sum(adj)
end

# vector + vector
function (+)(l::GenVector, r::GenVector)
    makeGenValue((+)(datum(l), datum(r)), l.tape, Add(l, r))
end

function (+)(ldatum::Vector{W}, r::GenVector) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((+)(ldatum, datum(r)), l.tape, Add(l, r))
end

function (+)(l::GenVector, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((+)(datum(l), rdatum), l.tape, Add(l, r))
end

function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenVector, W<:Real}
    op.left.adj += adj
    op.right.adj += adj
end

# scalar + matrix
# TODO not implemented yet

# matrix + scalar
# TODO not implemented yet

# matrix .+ vector (broadcast)
# TODO not implemented yet

# vector .+ matrix (broadcast)
# TODO not implemented yet

# matrix + matrix
# TODO not implemented yet


# ---- subtraction ----

@generate_binary_operator_type(Subtract)

# scalar - scalar
function (-)(l::GenScalar, r::GenScalar)
    makeGenValue((-)(datum(l), datum(r)), l.tape, Subtract(l, r))
end

function (-)(ldatum::Real, r::GenScalar)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((-)(ldatum, datum(r)), l.tape, Subtract(l, r))
end

function (-)(l::GenScalar, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((-)(datum(l), rdatum), l.tape, Subtract(l, r))
end

function propagate(op::Subtract{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj
    op.right.adj -= adj
end

# scalar - vector
function (-)(l::GenScalar, r::GenVector)
    makeGenValue((-)(datum(l), datum(r)), l.tape, Subtract(l, r))
end

function (-)(ldatum::Real, r::GenVector)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((-)(ldatum, datum(r)), l.tape, Subtract(l, r))
end

function (-)(l::GenScalar, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((-)(datum(l), rdatum), l.tape, Subtract(l, r))
end

function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    op.left.adj += sum(adj)
    op.right.adj -= adj
end

# vector - scalar
function (-)(l::GenVector, r::GenScalar)
    makeGenValue((-)(datum(l), datum(r)), l.tape, Subtract(l, r))
end

function (-)(ldatum::Vector{W}, r::GenScalar) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((-)(ldatum, datum(r)), l.tape, Subtract(l, r))
end

function (-)(l::GenVector, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((-)(datum(l), rdatum), l.tape, Subtract(l, r))
end

function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    op.left.adj += adj
    op.right.adj -= sum(adj)
end

# vector - vector
function (-)(l::GenVector, r::GenVector)
    makeGenValue((-)(datum(l), datum(r)), l.tape, Subtract(l, r))
end

function (-)(ldatum::Vector{W}, r::GenVector) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((-)(ldatum, datum(r)), l.tape, Subtract(l, r))
end

function (-)(l::GenVector, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((-)(datum(l), rdatum), l.tape, Subtract(l, r))
end

function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenVector, W<:Real}
    op.left.adj += adj
    op.right.adj -= adj
end

# scalar - matrix
# TODO not implemented yet

# matrix - scalar
# TODO not implemented yet

# matrix .- vector (broadcast)
# TODO not implemented yet

# vector .- matrix (broadcast)
# TODO not implemented yet

# matrix - matrix
# TODO not implemented yet


# ---- division ----

@generate_binary_operator_type(Divide)

# scalar / scalar
function (/)(l::GenScalar, r::GenScalar)
    makeGenValue((/)(datum(l), datum(r)), l.tape, Divide(l, r))
end

function (/)(ldatum::Real, r::GenScalar)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((/)(ldatum, datum(r)), l.tape, Divide(l, r))
end

function (/)(l::GenScalar, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((/)(datum(l), rdatum), l.tape, Divide(l, r))
end

function propagate(op::Divide{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end

# scalar ./ vector
function broadcast(::typeof(/), l::GenScalar, r::GenVector)
    makeGenValue(datum(l) ./ datum(r), l.tape, Divide(l, r))
end

function broadcast(::typeof(/), ldatum::Real, r::GenVector)
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum ./ datum(r), l.tape, Divide(l, r))
end

function broadcast(::typeof(/), l::GenScalar, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) ./ rdatum, l.tape, Divide(l, r))
end

function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# vector / scalar
function (/)(l::GenVector, r::GenScalar)
    makeGenValue((/)(datum(l), datum(r)), l.tape, Divide(l, r))
end

function (/)(ldatum::Vector{W}, r::GenScalar) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((/)(ldatum, datum(r)), l.tape, Divide(l, r))
end

function (/)(l::GenVector, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((/)(datum(l), rdatum), l.tape, Divide(l, r))
end

function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# vector ./ vector
function broadcast(::typeof(/), l::GenVector, r::GenVector)
    makeGenValue(datum(l) ./ datum(r), l.tape, Divide(l, r))
end

function broadcast(::typeof(/), ldatum::Vector{W}, r::GenVector) where W <: Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue(ldatum ./ datum(r), l.tape, Divide(l, r))
end

function broadcast(::typeof(/), l::GenVector, rdatum::Vector{W}) where W <: Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue(datum(l) ./ rdatum, l.tape, Divide(l, r))
end

function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenVector, U<:GenVector, W<:Real}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# scalar ./ matrix
# TODO not implemented yet

# matrix / scalar
# TODO not implemented yet

# matrix ./ vector (broadcast)
# TODO not implemented yet

# vector ./ matrix (broadcast)
# TODO not implemented yet


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
# TODO not implemented yet

# vector .* matrix (broadcast)
# TODO not implemented yet


# ---- matrix multiplication ----

@generate_binary_operator_type(MatrixMultiply)

# matrix * matrix
function (*)(l::GenMatrix, r::GenMatrix)
    makeGenValue((*)(datum(l), datum(r)), l.tape, MatrixMultiply(l, r))
end

function (*)(ldatum::Matrix{W}, r::GenMatrix) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, MatrixMultiply(l, r))
end

function (*)(l::GenMatrix, rdatum::Real)
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, MatrixMultiply(l, r))
end

function propagate(op::MatrixMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenMatrix, W<:Real}
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end

# matrix * vector
function (*)(l::GenMatrix, r::GenVector)
    makeGenValue((*)(datum(l), datum(r)), l.tape, MatrixMultiply(l, r))
end

function (*)(ldatum::Matrix{W}, r::GenVector) where W<:Real
    l = makeGenValue(ldatum, r.tape)
    makeGenValue((*)(ldatum, datum(r)), l.tape, MatrixMultiply(l, r))
end

function (*)(l::GenMatrix, rdatum::Vector{W}) where W<:Real
    r = makeGenValue(rdatum, l.tape)
    makeGenValue((*)(datum(l), rdatum), l.tape, MatrixMultiply(l, r))
end

function propagate(op::MatrixMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenMatrix, U<:GenVector, W<:Real}
    op.left.adj += adj * op.right.datum' # TODO ?
    op.right.adj += op.left.datum' * adj
end

# matrix .* vector (broadcast)
# TODO not implemented yet

# vector .* matrix (broadcast)
# TODO not implemented yet


# ---- unary plus ----
@generate_unary_operator_type(UnaryPlus)

# +scalar
function (+)(l::GenScalar)
    makeGenValue((+)(datum(l)), l.tape, UnaryPlus(l))
end

# +vector
function (+)(l::GenVector)
    makeGenValue((+)(datum(l)), l.tape, UnaryPlus(l))
end

# +matrix
function (+)(l::GenMatrix)
    makeGenValue((+)(datum(l)), l.tape, UnaryPlus(l))
end

function propagate(op::UnaryPlus, datum::T, adj::U) where {T,U}
    op.arg.adj += adj
end


# ---- unary minus ----
@generate_unary_operator_type(UnaryMinus)

# -scalar
function (-)(l::GenScalar)
    makeGenValue((-)(datum(l)), l.tape, UnaryMinus(l))
end

# -vector
function (-)(l::GenVector)
    makeGenValue((-)(datum(l)), l.tape, UnaryMinus(l))
end

# -matrix
function (-)(l::GenMatrix)
    makeGenValue((-)(datum(l)), l.tape, UnaryMinus(l))
end

function propagate(op::UnaryMinus, datum::T, adj::U) where {T,U}
    op.arg.adj -= adj
end


# ---- exp ----
@generate_unary_operator_type(Exp)

# exp(scalar)
function exp(l::GenScalar)
    makeGenValue(exp(datum(l)), l.tape, Exp(l))
end

# exp(vector)
function broadcast(::typeof(exp), l::GenVector)
    makeGenValue(exp.(datum(l)), l.tape, Exp(l))
end

# exp(matrix)
function broadcast(::typeof(exp), l::GenMatrix)
    makeGenValue(exp.(datum(l)), l.tape, Exp(l))
end

function propagate(op::Exp, datum::T, adj::U) where {T,U}
    op.arg.adj += adj .* datum
end


# ---- log ----
@generate_unary_operator_type(Log)

# log(scalar)
function log(l::GenScalar)
    makeGenValue(log(datum(l)), l.tape, Log(l))
end

# log(vector)
function broadcast(::typeof(log), l::GenVector)
    makeGenValue(log.(datum(l)), l.tape, Log(l))
end

# log(matrix)
function broadcast(::typeof(log), l::GenMatrix)
    makeGenValue(log.(datum(l)), l.tape, Log(l))
end

function propagate(op::Log, datum::T, adj::U) where {T,U}
    println(op.arg.tapeIdx)
    op.arg.adj += adj ./ op.arg.datum
end



# ---- lgamma ----
@generate_unary_operator_type(LogGamme)
