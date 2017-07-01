import Base.broadcast
import Base.+
import Base.-
import Base./
import Base.*

# code generation for binary operators

macro generate_binary_node_type(node_type)
    eval(quote
           struct $(node_type){T,U} <: AbstractOperator
                left::T
                right::U
            end
        end)
end

macro generate_gen_binary_operator(op, node_type, a_type, b_type)
    eval(quote
            function ($op)(l::$a_type, r::$b_type)
                makeGenValue(($op)(datum(l), datum(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_gen_concrete_binary_operator(op, node_type, a_type, b_type)
    eval(quote
            function ($op)(ldatum::$a_type, r::$b_type)
                l = makeGenValue(ldatum, r.tape)
                makeGenValue(($op)(ldatum, datum(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_concrete_gen_binary_operator(op, node_type, a_type, b_type)
    eval(quote
            function ($op)(l::$a_type, rdatum::$b_type)
                r = makeGenValue(rdatum, k.tape)
                makeGenValue(($op)(datum(l), rdatum), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_gen_binary_broadcast(op, node_type, a_type, b_type)
    eval(quote
            function broadcast(::typeof($op), l::$a_type, r::$b_type)
                makeGenValue(broadcast($op, datum(l), datum(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_gen_concrete_binary_broadcast(op, node_type, a_type, b_type)
    eval(quote
            function broadcast(::typeof($op), ldatum::$a_type, r::$b_type)
                l = makeGenValue(ldatum, r.tape)
                makeGenValue(broadcast($op, ldatum, datum(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_concrete_gen_binary_broadcast(op, node_type, a_type, b_type)
    eval(quote
            function broadcast(::typeof($op), l::$a_type, rdatum::$b_type)
                r = makeGenValue(rdatum, k.tape)
                makeGenValue(broadcast($op, datum(l), rdatum), l.tape, ($node_type)(l, r))
            end
        end)
end

# code generation for unary operators

macro generate_unary_node_type(node_type)
    eval(quote
           struct $(node_type){T} <: AbstractOperator
                arg::T
            end
        end)
end

macro generate_gen_unary_operator(op, node_type, a_type)
    eval(quote
            function ($op)(l::$a_type)
                makeGenValue(($op)(datum(l)), l.tape, ($node_type)(l))
            end
        end)
end

macro generate_gen_unary_broadcast(op, node_type, a_type)
    eval(quote
            function broadcast(::typeof($op), l::$a_type)
                makeGenValue(broadcast($op, datum(l)), l.tape, ($node_type)(l))
            end
        end)
end


# create Gen value from a concrete value

makeGenValue(datum::Real, tape::Tape) = GenScalar(datum, tape, Input())
makeGenValue(datum::Real, tape::Tape, op::AbstractOperator) = GenScalar(datum, tape, op)

makeGenValue(datum::Vector{W}, tape::Tape) where W<:Real = GenColumnVector(datum, tape, Input())
makeGenValue(datum::Vector{W}, tape::Tape, op::AbstractOperator) where W<:Real = GenColumnVector(datum, tape, op)

makeGenValue(datum::RowVector{W,Vector{W}}, tape::Tape) where W<:Real = GenRowVector(datum, tape, Input())
makeGenValue(datum::RowVector{W,Vector{W}}, tape::Tape, op::AbstractOperator) where W<:Real = GenRowVector(datum, tape, op)

makeGenValue(datum::Matrix{W}, tape::Tape) where W<:Real = GenMatrix(datum, tape, Input())
makeGenValue(datum::Matrix{W}, tape::Tape, op::AbstractOperator) where W<:Real = GenMatrix(datum, tape, op)


# NOTES:
# adj is always based on Float64
# the pattern is to define the opertaors including combination of a concrete
# datum and a GenValue, and then define one backpropagation method


## ---- addition ----
#
#@generate_binary_node_type(Add)
#@generate_binary_operators(+, Add)
#@generate_binary_broadcasts(+, Add)
#
## scalar + scalar
#function propagate(op::Add{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    #op.left.adj += adj
    #op.right.adj += adj
#end
#
## scalar + vector
#function propagate(op::Add{T,U}, datum::ColumnOrRowVector{W}, adj::ColumnOrRowVector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    #op.left.adj += sum(adj)
    #op.right.adj += adj
#end
#
## vector + scalar
#function propagate(op::Add{T,U}, datum::ColumnOrRowVector{W}, adj::ColumnOrRowVector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    #op.left.adj += adj
    #op.right.adj += sum(adj)
#end
#
## column vector + column vector
#function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    #op.left.adj += adj
    #op.right.adj += adj
#end
#
## row vector + row vector
#function propagate(op::Add{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    #op.left.adj += adj
    #op.right.adj += adj
#end
#
## row vector .+ column vector
## TODO not implemented yet
#
## column vector .+ row vector
## TODO not implemented yet
#
## scalar + matrix
## TODO not implemented yet
#
## matrix + scalar
## TODO not implemented yet
#
## matrix .+ vector (broadcast)
## TODO not implemented yet
#
## vector .+ matrix (broadcast)
## TODO not implemented yet
#
## matrix + matrix
## TODO not implemented yet
#
#
## ---- subtraction ----
#@generate_binary_node_type(Subtract)
#@generate_binary_operators(-, Subtract)
#@generate_binary_broadcasts(-, Subtract)
#
## scalar - scalar
#function propagate(op::Subtract{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    #op.left.adj += adj
    #op.right.adj -= adj
#end
#
## scalar - vector
#function propagate(op::Subtract{T,U}, datum::ColumnOrRowVector{W}, adj::ColumnOrRowVector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    #op.left.adj += sum(adj)
    #op.right.adj -= adj
#end
#
## vector - scalar
#function propagate(op::Subtract{T,U}, datum::ColumnOrRowVector{W}, adj::ColumnOrRowVector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    #op.left.adj += adj
    #op.right.adj -= sum(adj)
#end
#
## column vector - column vector
#function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    #op.left.adj += adj
    #op.right.adj -= adj
#end
#
## row vector - row vector
#function propagate(op::Subtract{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    #op.left.adj += adj
    #op.right.adj -= adj
#end
#
## row vector .- column vector
## TODO not implemented yet
#
## column vector .- row vector
## TODO not implemented yet
#
## scalar - matrix
## TODO not implemented yet
#
## matrix - scalar
## TODO not implemented yet
#
## matrix .- vector (broadcast)
## TODO not implemented yet
#
## vector .- matrix (broadcast)
## TODO not implemented yet
#
## matrix - matrix
## TODO not implemented yet
#
#
## ---- division ----
#@generate_binary_node_type(Divide)
#@generate_binary_operators(-, Divide)
#@generate_binary_broadcasts(-, Divide)
#
## scalar / scalar
#function propagate(op::Divide{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    #op.left.adj += adj / op.right.datum
    #op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
#end
#
## scalar ./ vector
#function propagate(op::Divide{T,U}, datum::ColumnOrRowVector{W}, adj::ColumnOrRowVector{Float64}) where {T<:GenScalar, U<:GenVector, W<:Real}
    #op.left.adj += sum(adj ./ op.right.datum)
    #op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
#end
#
## vector / scalar
#function propagate(op::Divide{T,U}, datum::ColumnOrRowVector{W}, adj::ColumnOrRowVector{Float64}) where {T<:GenVector, U<:GenScalar, W<:Real}
    #op.left.adj += adj / op.right.datum
    #op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
#end
#
## column vector ./ column vector
#function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    #op.left.adj += adj ./ op.right.datum
    #op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
#end
#
## row vector ./ row vector
#function propagate(op::Divide{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    #op.left.adj += adj ./ op.right.datum
    #op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
#end
        #
## row vector ./ column vector (broadcast)
## TODO not implemented yet
#
## column vector ./ row vector (broadcast)
## TODO not implemented yet
#
## scalar ./ matrix
## TODO not implemented yet
#
## matrix / scalar
## TODO not implemented yet
#
## matrix ./ vector (broadcast)
## TODO not implemented yet
#
## vector ./ matrix (broadcast)
## TODO not implemented yet
#

# ---- element-wise multiplication ----

@generate_binary_node_type(ElementwiseMultiply)

# scalar * scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, Real, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenScalar, Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Real, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj * op.right.datum
    op.right.adj += adj * op.left.datum
end

# scalar * column vector

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, Real, GenColumnVector)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Real, GenColumnVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenColumnVector, W<:Real}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# scalar * row vector

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenRowVector)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, Real, GenRowVector)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Real, GenRowVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenScalar, U<:GenRowVector, W<:Real}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# column vector * scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenColumnVector, Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, Vector{W} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Vector{W} where W<:Real, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenScalar, W<:Real}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# row vector * scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenRowVector, Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, RowVector{W, Vector{W}} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenRowVector, Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, RowVector{W, Vector{W}} where W<:Real, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenScalar, W<:Real}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# column vector .* column vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Vector{W} where W<:Real, GenColumnVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# row vector .* row vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenRowVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, RowVector{W,Vector{W}} where W<:Real, GenRowVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# column vector .* row vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Vector{W} where W<:Real, GenRowVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenColumnVector, U<:GenRowVector, W<:Real}
    op.left.adj += vec(sum(adj .* op.right.datum, 2))
    op.right.adj += RowVector(vec(sum(adj .* op.left.datum, 1)))
end

# row vector .* column vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenRowVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenRowVector, Vector{W} where W<: Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, RowVector{W,Vector{W}} where W<:Real, GenColumnVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenRowVector, U<:GenColumnVector, W<:Real}
    op.left.adj += RowVector(vec(sum(adj .* op.right.datum, 1)))
    op.right.adj += vec(sum(adj .* op.left.datum, 2))
end

# scalar * matrix

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenMatrix)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, Matrix{W} where W<:Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, Real, GenMatrix)

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenScalar, U<:GenMatrix, W<:Real}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# matrix * scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenMatrix, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenMatrix, Real)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, Matrix{W} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenMatrix, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenMatrix, Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Matrix{W} where W<:Real, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenScalar, W<:Real}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# matrix .* matrix
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenMatrix, GenMatrix)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenMatrix, Matrix{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, Matrix{W} where W<:Real, GenMatrix)

function propagate(op::ElementwiseMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenMatrix, W<:Real}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# matrix .* vector (broadcast)
# TODO not implemented yet

# vector .* matrix (broadcast)
# TODO not implemented yet


# ---- matrix multiply ----

@generate_binary_node_type(MatrixMultiply)

# matrix * matrix

@generate_gen_binary_operator(*, MatrixMultiply, GenMatrix, GenMatrix)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenMatrix, Matrix{W} where W<:Real)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, Matrix{W} where W<:Real, GenMatrix)

function propagate(op::MatrixMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenMatrix, W<:Real}
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end

# matrix * column vector = column vector

@generate_gen_binary_operator(*, MatrixMultiply, GenMatrix, GenColumnVector)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenMatrix, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, Matrix{W} where W<:Real, GenColumnVector)

function propagate(op::MatrixMultiply{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenMatrix, U<:GenColumnVector, W<:Real}
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end

# row vector * matrix = row_vector

@generate_gen_binary_operator(*, MatrixMultiply, GenRowVector, GenMatrix)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenRowVector, Matrix{W} where W<:Real)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, RowVector{W,Vector{W}} where W <: Real, GenMatrix)

function propagate(op::MatrixMultiply{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenMatrix, W<:Real}
    # TODO
end

# column vector * row vector (= column vector .* row vector)
# TODO

# row vector * column_vector 
# TODO



## ---- unary plus ----
#@generate_unary_operator_type(UnaryPlus)
#
## + scalar, + column vector, + row vector, + matrix
#function (+)(l::GenValue)
    #makeGenValue((+)(datum(l)), l.tape, UnaryPlus(l))
#end
#
#function propagate(op::UnaryPlus, datum::T, adj::U) where {T,U}
    #op.arg.adj += adj
#end
#
#
## ---- unary minus ----
#@generate_unary_operator_type(UnaryMinus)
#
## - scalar, - column vector, - row vector, - matrix
#function (-)(l::GenValue)
    #makeGenValue((-)(datum(l)), l.tape, UnaryMinus(l))
#end
#
#function propagate(op::UnaryMinus, datum::T, adj::U) where {T,U}
    #op.arg.adj -= adj
#end


# ---- exp ----

import Base.exp
@generate_unary_node_type(Exp)

@generate_gen_unary_operator(exp, Exp, GenScalar)
@generate_gen_unary_broadcast(exp, Exp, GenVector)
@generate_gen_unary_broadcast(exp, Exp, GenMatrix)

function propagate(op::Exp, datum::T, adj::U) where {T,U}
    op.arg.adj += adj .* datum
end

# ---- log ----

import Base.log
@generate_unary_node_type(Log)

@generate_gen_unary_operator(log, Log, GenScalar)
@generate_gen_unary_broadcast(log, Log, GenVector)
@generate_gen_unary_broadcast(log, Log, GenMatrix)

function propagate(op::Log, datum::T, adj::U) where {T,U}
    op.arg.adj += adj ./ op.arg.datum
end

# ---- lgamma ----

import SpecialFunctions.lgamma
import SpecialFunctions.digamma
@generate_unary_node_type(LogGamma)

@generate_gen_unary_operator(lgamma, LogGamma, GenScalar)
@generate_gen_unary_broadcast(lgamma, LogGamma, GenVector)
@generate_gen_unary_broadcast(lgamma, LogGamma, GenMatrix)

function propagate(op::LogGamma, datum::T, adj::U) where {T,U}
    op.arg.adj += adj .* digamma.(op.arg.datum)
end


#
## ---- sum ----
#import Base.sum
#@generate_unary_operator_type(Sum)
#
## sum(scalar)
#function sum(l::GenScalar)
    #makeGenValue(sum(datum(l)), l.tape, Sum(l))
#end
#
## sum(vector)
#function sum(l::GenColumnVector)
    #makeGenValue(sum(datum(l)), l.tape, Sum(l))
#end
#
## sum(matrix)
#function sum(l::GenMatrix)
    #makeGenValue(sum(datum(l)), l.tape, Sum(l))
#end
#
#function propagate(op::Sum, datum::T, adj::U) where {T,U}
    #op.arg.adj += adj
#end


# ---- transpose ----

import Base.transpose
import Base.ctranspose
@generate_unary_node_type(Transpose)

# transpose scalar

@generate_gen_unary_operator(transpose, Transpose, GenScalar)
@generate_gen_unary_operator(ctranspose, Transpose, GenScalar)

function propagate(op::Transpose{U}, datum::W, adj::Float64) where {U<:GenScalar, W<:Real}
    op.arg.adj += adj'
end

# transpose matrix

@generate_gen_unary_operator(transpose, Transpose, GenMatrix)
@generate_gen_unary_operator(ctranspose, Transpose, GenMatrix)

function propagate(op::Transpose{U}, datum::Matrix{W}, adj::Matrix{Float64}) where {U<:GenMatrix, W<:Real}
    op.arg.adj += adj'
end

# transpose column vector (becomes row vector)

@generate_gen_unary_operator(transpose, Transpose, GenColumnVector)
@generate_gen_unary_operator(ctranspose, Transpose, GenColumnVector)

function propagate(op::Transpose{U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {U<:GenColumnVector,W<:Real}
    op.arg.adj += adj'
end

# transpose row vector (becomes column vector)

@generate_gen_unary_operator(transpose, Transpose, GenRowVector)
@generate_gen_unary_operator(ctranspose, Transpose, GenRowVector)

function propagate(op::Transpose{U}, datum::Vector{W}, adj::Vector{Float64}) where {U<:GenRowVector,W<:Real}
    op.arg.adj += adj'
end


# TODO handle slice indexing
