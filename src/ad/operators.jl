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


# ---- addition ----

@generate_binary_node_type(Add)

# scalar + scalar, scalar .+ scalar

@generate_gen_binary_operator(+, Add, GenScalar, GenScalar)
@generate_gen_concrete_binary_operator(+, Add, GenScalar, Real)
@generate_concrete_gen_binary_operator(+, Add, Real, GenScalar)
@generate_gen_binary_broadcast(+, Add, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(+, Add, GenScalar, Real)
@generate_concrete_gen_binary_broadcast(+, Add, Real, GenScalar)

function propagate(op::Add{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj
    op.right.adj += adj
end

# scalar + column vector, scalar .+ column vector

@generate_gen_binary_operator(+, Add, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_operator(+, Add, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(+, Add, Real, GenColumnVector)
@generate_gen_binary_broadcast(+, Add, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(+, Add, Real, GenColumnVector)

function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenColumnVector, W<:Real}
    op.left.adj += sum(adj)
    op.right.adj += adj
end

# scalar + row vector, scalar .+ row vector

@generate_gen_binary_operator(+, Add, GenScalar, GenRowVector)
@generate_gen_concrete_binary_operator(+, Add, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_operator(+, Add, Real, GenRowVector)
@generate_gen_binary_broadcast(+, Add, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(+, Add, Real, GenRowVector)

function propagate(op::Add{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenScalar, U<:GenRowVector, W<:Real}
    op.left.adj += sum(adj)
    op.right.adj += adj
end

# column vector + scalar, column vector .+ scalar

@generate_gen_binary_operator(+, Add, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(+, Add, GenColumnVector, Real)
@generate_concrete_gen_binary_operator(+, Add, Vector{W} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(+, Add, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(+, Add, GenColumnVector, Real)
@generate_concrete_gen_binary_broadcast(+, Add, Vector{W} where W<:Real, GenScalar)

function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenScalar, W<:Real}
    op.left.adj += adj
    op.right.adj += sum(adj)
end

# row vector + scalar, row vector .+ scalar

@generate_gen_binary_operator(+, Add, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(+, Add, GenRowVector, Real)
@generate_concrete_gen_binary_operator(+, Add, RowVector{W, Vector{W}} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(+, Add, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(+, Add, GenRowVector, Real)
@generate_concrete_gen_binary_broadcast(+, Add, RowVector{W, Vector{W}} where W<:Real, GenScalar)

function propagate(op::Add{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenScalar, W<:Real}
    op.left.adj += adj
    op.right.adj += sum(adj)
end

# column vector + column vector, column vector .+ column vector

@generate_gen_binary_operator(+, Add, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_operator(+, Add, GenColumnVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(+, Add, Vector{W} where W<:Real, GenColumnVector)
@generate_gen_binary_broadcast(+, Add, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenColumnVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(+, Add, Vector{W} where W<:Real, GenColumnVector)

function propagate(op::Add{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    op.left.adj += adj
    op.right.adj += adj
end

# row vector + row vector, row vector .+ row vector
@generate_gen_binary_operator(+, Add, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_operator(+, Add, GenRowVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_operator(+, Add, RowVector{W,Vector{W}} where W<:Real, GenRowVector)
@generate_gen_binary_broadcast(+, Add, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenRowVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(+, Add, RowVector{W,Vector{W}} where W<:Real, GenRowVector)

function propagate(op::Add{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    op.left.adj += adj
    op.right.adj += adj
end

# column vector .+ row vector
# TODO not implemented yet

# row vector .+ column vector
# TODO not implemented yet

# scalar + matrix, scalar .+ matrix

# matrix + scalar, matrix .+ scalar
# TODO not implemented yet

# matrix + matrix, matrix .+ matrix
# TODO not implemented yet

# matrix .+ vector (broadcast)
# TODO not implemented yet

# vector .+ matrix (broadcast)
# TODO not implemented yet


# ---- subtraction ----

@generate_binary_node_type(Subtract)

# scalar - scalar, scalar .- scalar

@generate_gen_binary_operator(-, Subtract, GenScalar, GenScalar)
@generate_gen_concrete_binary_operator(-, Subtract, GenScalar, Real)
@generate_concrete_gen_binary_operator(-, Subtract, Real, GenScalar)
@generate_gen_binary_broadcast(-, Subtract, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenScalar, Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, Real, GenScalar)

function propagate(op::Subtract{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj
    op.right.adj -= adj
end

# scalar - column vector, scalar .- column vector

@generate_gen_binary_operator(-, Subtract, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(-, Subtract, Real, GenColumnVector)
@generate_gen_binary_broadcast(-, Subtract, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, Real, GenColumnVector)

function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenColumnVector, W<:Real}
    op.left.adj += sum(adj)
    op.right.adj -= adj
end

# scalar - row vector, scalar .- row vector

@generate_gen_binary_operator(-, Subtract, GenScalar, GenRowVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_operator(-, Subtract, Real, GenRowVector)
@generate_gen_binary_broadcast(-, Subtract, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, Real, GenRowVector)

function propagate(op::Subtract{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenScalar, U<:GenRowVector, W<:Real}
    op.left.adj += sum(adj)
    op.right.adj -= adj
end

# column vector - scalar, column vector .- scalar

@generate_gen_binary_operator(-, Subtract, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(-, Subtract, GenColumnVector, Real)
@generate_concrete_gen_binary_operator(-, Subtract, Vector{W} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(-, Subtract, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenColumnVector, Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, Vector{W} where W<:Real, GenScalar)

function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenScalar, W<:Real}
    op.left.adj += adj
    op.right.adj -= sum(adj)
end

# row vector - scalar, row vector .- scalar

@generate_gen_binary_operator(-, Subtract, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(-, Subtract, GenRowVector, Real)
@generate_concrete_gen_binary_operator(-, Subtract, RowVector{W, Vector{W}} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(-, Subtract, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenRowVector, Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, RowVector{W, Vector{W}} where W<:Real, GenScalar)

function propagate(op::Subtract{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenScalar, W<:Real}
    op.left.adj += adj
    op.right.adj -= sum(adj)
end

# column vector - column vector, column vector .- column vector

@generate_gen_binary_operator(-, Subtract, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenColumnVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(-, Subtract, Vector{W} where W<:Real, GenColumnVector)
@generate_gen_binary_broadcast(-, Subtract, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenColumnVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, Vector{W} where W<:Real, GenColumnVector)

function propagate(op::Subtract{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    op.left.adj += adj
    op.right.adj -= adj
end

# row vector - row vector, row vector .- row vector
@generate_gen_binary_operator(-, Subtract, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenRowVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_operator(-, Subtract, RowVector{W,Vector{W}} where W<:Real, GenRowVector)
@generate_gen_binary_broadcast(-, Subtract, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenRowVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(-, Subtract, RowVector{W,Vector{W}} where W<:Real, GenRowVector)

function propagate(op::Subtract{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    op.left.adj += adj
    op.right.adj -= adj
end

# column vector .- row vector
# TODO not implemented yet

# row vector .- column vector
# TODO not implemented yet

# scalar - matrix, scalar .- matrix

# matrix - scalar, matrix .- scalar
# TODO not implemented yet

# matrix - matrix, matrix .- matrix
# TODO not implemented yet

# matrix .- vector (broadcast)
# TODO not implemented yet

# vector .- matrix (broadcast)
# TODO not implemented yet



# ---- division ----

@generate_binary_node_type(Divide)

# scalar / scalar, scalar ./ scalar

@generate_gen_binary_operator(/, Divide, GenScalar, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenScalar, Real)
@generate_concrete_gen_binary_operator(/, Divide, Real, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Real, GenScalar)

function propagate(op::Divide{T,U}, datum::Real, adj::Float64) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end

# scalar ./ column vector

@generate_gen_binary_broadcast(/, Divide, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Real, GenColumnVector)

function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenScalar, U<:GenColumnVector, W<:Real}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# scalar ./ row vector

@generate_gen_binary_broadcast(/, Divide, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Real, GenRowVector)

function propagate(op::Divide{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenScalar, U<:GenRowVector, W<:Real}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# column vector / scalar, column vector ./ scalar

@generate_gen_binary_operator(/, Divide, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenColumnVector, Real)
@generate_concrete_gen_binary_operator(/, Divide, Vector{W} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenColumnVector, Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Vector{W} where W<:Real, GenScalar)

function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenScalar, W<:Real}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# row vector / scalar, row vector ./ scalar

@generate_gen_binary_operator(/, Divide, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenRowVector, Real)
@generate_concrete_gen_binary_operator(/, Divide, RowVector{W, Vector{W}} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenRowVector, Real)
@generate_concrete_gen_binary_broadcast(/, Divide, RowVector{W, Vector{W}} where W<:Real, GenScalar)

function propagate(op::Divide{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenScalar, W<:Real}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# column vector ./ column vector

@generate_gen_binary_broadcast(/, Divide, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenColumnVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Vector{W} where W<:Real, GenColumnVector)

function propagate(op::Divide{T,U}, datum::Vector{W}, adj::Vector{Float64}) where {T<:GenColumnVector, U<:GenColumnVector, W<:Real}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# row vector ./ row vector

@generate_gen_binary_broadcast(/, Divide, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenRowVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_broadcast(/, Divide, RowVector{W,Vector{W}} where W<:Real, GenRowVector)

function propagate(op::Divide{T,U}, datum::RowVector{W,Vector{W}}, adj::RowVector{Float64,Vector{Float64}}) where {T<:GenRowVector, U<:GenRowVector, W<:Real}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# column vector ./ row vector
# TODO not implemented yet

# row vector ./ column vector
# TODO not implemented yet

# scalar ./ matrix

@generate_gen_binary_broadcast(/, Divide, GenScalar, GenMatrix)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, Matrix{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Real, GenMatrix)

function propagate(op::Divide{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenScalar, U<:GenMatrix, W<:Real}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# matrix / scalar, matrix ./ scalar

@generate_gen_binary_operator(/, Divide, GenMatrix, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenMatrix, Real)
@generate_concrete_gen_binary_operator(/, Divide, Matrix{W} where W<:Real, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenMatrix, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenMatrix, Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Matrix{W} where W<:Real, GenScalar)

function propagate(op::Divide{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenScalar, W<:Real}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# matrix ./ matrix
@generate_gen_binary_broadcast(/, Divide, GenMatrix, GenMatrix)
@generate_gen_concrete_binary_broadcast(/, Divide, GenMatrix, Matrix{W} where W<:Real)
@generate_concrete_gen_binary_broadcast(/, Divide, Matrix{W} where W<:Real, GenMatrix)

function propagate(op::Divide{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenMatrix, U<:GenMatrix, W<:Real}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# matrix ./ vector (broadcast)
# TODO not implemented yet

# vector ./ matrix (broadcast)
# TODO not implemented yet



# ---- element-wise multiplication ----

@generate_binary_node_type(ElementwiseMultiply)

# scalar * scalar, scalar .* scalar

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

# scalar * column vector, scalar .* column vector

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

# scalar * row vector, scalar .* row vector

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

# column vector * scalar, column vector .* scalar

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

# row vector * scalar, row vector .* scalar

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

# matrix * scalar, matrix .* scalar

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

# column vector * row vector (outer product)
# note: this is equivalent to column vector .* row vector
@generate_gen_binary_operator(*, MatrixMultiply, GenColumnVector, GenRowVector)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenColumnVector, RowVector{W,Vector{W}} where W<:Real)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, Vector{W} where W <: Real, GenRowVector)

function propagate(op::MatrixMultiply{T,U}, datum::Matrix{W}, adj::Matrix{Float64}) where {T<:GenColumnVector, U<:GenRowVector, W<:Real}
    op.left.adj += vec(sum(adj .* op.right.datum, 2))
    op.right.adj += RowVector(vec(sum(adj .* op.left.datum, 1)))
end

# row vector * column_vector = scalar (inner product)
@generate_gen_binary_operator(*, MatrixMultiply, GenRowVector, GenColumnVector)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenRowVector, Vector{W} where W<:Real)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, RowVector{W,Vector{W}} where W <: Real, GenColumnVector)

function propagate(op::MatrixMultiply{T,U}, datum::Real, adj::Float64) where {T<:GenRowVector, U<:GenColumnVector}
    op.left.adj += adj * op.right.datum'
    op.right.adj += adj * op.left.datum'
end


# ---- unary plus ----

@generate_unary_node_type(UnaryPlus)
@generate_gen_unary_operator(+, UnaryPlus, GenValue)

function propagate(op::UnaryPlus, datum::T, adj::U) where {T,U}
    op.arg.adj += adj
end

# ---- unary minus ----

@generate_unary_node_type(UnaryMinus)
@generate_gen_unary_operator(-, UnaryMinus, GenValue)

function propagate(op::UnaryMinus, datum::T, adj::U) where {T,U}
    op.arg.adj -= adj
end

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

# TODO handle slice indexing. This might be handled currently but inefficiently by Julia's
# AbstractArray indexing facilities, which will probably produce an array of GenScalars
