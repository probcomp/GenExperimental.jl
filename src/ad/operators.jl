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
                makeGenValue(($op)(concrete(l), concrete(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_concrete_gen_binary_operator(op, node_type, a_type, b_type)
    eval(quote
            function ($op)(ldatum::$a_type, r::$b_type)
                l = makeGenValue(ldatum, r.tape)
                makeGenValue(($op)(ldatum, concrete(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_gen_concrete_binary_operator(op, node_type, a_type, b_type)
    eval(quote
            function ($op)(l::$a_type, rdatum::$b_type)
                r = makeGenValue(rdatum, l.tape)
                makeGenValue(($op)(concrete(l), rdatum), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_gen_binary_broadcast(op, node_type, a_type, b_type)
    eval(quote
            function ewise(::typeof($op), l::$a_type, r::$b_type)
                makeGenValue(broadcast($op, concrete(l), concrete(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_concrete_gen_binary_broadcast(op, node_type, a_type, b_type)
    eval(quote
            function ewise(::typeof($op), ldatum::$a_type, r::$b_type)
                l = makeGenValue(ldatum, r.tape)
                makeGenValue(broadcast($op, ldatum, concrete(r)), l.tape, ($node_type)(l, r))
            end
        end)
end

macro generate_gen_concrete_binary_broadcast(op, node_type, a_type, b_type)
    eval(quote
            function ewise(::typeof($op), l::$a_type, rdatum::$b_type)
                r = makeGenValue(rdatum, l.tape)
                makeGenValue(broadcast($op, concrete(l), rdatum), l.tape, ($node_type)(l, r))
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
                makeGenValue(($op)(concrete(l)), l.tape, ($node_type)(l))
            end
        end)
end

macro generate_gen_unary_broadcast(op, node_type, a_type)
    eval(quote
            function ewise(::typeof($op), l::$a_type)
                makeGenValue(broadcast($op, concrete(l)), l.tape, ($node_type)(l))
            end
        end)
end



# NOTES:
# adj is always based on ScalarAdjoint
# the pattern is to define the opertaors including combination of a concrete
# datum and a GenValue, and then define one backpropagation method


# ---- addition ----

@generate_binary_node_type(Add)

# scalar + scalar, scalar .+ scalar

@generate_gen_binary_operator(+, Add, GenScalar, GenScalar)
@generate_gen_concrete_binary_operator(+, Add, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_operator(+, Add, ConcreteScalar, GenScalar)
@generate_gen_binary_broadcast(+, Add, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(+, Add, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteScalar, GenScalar)

function propagate(op::Add{T,U}, datum::ConcreteScalar, adj::ScalarAdjoint) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj
    op.right.adj += adj
end

# scalar + column vector, scalar .+ column vector

@generate_gen_binary_operator(+, Add, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_operator(+, Add, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(+, Add, ConcreteScalar, GenColumnVector)
@generate_gen_binary_broadcast(+, Add, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteScalar, GenColumnVector)

function propagate(op::Add{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenScalar, U<:GenColumnVector}
    op.left.adj += sum(adj)
    op.right.adj += adj
end

# scalar + row vector, scalar .+ row vector

@generate_gen_binary_operator(+, Add, GenScalar, GenRowVector)
@generate_gen_concrete_binary_operator(+, Add, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_operator(+, Add, ConcreteScalar, GenRowVector)
@generate_gen_binary_broadcast(+, Add, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteScalar, GenRowVector)

function propagate(op::Add{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenScalar, U<:GenRowVector}
    op.left.adj += sum(adj)
    op.right.adj += adj
end

# column vector + scalar, column vector .+ scalar

@generate_gen_binary_operator(+, Add, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(+, Add, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(+, Add, ConcreteColumnVector, GenScalar)
@generate_gen_binary_broadcast(+, Add, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(+, Add, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteColumnVector, GenScalar)

function propagate(op::Add{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenScalar}
    op.left.adj += adj
    op.right.adj += sum(adj)
end

# row vector + scalar, row vector .+ scalar

@generate_gen_binary_operator(+, Add, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(+, Add, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(+, Add, ConcreteRowVector, GenScalar)
@generate_gen_binary_broadcast(+, Add, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(+, Add, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteRowVector, GenScalar)

function propagate(op::Add{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenScalar}
    op.left.adj += adj
    op.right.adj += sum(adj)
end

# column vector + column vector, column vector .+ column vector

@generate_gen_binary_operator(+, Add, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_operator(+, Add, GenColumnVector, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(+, Add, ConcreteColumnVector, GenColumnVector)
@generate_gen_binary_broadcast(+, Add, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenColumnVector, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteColumnVector, GenColumnVector)

function propagate(op::Add{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenColumnVector}
    op.left.adj += adj
    op.right.adj += adj
end

# row vector + row vector, row vector .+ row vector
@generate_gen_binary_operator(+, Add, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_operator(+, Add, GenRowVector, ConcreteRowVector)
@generate_concrete_gen_binary_operator(+, Add, ConcreteRowVector, GenRowVector)
@generate_gen_binary_broadcast(+, Add, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(+, Add, GenRowVector, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(+, Add, ConcreteRowVector, GenRowVector)

function propagate(op::Add{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenRowVector}
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
@generate_gen_concrete_binary_operator(-, Subtract, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteScalar, GenScalar)
@generate_gen_binary_broadcast(-, Subtract, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteScalar, GenScalar)

function propagate(op::Subtract{T,U}, datum::ConcreteScalar, adj::ScalarAdjoint) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj
    op.right.adj -= adj
end

# scalar - column vector, scalar .- column vector

@generate_gen_binary_operator(-, Subtract, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteScalar, GenColumnVector)
@generate_gen_binary_broadcast(-, Subtract, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteScalar, GenColumnVector)

function propagate(op::Subtract{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenScalar, U<:GenColumnVector}
    op.left.adj += sum(adj)
    op.right.adj -= adj
end

# scalar - row vector, scalar .- row vector

@generate_gen_binary_operator(-, Subtract, GenScalar, GenRowVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteScalar, GenRowVector)
@generate_gen_binary_broadcast(-, Subtract, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteScalar, GenRowVector)

function propagate(op::Subtract{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenScalar, U<:GenRowVector}
    op.left.adj += sum(adj)
    op.right.adj -= adj
end

# column vector - scalar, column vector .- scalar

@generate_gen_binary_operator(-, Subtract, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(-, Subtract, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteColumnVector, GenScalar)
@generate_gen_binary_broadcast(-, Subtract, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteColumnVector, GenScalar)

function propagate(op::Subtract{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenScalar}
    op.left.adj += adj
    op.right.adj -= sum(adj)
end

# row vector - scalar, row vector .- scalar

@generate_gen_binary_operator(-, Subtract, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(-, Subtract, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteRowVector, GenScalar)
@generate_gen_binary_broadcast(-, Subtract, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteRowVector, GenScalar)

function propagate(op::Subtract{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenScalar}
    op.left.adj += adj
    op.right.adj -= sum(adj)
end

# column vector - column vector, column vector .- column vector

@generate_gen_binary_operator(-, Subtract, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenColumnVector, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteColumnVector, GenColumnVector)
@generate_gen_binary_broadcast(-, Subtract, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenColumnVector, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteColumnVector, GenColumnVector)

function propagate(op::Subtract{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenColumnVector}
    op.left.adj += adj
    op.right.adj -= adj
end

# row vector - row vector, row vector .- row vector
@generate_gen_binary_operator(-, Subtract, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_operator(-, Subtract, GenRowVector, ConcreteRowVector)
@generate_concrete_gen_binary_operator(-, Subtract, ConcreteRowVector, GenRowVector)
@generate_gen_binary_broadcast(-, Subtract, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(-, Subtract, GenRowVector, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(-, Subtract, ConcreteRowVector, GenRowVector)

function propagate(op::Subtract{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenRowVector}
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
@generate_gen_concrete_binary_operator(/, Divide, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_operator(/, Divide, ConcreteScalar, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteScalar, GenScalar)

function propagate(op::Divide{T,U}, datum::ConcreteScalar, adj::ScalarAdjoint) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj / op.right.datum
    op.right.adj += adj * (-op.left.datum / (op.right.datum * op.right.datum))
end

# scalar ./ column vector

@generate_gen_binary_broadcast(/, Divide, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteScalar, GenColumnVector)

function propagate(op::Divide{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenScalar, U<:GenColumnVector}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# scalar ./ row vector

@generate_gen_binary_broadcast(/, Divide, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteScalar, GenRowVector)

function propagate(op::Divide{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenScalar, U<:GenRowVector}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# column vector / scalar, column vector ./ scalar

@generate_gen_binary_operator(/, Divide, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(/, Divide, ConcreteColumnVector, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteColumnVector, GenScalar)

function propagate(op::Divide{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenScalar}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# row vector / scalar, row vector ./ scalar

@generate_gen_binary_operator(/, Divide, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(/, Divide, ConcreteRowVector, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteRowVector, GenScalar)

function propagate(op::Divide{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenScalar}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# column vector ./ column vector

@generate_gen_binary_broadcast(/, Divide, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenColumnVector, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteColumnVector, GenColumnVector)

function propagate(op::Divide{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenColumnVector}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# row vector ./ row vector

@generate_gen_binary_broadcast(/, Divide, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(/, Divide, GenRowVector, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteRowVector, GenRowVector)

function propagate(op::Divide{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenRowVector}
    op.left.adj += adj ./ op.right.datum
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# column vector ./ row vector
# TODO not implemented yet

# row vector ./ column vector
# TODO not implemented yet

# scalar ./ matrix

@generate_gen_binary_broadcast(/, Divide, GenScalar, GenMatrix)
@generate_gen_concrete_binary_broadcast(/, Divide, GenScalar, ConcreteMatrix)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteScalar, GenMatrix)

function propagate(op::Divide{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenScalar, U<:GenMatrix}
    op.left.adj += sum(adj ./ op.right.datum)
    op.right.adj += adj .* (-op.left.datum ./ (op.right.datum .* op.right.datum))
end

# matrix / scalar, matrix ./ scalar

@generate_gen_binary_operator(/, Divide, GenMatrix, GenScalar)
@generate_gen_concrete_binary_operator(/, Divide, GenMatrix, ConcreteScalar)
@generate_concrete_gen_binary_operator(/, Divide, ConcreteMatrix, GenScalar)
@generate_gen_binary_broadcast(/, Divide, GenMatrix, GenScalar)
@generate_gen_concrete_binary_broadcast(/, Divide, GenMatrix, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteMatrix, GenScalar)

function propagate(op::Divide{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenMatrix, U<:GenScalar}
    op.left.adj += adj / op.right.datum
    op.right.adj += sum(adj .* (-op.left.datum / (op.right.datum * op.right.datum)))
end

# matrix ./ matrix
@generate_gen_binary_broadcast(/, Divide, GenMatrix, GenMatrix)
@generate_gen_concrete_binary_broadcast(/, Divide, GenMatrix, ConcreteMatrix)
@generate_concrete_gen_binary_broadcast(/, Divide, ConcreteMatrix, GenMatrix)

function propagate(op::Divide{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenMatrix, U<:GenMatrix}
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
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteScalar, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenScalar, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenScalar, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteScalar, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteScalar, adj::ScalarAdjoint) where {T<:GenScalar, U<:GenScalar}
    op.left.adj += adj * op.right.datum
    op.right.adj += adj * op.left.datum
end

# scalar * column vector, scalar .* column vector

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteScalar, GenColumnVector)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenScalar, GenColumnVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenScalar, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteScalar, GenColumnVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenScalar, U<:GenColumnVector}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# scalar * row vector, scalar .* row vector

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenRowVector)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteScalar, GenRowVector)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenScalar, GenRowVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenScalar, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteScalar, GenRowVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenScalar, U<:GenRowVector}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# column vector * scalar, column vector .* scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteColumnVector, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteColumnVector, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenScalar}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# row vector * scalar, row vector .* scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenRowVector, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteRowVector, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenRowVector, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenRowVector, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteRowVector, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenScalar}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# column vector .* column vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteColumnVector, GenColumnVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenColumnVector, U<:GenColumnVector}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# row vector .* row vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenRowVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenRowVector, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteRowVector, GenRowVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenRowVector}
    op.left.adj += adj .* op.right.datum
    op.right.adj += adj .* op.left.datum
end

# column vector .* row vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, GenRowVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenColumnVector, ConcreteRowVector)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteColumnVector, GenRowVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenColumnVector, U<:GenRowVector}
    op.left.adj += vec(sum(adj .* op.right.datum, 2))
    op.right.adj += RowVector(vec(sum(adj .* op.left.datum, 1)))
end

# row vector .* column vector

@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenRowVector, GenColumnVector)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenRowVector, ConcreteColumnVector)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteRowVector, GenColumnVector)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenRowVector, U<:GenColumnVector}
    op.left.adj += RowVector(vec(sum(adj .* op.right.datum, 1)))
    op.right.adj += vec(sum(adj .* op.left.datum, 2))
end

# scalar * matrix

@generate_gen_binary_operator(*, ElementwiseMultiply, GenScalar, GenMatrix)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenScalar, ConcreteMatrix)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteScalar, GenMatrix)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenScalar, U<:GenMatrix}
    op.left.adj += sum(adj .* op.right.datum)
    op.right.adj += adj * op.left.datum
end

# matrix * scalar, matrix .* scalar

@generate_gen_binary_operator(*, ElementwiseMultiply, GenMatrix, GenScalar)
@generate_gen_concrete_binary_operator(*, ElementwiseMultiply, GenMatrix, ConcreteScalar)
@generate_concrete_gen_binary_operator(*, ElementwiseMultiply, ConcreteMatrix, GenScalar)
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenMatrix, GenScalar)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenMatrix, ConcreteScalar)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteMatrix, GenScalar)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenMatrix, U<:GenScalar}
    op.left.adj += adj * op.right.datum
    op.right.adj += sum(adj .* op.left.datum)
end

# matrix .* matrix
@generate_gen_binary_broadcast(*, ElementwiseMultiply, GenMatrix, GenMatrix)
@generate_gen_concrete_binary_broadcast(*, ElementwiseMultiply, GenMatrix, ConcreteMatrix)
@generate_concrete_gen_binary_broadcast(*, ElementwiseMultiply, ConcreteMatrix, GenMatrix)

function propagate(op::ElementwiseMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenMatrix, U<:GenMatrix}
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
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenMatrix, ConcreteMatrix)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, ConcreteMatrix, GenMatrix)

function propagate(op::MatrixMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenMatrix, U<:GenMatrix}
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end

# matrix * column vector = column vector

@generate_gen_binary_operator(*, MatrixMultiply, GenMatrix, GenColumnVector)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenMatrix, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, ConcreteMatrix, GenColumnVector)

function propagate(op::MatrixMultiply{T,U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {T<:GenMatrix, U<:GenColumnVector}
    op.left.adj += adj * op.right.datum'
    op.right.adj += op.left.datum' * adj
end

# row vector * matrix = row_vector

@generate_gen_binary_operator(*, MatrixMultiply, GenRowVector, GenMatrix)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenRowVector, ConcreteMatrix)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, ConcreteRowVector, GenMatrix)

function propagate(op::MatrixMultiply{T,U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {T<:GenRowVector, U<:GenMatrix}
    # TODO
end

# column vector * row vector (outer product)
# note: this is equivalent to column vector .* row vector
@generate_gen_binary_operator(*, MatrixMultiply, GenColumnVector, GenRowVector)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenColumnVector, ConcreteRowVector)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, ConcreteColumnVector, GenRowVector)

function propagate(op::MatrixMultiply{T,U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {T<:GenColumnVector, U<:GenRowVector}
    op.left.adj += vec(sum(adj .* op.right.datum, 2))
    op.right.adj += RowVector(vec(sum(adj .* op.left.datum, 1)))
end

# row vector * column_vector = scalar (inner product)
@generate_gen_binary_operator(*, MatrixMultiply, GenRowVector, GenColumnVector)
@generate_gen_concrete_binary_operator(*, MatrixMultiply, GenRowVector, ConcreteColumnVector)
@generate_concrete_gen_binary_operator(*, MatrixMultiply, ConcreteRowVector, GenColumnVector)

function propagate(op::MatrixMultiply{T,U}, datum::ConcreteScalar, adj::ScalarAdjoint) where {T<:GenRowVector, U<:GenColumnVector}
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
@generate_gen_unary_broadcast(exp, Exp, Union{GenVector, GenMatrix})

function propagate(op::Exp, datum::T, adj::U) where {T,U}
    op.arg.adj += adj .* datum
end

# ---- log ----

import Base.log
@generate_unary_node_type(Log)

@generate_gen_unary_operator(log, Log, GenScalar)
@generate_gen_unary_broadcast(log, Log, Union{GenVector, GenMatrix})

function propagate(op::Log, datum::T, adj::U) where {T,U}
    op.arg.adj += adj ./ op.arg.datum
end

# ---- lgamma ----

import SpecialFunctions.lgamma
import SpecialFunctions.digamma
@generate_unary_node_type(LogGamma)

@generate_gen_unary_operator(lgamma, LogGamma, GenScalar)
@generate_gen_unary_broadcast(lgamma, LogGamma, Union{GenVector, GenMatrix})

function propagate(op::LogGamma, datum::T, adj::U) where {T,U}
    op.arg.adj += adj .* digamma.(op.arg.datum)
end


# ---- sum ----
import Base.sum
@generate_unary_node_type(Sum)
@generate_gen_unary_operator(sum, Sum, GenValue)

function propagate(op::Sum, datum::T, adj::U) where {T,U}
    op.arg.adj += adj 
end


# ---- transpose ----

import Base.transpose
import Base.ctranspose
@generate_unary_node_type(Transpose)
@generate_gen_unary_operator(transpose, Transpose, GenValue)
@generate_gen_unary_operator(ctranspose, Transpose, GenValue)

# transpose scalar
function propagate(op::Transpose{U}, datum::ConcreteScalar, adj::ScalarAdjoint) where {U<:GenScalar}
    op.arg.adj += adj'
end

# transpose matrix
function propagate(op::Transpose{U}, datum::ConcreteMatrix, adj::MatrixAdjoint) where {U<:GenMatrix}
    op.arg.adj += adj'
end

# transpose column vector (becomes row vector)
function propagate(op::Transpose{U}, datum::ConcreteRowVector, adj::RowVectorAdjoint) where {U<:GenColumnVector}
    op.arg.adj += adj'
end

# transpose row vector (becomes column vector)
function propagate(op::Transpose{U}, datum::ConcreteColumnVector, adj::ColumnVectorAdjoint) where {U<:GenRowVector}
    op.arg.adj += adj'
end

# TODO handle slice indexing. This might be handled currently but inefficiently by Julia's
# AbstractArray indexing facilities, which will probably produce an array of GenScalars


# ---- logsumexp ----
@generate_unary_node_type(LogSumExp)

# generate operator for logsumexp of a GenColumnVector
@generate_gen_unary_operator(logsumexp, LogSumExp, GenColumnVector)

# operator for logsumexp of a general (possibly mixed) array
function logsumexp(arr::Vector{Any})
    tape::Nullable{Tape} = Nullable()
    for el in arr
        if isa(el, GenScalar)
            if isnull(tape)
                tape = el.tape
            else
                if get(tape) != el.tape
                    error("Tapes do not match")
                end
            end
        end
    end
    concrete_arr = map(concrete, arr)
    min_arr = maximum(concrete_arr)
    if isnull(tape)
        min_arr + log(sum(exp.(concrete_arr - min_arr)))
    else
        makeGenValue(min_arr + log(sum(exp.(concrete_arr - min_arr))), get(tape), LogSumExp(arr))
    end
end

function propagate(op::LogSumExp{U}, datum::ConcreteScalar, adj::ConcreteScalar) where {U <: GenColumnVector}
    op.arg.adj += exp.(op.arg.datum - datum) * adj
end

function propagate(op::LogSumExp{Vector{Any}}, datum::ConcreteScalar, adj::ConcreteScalar)
    # operand is a vector where only some elements are GenScalars, and others are not
    for i=1:length(op.arg)
        if isa(op.arg[i], GenScalar)
            op.arg[i].adj += exp(op.arg[i].datum - datum) * adj
        end
    end
end
