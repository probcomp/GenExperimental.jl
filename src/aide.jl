"""
Estimate the KL divergence between two generators with the same output type.

AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms
Marco F. Cusumano-Towner, Vikash K. Mansinghka
https://arxiv.org/abs/1705.07224
"""
function aide(p::AtomicGenerator{T}, p_args::Tuple,
              q::AtomicGenerator{T}, q_args::Tuple) where {T}
    trace = AtomicTrace(T)
    propose!(trace, (), T)
    (p_log_weight, _) = generate!(p, p_args, trace)
    constrain!(trace, (), trace[()])
    (q_log_weight, _) = generate!(q, q_args, trace)
    p_log_weight - q_log_weight
end

export aide
