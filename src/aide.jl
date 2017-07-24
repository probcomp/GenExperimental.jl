"""
Estimate the KL divergence between two generators with the same output type.

AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms
Marco F. Cusumano-Towner, Vikash K. Mansinghka
https://arxiv.org/abs/1705.07224
"""
function aide(p::AtomicGenerator{T}, p_args::Tuple,
              q::AtomicGenerator{T}, q_args::Tuple) where {T}
    generator = compose(q, p, ())
    (score, _) = generate!(generator, (q_args, p_args), AtomicTrace(T))
    -score
end

export aide
