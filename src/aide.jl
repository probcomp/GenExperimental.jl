"""
Estimate the KL divergence between two generators with the same output type.

AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms
Marco F. Cusumano-Towner, Vikash K. Mansinghka
https://arxiv.org/abs/1705.07224
"""
function aide(p::AtomicGenerator{T}, p_args::Tuple,
              q::AtomicGenerator{T}, q_args::Tuple) where {T}
    generator = nested(p, q, Dict([() => ()]))
    outputs = AddressTrie()
    conditions = AddressTrie()
    (score, _) = simulate!(generator, (p_args, q_args), outputs, conditions, AtomicTrace(T))
    return score
end

function aide(p::Generator{T}, p_args::Tuple, q::Generator{U}, q_args::Tuple, mapping::Dict) where {T, U}
    generator = nested(p, q, mapping)
    outputs = AddressTrie()
    conditions = AddressTrie()
    trace = empty_trace(generator)
    (score, _) = simulate!(generator, (p_args, q_args), outputs, conditions, trace)
    return score
end

export aide
