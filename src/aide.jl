function estimate_elbo(p::AtomicGenerator{T}, p_args::Tuple,
                       q::AtomicGenerator{T}, q_args::Tuple) where {T}
    trace = AtomicTrace(T)
    propose!(trace, (), T)
    (p_log_weight, _) = generate!(p, (p_args), trace)
    constrain!(trace, (), value(trace, ()))
    (q_log_weight, ) = generate!(q, (q_args), trace)
    p_log_weight - q_log_weight
end

export estimate_elbo
