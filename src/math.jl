function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr - min_arr)))
end

# TODO make differentiable https://github.com/probcomp/Gen.jl/issues/65

export logsumexp
