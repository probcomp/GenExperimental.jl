function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr - min_arr)))
end

export logsumexp
