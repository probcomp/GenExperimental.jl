#########################################
# Nested inference generator combinator #
#########################################

struct NestedInferenceGenerator{T} <: Generator{T}
    p::Generator{T}
    q::Generator
    
    # mapping from p_address to q_address
    # NOTE: this is different from before
    mapping::Dict
end

empty_trace(g::NestedInferenceGenerator) = empty_trace(g.p)

function nested(p::Generator{T}, q::Generator, mapping::Dict) where {T}
    NestedInferenceGenerator(p, q, mapping)
end

function regenerate!(g::NestedInferenceGenerator{T}, args::Tuple, outputs,
                     conditions, trace::T) where {T}
    (p_args, q_args) = args
    q_trace = empty_trace(g.q)
    q_outputs = AddressTrie()
    q_conditions = AddressTrie()

    for (p_addr, q_addr) in g.mapping
        if (p_addr in outputs || p_addr in conditions)

            # any addresses in the query that are mapped become conditions on q
            push!(q_conditions, q_addr)
            q_trace[q_addr] = trace[p_addr]
        else

            # any mapped addresses that aren't in the query become outputs of q
            push!(q_outputs, q_addr)
        end
    end

    (q_score, q_retval) = simulate!(g.q, q_args, q_outputs, q_conditions, q_trace)

    # NOTE: room for performance optimization here, in avoiding deepcopy
    p_outputs = deepcopy(outputs)

    # register all the values proposed from q as outputs of p
    for (p_addr, q_addr) in g.mapping
        if q_addr in q_outputs
            @assert !(p_addr in outputs)
            push!(p_outputs, p_addr)
            trace[p_addr] = q_trace[q_addr]
        end
    end

    (p_score, p_retval) = regenerate!(g.p, p_args, p_outputs, conditions, trace)

    score = p_score - q_score
    (score, p_retval)
end


function simulate!(g::NestedInferenceGenerator{T}, args::Tuple, outputs,
                   conditions, trace::T) where {T}
    (p_args, q_args) = args

    # NOTE: room for performance optimization here, in avoiding deepcopy
    p_outputs = deepcopy(outputs)
    for (p_addr, _) in g.mapping

        # a mapped address that is a condition stays a condition on p
        # a mapped address that is an output stay an output on p
        if !(p_addr in outputs || p_addr in conditions)

            # a mapped address that does not appear in the query becomes an
            # output of p
            push!(p_outputs, p_addr)
        end
    end

    (p_score, p_retval) = simulate!(g.p, p_args, p_outputs, conditions, trace)

    q_trace = empty_trace(g.q)
    q_outputs = AddressTrie()
    q_conditions = AddressTrie()

    for (p_addr, q_addr) in g.mapping
        if (p_addr in outputs || p_addr in conditions)
            push!(q_conditions, q_addr)
        else
            push!(q_outputs, q_addr)
        end
        q_trace[q_addr] = trace[p_addr]
    end

    (q_score, q_retval) = regenerate!(g.q, q_args, q_outputs, q_conditions, q_trace)

    score = p_score - q_score
    (score, p_retval)
end

export nested
