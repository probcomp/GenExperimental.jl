struct ResimulationSIRGenerator{T,V} <: AtomicGenerator{V}
    # target trace type is T
    target::Generator{T}

    # the type of the latent
    latent_type::V
    
    # given a T and a V, populate T with values from V (but don't constrain them)
    populate_latents!::Function

    # given a T, constrain the elements corresponding to V
    constrain_latents!::Function

    # given a T, return a V
    extract_latents::Function
end

empty_trace(::ResimulationSIRGenerator{T}) = AtomicTrace(AtomicTrace{T})

function generate!(g::ResimulationSIRGenerator{T}, args::Tuple{Int, Tuple, T},
                   trace::AtomicTrace{T})
    (num, target_args, constraints) = args
    traces = Vector{T}(num)
    scores = Vector{Float64}(num)
    local chosen::Int
    local chosen_joint_score::Float64
    if (trace.mode == propose) || (trace.mode == record)

        #### Run SIR forward ####

        for i=1:num
            traces[i] = empty_trace(g.target)

            # score_i = p(consraints | x_i) for x_i ~ p(x)
            merge!(traces[i], constraints) # OK
            (scores[i], _) = generate!(g.target, target_args, traces[i])
        end

        # k ~ categorical(normalized weights)
        chosen = categorical_log(scores)

        # p(x_k, constraints)
        g.constrain_latents!(traces[chosen]) # so that the call to generate! doesn't overwrite them.
        chosen_joint_score = generate!(g.target, target_args, traces[chosen])

        # set output value x_k
        trace.value = Nullable(extract_latents(traces[chosen]))

    elseif trace.mode == constrain

        #### Run conditional SIR ####

        # x_k - a trace of type T, with constraints for the hypothesis
        chosen_trace = empty_trace(g.target)
        g.populate_latents!(chosen_trace, value(trace)) 

        # p(x_k)
        g.constrain_latents!(chosen_trace)
        (chosen_prior_score, _) = generate!(g.target, target_args, chosen_trace)

        # p(x_k, constraints)
        merge!(traces[i], constraints)
        chosen_joint_score = generate!(g.target, target_args, chosen_trace)

        # k ~ uniform(1..num)
        chosen = uniform_discrete(1, num)
        traces[chosen] = chosen_trace

        # score_k = p(x_k, constraints) / p(x_k) = p(constraints | x_k)
        scores[chosen] = chosen_joint_score - chosen_prior_score

        for i=1:num
            if i != chosen
                traces[i] = empty_trace(g.target)

                # score_i = p(constraints | x_i) for x_i ~ p(x)
                merge!(traces[i], constraints)
                (scores[i], _) = generate!(g.target, target_args, traces[i])
            end
        end
    else
        error("mode not implemented: $(trace.mode)")
    end
    score = chosen_joint_score - (logsumexp(scores) - log(num))
    if (trace.mode == propose) || (trace.mode == constrain)
        (score, value(trace))
    elseif trace.mode == record
        (0., value(trace))
    else
        # already checked this above
        @assert false
    end
end

export ResimulationSIRGenerator
export sir
