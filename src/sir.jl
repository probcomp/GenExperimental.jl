#############################################################
# Generic resimulation sampling importance resampling (SIR) #
#############################################################

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

function generate!(g::ResimulationSIRGenerator{T,V}, args::Tuple{Int, Tuple, T}, trace::AtomicTrace{V})
    (num, target_args, constraints) = args
    traces = Vector{T}(num)
    scores = Vector{Float64}(num)
    local chosen::Int
    local chosen_joint_score::Float64


    #### Run SIR forward ####
    if (trace.mode == propose) || (trace.mode == record)

        for i=1:num
            traces[i] = empty_trace(g.target)

            # score_i = p(consraints | x_i) for x_i ~ p(x)
            merge!(traces[i], constraints)
            (scores[i], _) = generate!(g.target, target_args, traces[i])
        end

        # k ~ categorical(normalized weights)
        chosen = categorical_log(scores)

        # p(x_k, constraints)
        g.constrain_latents!(traces[chosen]) # so that the call to generate! doesn't overwrite them.
        chosen_joint_score = generate!(g.target, target_args, traces[chosen])

        # set output value x_k
        trace.value = Nullable(extract_latents(traces[chosen]))


    #### Run conditional SIR ####
    elseif trace.mode == constrain

        # x_k - a trace of type T, with constraints for the hypothesis
        chosen_trace = empty_trace(g.target)
        g.populate_latents!(chosen_trace, value(trace)) 

        # p(x_k)
        g.constrain_latents!(chosen_trace)
        (chosen_prior_score, _) = generate!(g.target, target_args, chosen_trace)

        # p(x_k, constraints)
        merge!(chosen_trace, constraints)
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


################################################################
# Generic custom proposal sampling importance resampling (SIR) #
################################################################


struct SIRGenerator{T,V} <: AtomicGenerator{V}
    # target trace type is T
    target::Generator{T}

    # produces latents of type V
    proposal::AtomicGenerator{V}

    # given a T and a V, populate T with values from V (but don't constrain them)
    populate_latents!::Function

    # given a T, constrain the elements corresponding to V
    constrain_latents!::Function

    # given a T, return a V
    extract_latents::Function
end

# TODO how can we make use of compose() ?
# TODO do we have the user implement compose() for their pair of generators target and proposal?

function generate!(g::ResimulationSIRGenerator{T,V}, args::Tuple{Int, Tuple, Tuple, T}, trace::AtomicTrace{V})
    (num, target_args, proposal_args, constraints) = args
    traces = Vector{T}(num)
    scores = Vector{Float64}(num)
    local chosen::Int
    local chosen_joint_score::Float64

    #### Run SIR forward ####
    if (trace.mode == propose) || (trace.mode == record)

        for i=1:num
            traces[i] = empty_trace(g.target)

            # score_i = p(x_i, constraints) / q(x_i) for x_i ~ q(x)
            merge!(traces[i], constraints)
            (proposal_score, latents) = generate!(g.proposal, proposal_args, AtomicTrace(V))
            g.populate_latents(traces[i], latents)
            g.constrain_latents(traces[i])
            model_joint_score = generate!(g.target, target_args, traces[i])
            scores[i] = model_joint_score - proposal_score
        end

        # k ~ categorical(normalized weights)
        chosen = categorical_log(scores)

        # p(x_k, constraints)
        chosen_joint_score = generate!(g.target, target_args, traces[chosen])

        # set output value x_k
        trace.value = Nullable(extract_latents(traces[chosen]))

    #### Run conditional SIR ####
    elseif trace.mode == constrain

        # x_k - a trace of type T, with constraints for the hypothesis
        chosen_trace = empty_trace(g.target)
        g.populate_latents!(chosen_trace, value(trace)) 

        # q(x_k)
        (chosen_proposal_score, _) = generate!(g.proposal, proposal_args, trace)

        # p(x_k, constraints)
        merge!(chosen_trace, constraints)
        chosen_joint_score = generate!(g.target, target_args, chosen_trace)

        # k ~ uniform(1..num)
        chosen = uniform_discrete(1, num)
        traces[chosen] = chosen_trace

        # score_k = p(x_k, constraints) / p(x_k) = p(constraints | x_k)
        scores[chosen] = chosen_joint_score - chosen_prior_score

        for i=1:num
            if i != chosen
                traces[i] = empty_trace(g.target)

                # score_i = p(x_i, constraints) / q(x_i) for x_i ~ q(x)
                merge!(traces[i], constraints)
                proposal_trace = AtomicTrace(V)
                propose!(proposal_trace, (), V)
                (proposal_score, latents) = generate!(g.proposal, proposal_args, proposal_trace)
                g.populate_latents(traces[i], latents)
                g.constrain_latents(traces[i])
                model_joint_score = generate!(g.target, target_args, traces[i])
                scores[i] = model_joint_score - proposal_score
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
export SIRGenerator
