################################################################
# Generic custom proposal sampling importance resampling (SIR) #
################################################################
#
# generate!(sir_generator, (num_samples, target_args, proposal_args, constraints), trace)
#
# trace is an AtomicTrace that stores a value that is a trace of target.
# trace is expected to have constraints matching those of the given constraints at all times.
# trace is expected to have constraints for the latent variables at all times.
#
# constraints is a trace of target with some choices constrained (represents the observations)

struct SIRGenerator{T} <: AtomicGenerator{T}
    # target trace type is T
    target::Generator{T}
    mapping::Dict
    composition::Generator{T}
end

function SIRGenerator(target::Generator, proposal::Generator, mapping::Dict)
    composition = compose(target, proposal, mapping)
    SIRGenerator(target, mapping, composition)
end

function generate!(g::SIRGenerator{T}, args::Tuple{Int, Tuple, Tuple, T}, trace::AtomicTrace{T}) where {T}
    (num, target_args, proposal_args, constraints) = args
    traces = Vector{T}(num)
    scores = Vector{Float64}(num)
    local chosen::Int
    local chosen_joint_score::Float64

    #### Run SIR forward ####
    if (trace.mode == propose) || (trace.mode == record)

        for i=1:num
            traces[i] = deepcopy(constraints)

            # score_i = p(x_i, constraints) / q(x_i) for x_i ~ q(x)
            (scores[i], _) = generate!(g.composition, ((target_args, proposal_args)), traces[i])
        end

        # k ~ categorical(normalized weights)
        chosen = categorical_log(scores)

        # p(x_k, constraints), which is needed for the score
        for (_, (p_addr, _)) in g.mapping
            constrain!(traces[chosen], p_addr, traces[chosen][p_addr])
        end
        (chosen_joint_score, _) = generate!(g.target, target_args, traces[chosen])

        # set output value x_k
        # NOTE: the trace that is returned has both the original constraints and latents constrained
        trace.value = Nullable(traces[chosen])

    #### Run conditional SIR ####
    elseif trace.mode == constrain

        # k ~ uniform(1..num)
        chosen = uniform_discrete(1, num)

        # NOTE: the given trace must have hypothesis AND observations already constrained
        # TODO: we do not check this---behavior depends on the given output trace's constraints
        traces[chosen] = value(trace, ())
        (chosen_joint_score, _) = generate!(g.target, target_args, traces[chosen])
    
        for i=1:num
            if i != chosen
                traces[i] = deepcopy(constraints)
            end

            # score_i = p(x_i, constraints) / q(x_i)
            # if chosen: for x_i and constraints fixed
            # if not chosen: for x_i ~ q(x)
            (scores[i], _) = generate!(g.composition, ((target_args, proposal_args)), traces[i])
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

export SIRGenerator
