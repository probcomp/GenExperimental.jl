using Gen
using Distributions

# TODO: think about a general form for SIR and general SIR regenerator

# the probabilistic model
@program my_model(xs::Vector{Float64}) begin
    slope = normal(0., 1.) ~ "slope"
    intercept = normal(0., 1.) ~ "intercept"
    line = slope * xs + intercept
    ys = map((i) -> normal(line[i], 1.) ~ "y$i", 1:length(xs))
end

# the SIR module 

immutable LinearRegressionParameters
    slope::Float64
    intercept::Float64
end

immutable SIR <: Module{LinearRegressionParameters}
end

function regenerate(::SIR, output::LinearRegressionParameters,
                    xs::Vector{Float64}, ys::Vector{Float64},
                    num_particles::Int)::Float64
    if length(xs) != length(ys)
        error("length o fxs and length of ys do not match")
    end
    particles = Vector{Trace}(num_particles)

    # pick a random index from 1..num_particles
    chosen = rand(Categorical(num_particles)) 
    particles[chosen] = output

    # sample the other particles from the importance distribution
    for i=1:num_particles
        if i == chosen
            continue
        end
        trace = Trace()
        for (j, y) in enumerate(ys)
            constrain!(trace, "y$i", y)
        end
        @generate(trace, my_model(xs))
        particles[i] = trace
    end

    # compute the module's log_weight
    log_weights = map((trace) -> score(trace), particles)
    weights = exp(log_weights - logsumexp(log_weights))

    # p(u, x) / q(u; x) = pi(x, y) / mean(weights)
    chosen_trace = Trace()
    constrain!(chosen_trace, "slope", output.slope)
    constrain!(chosen_trace, "intercept", output.intercept)
    for (j, y) in enumerate(ys)
        constrain!(chosen_trace, "y$i", y)
    end
    @generate(chosen_trace, my_model(xs))
    return score(model) - (logsumexp(log_weights) - log(num_particles))
end

function simulate(::SIR,
                  xs::Vector{Float64}, ys::Vector{Float64},
                  num_particles::Int)::Tuple{LinearRegressionParameters, Float64}
    particles = Vector{Trace}(num_particles)
    for i=1:num_particles
        trace = Trace()
        for (j, y) in enumerate(ys)
            constrain!(trace, "y$i", y)
        end
        @generate(trace, my_model(xs))
        particles[i] = trace
    end
    log_weights = map((trace) -> score(trace), particles)
    weights = exp(log_weights - logsumexp(log_weights))
    chosen = Categorical(weights)
    chosen_trace = particles[chosen]

    # p(u, x) / q(u; x) = pi(x, y) / mean(weights)
    log_weight = score(chosen_trace) - (logsumexp(log_weights) - log(num_particles))
    slope = value(chosen_trace, "slope")
    intercept = value(chosen_trace, "intercept")
    output = LinearRegressionParameters(slope, intercept)
    return (output, log_weight)
end

# now that we have the module, we can do:

# 1. use it as a primitive in a probabilistic program:

# a = sir(xs, ys, num_particle) ~ "output"

# 2. implement AIDE for it

# ---------------------------------------------------------

