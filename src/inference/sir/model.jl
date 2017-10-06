const MODEL_ADDR_X = 1
const MODEL_ADDR_Y = 2
const MODEL_ADDR_Z = 3

@program model() begin
    x = @g(flip(0.5), ADDR_X)
    y = @g(flip(0.5), ADDR_Y)
    @g(normal(x + y, 1.), ADDR_Z)
end

const MODEL_ADDR_X = 1
const PROPOSAL_ADDR_Y = 2
const PROPOSAL_ADDR_Z = 3

@program proposal() begin
    z = @e(0., PROPOSAL_ADDR_Z)
    if (z <= 0)
        @g(flip(0.2), PROPOSAL_ADDR_X)
        @g(flip(0.2), PROPOSAL_ADDR_Y)
    elseif (z <= 2)
        @g(flip(0.5), PROPOSAL_ADDR_X)
        @g(flip(0.5), PROPOSAL_ADDR_Y)
    else
        @g(flip(0.8), PROPOSAL_ADDR_X)
        @g(flip(0.8), PROPOSAL_ADDR_Y)
    end
end

SIRGenerator(model, proposal, Dict{Int,Int}(MODEL_ADDR_X => PROPOSAL_ADDR_X, MODEL_ADDR_Y => PROPOSAL_ADDR_Y, MODEL_ADDR_Z => PROPOSAL_ADDR_Z))

# TODO can the SIRGEnerator type automatically generate the code?
# TODO does it hvae to generate a new type?

# ---- generated code ----- #

# the fast regeneration code that we want to generate, for a Type 2 Inference
# Generator that uses SIR

struct Latents 
    x::Bool
    y::Bool
end

# TODO this does not conform to the standard trace interface.
struct SIRGenerator <: Generator{DictTrace} end

# TODO the regenerate! interface should be changed; there is overhead with
# checking the addresses. there should be an option of specifying a separate
# method for each query (so that the user can directly ask the query they want,
# instead of going through the very genreic interface).

# KEY question: the Generator interface is currently very generic. If we know
# that our Generator only answers one query, then we can specialize a procedure
# for just that query. What about users of the Generator? The users will access
# it through the generic Generator interface, and their optimizer will be
# responsible for transforming the generic Generator query into the specific
# procedure we give. To facilitate this, we will produce a static table that
# maps different sets of static queries to specialized procedures, (in addition
# to providing a Generic regenerate! procedure, which invokes the other ones dynamically).

# TODO produce the lookup table, which maps statically known queries or query
# patterns to specific procedures to run, as well as how to form the arguments.
# other compilers will then reference this lookup table when optimizing the
# calling code.

# generated regenerate procedure for answering queries 
function generated_regenerate_1(g::SIRGenerator, num_particles::Int, z::Float64)
    scores = Vector{Float64}(num_particles)
    latents = Vector{Latents}(num_particles)
    for i=1:num_particles
        scores[i], latents[i] = generated_proposal_1_simulate(g, z)
        scores[i] += generated_model_1_regenerate(g, z, latents[i].x, latents[i].y)
    end
    denom = logsumexp(scores)
    probs = exp.(scores - denom)
    k = rand(Distributions.Categorical(probs / sum(probs)))
    score = denom - log(num_particles)
    return score, latents[k]
end

function generated_proposal_1_simulate(g::SIRGenerator, z::Float64)
    if (z <= 0)
        score_x, x = generated_simulate(Flip(), 0.2) # Flip generator must have already been compiled
        score_y, y = generated_simulate(Flip(), 0.2)
    elseif (z <= 2)
        score_x, x = generated_simulate(Flip(), 0.5) # Flip generator must have already been compiled
        score_y, y = generated_simulate(Flip(), 0.5)
    else
        score_x, x = generated_simulate(Flip(), 0.8) # Flip generator must have already been compiled
        score_y, y = generated_simulate(Flip(), 0.8)
    end
    return score_x + score_y, Latents(x, y)
end

# generic regenerate function
function regenerate!(g::SIRGenerator, args::Tuple{Int}, outputs, conditions, trace::DictTrace)
    if MODEL_ADDR_Z in outputs && isempty(conditions)
        for out in outputs
            if out != MODEL_ADDR_Z
                error("invalid query")
            end
        end
    end
    z = trace[MODEL_ADDR_Z]
    score, latents = generated_regenerate_1(g, args[1], z)
    trace[MODEL_ADDR_X] = latents.x
    trace[MODEL_ADDR_Y] = latents.y
    return score, nothing
end
