using Gen

function logsumexp(logx::Array{Float64,1})
    maxlog = maximum(logx)
    maxlog + log(sum(exp(logx - maxlog)))
end

# TODO how to make sure two modules outputs rae the same type when using AIDE?

immutable Flip <: Gen.Module{Bool}
end
function regenerate{T}(flip::Flip, x::Bool, p::T)::T
    x ? log(p) : log(1.0 - p)
end
function simulate{T}(flip::Flip, p::T)::Tuple{Bool,T}
    x = rand() < p
    x, regenerate(flip, x, p)
end
flip{N}(p::N) = simulate(Flip(), p)[1]
register_module(:flip, Flip())

@program model(arg::Float64) begin
    a = flip(arg) ~ "asdf"
    c = flip(0.5)
    d = 0.513 ~ "b"
end

trace = Trace()
@generate(trace, model(0.1))
print(trace)
println(trace.log_weight)

trace = Trace()
constrain!(trace, "asdf", false)
@generate(trace, model(0.4))
print(trace)
println(trace.log_weight)

function estimate_elbo{T}(p::Gen.Module{T}, q::Gen.Module{T}, KP::Int, KQ::Int, p_args::Tuple, q_args::Tuple)
    p_log_weights = Array{Float64,1}(KP)
    q_log_weights = Array{Float64,1}(KQ)
    x, p_log_weights[1] = simulate(p, p_args...)
    for k=2:KP
        p_log_weights[k] = regenerate(p, x, p_args...) # what about context
    end
    for k=1:KQ
        q_log_weights[k] = regenerate(q, x, q_args...) # what about context
    end
    ((logsumexp(q_log_weights) - log(KQ)) -
     (logsumexp(p_log_weights) - log(KP)))
end

println("0.5 to 0.1")
x = Float64[]
for i=1:10000
    push!(x, estimate_elbo(Flip(), Flip(), 2, 2, (0.5,), (0.1,)))
end
println(mean(x))

println("0.5 to 0.5")
x = Float64[]
for i=1:10000
    push!(x, estimate_elbo(Flip(), Flip(), 2, 2, (0.5,), (0.5,)))
end
println(mean(x))

#f = Flip()
#println(simulate(f, 1.5))

#@register_module(:flip, flip_simulate, flip_regenerate)

#println(flip(0.5))
