modules = Dict()
macro register_module(name, simulator, regenerator)
    if name.head != :quote error("invalid module name") end
    name = name.args[1]
    modules[name] = Pair(simulator, regenerator) # simulator returns val and log weight
    #println(simulator)
    statement = quote
        export $simulator
        export $regenerator
        $name = (args...) -> ($simulator)(args...)[1]
        export $name
    end
    #println(statement)
    eval(statement)
    #eval(quote export $simulator end)
    #eval(quote export $regenerator end)
    ##statement = quote $name = (args...) -> ($simulator)(args...)[1] end # todo do this without killing types
    #statement = quote $name = (args...) -> ($simulator)(args...)[1] end # todo do this without killing types
    #println(statement)
    #eval(statement)
    #eval(quote export $name end)
end
macro register_module2(name, simulator, regenerator)
    if name.head != :quote error("invalid module name") end
    name = name.args[1]
    modules[name] = Pair(simulator, regenerator) # simulator returns val and log weight
end


# TODO: ARE WE RETURNING -LOG WEIGHT FOR SIMULATE OR NOT?

# TODO: should we implement custom auto-diff operators for these density functions?
# it would give an order of magnitude less AD on tape?

# Uniform
#function uniform_regenerate(x::Float64)
    #x < 0 || x > 1 ? -Inf : 0.0
#end
#function uniform_simulate()
    #rand(), 0.0
#end
#@register_module(:uniform, uniform_simulate, uniform_regenerate)

# Uniform continuous
function uniform_regenerate(x::Float64, lower::Real, upper::Real)
    x < lower || x > upper ? -Inf : -log(upper - lower)
end
function uniform_simulate(lower::Real, upper::Real)
    x = rand() * (upper - lower) + lower
    x, uniform_regenerate(x, lower, upper)
end
@register_module(:uniform, uniform_simulate, uniform_regenerate)



# Bernoulli
function flip_regenerate{N}(x::Bool, p::N)
    x ? log(p) : log(1.0 - p) # TODO use log1p?
end
function flip_simulate{N}(p::N)
    x = rand() < p
    x, flip_regenerate(x, p)
end
@register_module(:flip, flip_simulate, flip_regenerate)

# Normal
function normal_regenerate{M,N,O}(x::M, mu::N, std::O)
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end
function normal_simulate{M,N}(mu::M, std::N)
    x = rand(Normal(concrete(mu), concrete(std)))
    x, normal_regenerate(x, mu, std)
end
@register_module(:normal, normal_simulate, normal_regenerate)

# Gamma (k = shape, s = scale)
function gamma_regenerate{M,N,O}(x::M, k::N, s::O)
    (k - 1.0) * log(x) - (x / s) - k * log(s) - lgamma(k)
end
function gamma_simulate{M,N}(k::M, s::N) 
    x = rand(Gamma(k, s))
    x, gamma_regenerate(x, k, s)
end
@register_module(:gamma, gamma_simulate, gamma_regenerate)

export @register_module
export @register_module2
