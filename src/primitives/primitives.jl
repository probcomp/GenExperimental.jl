# each primitive is defined in a separate source file
# each source file exports its own symbols
# NOTE: every new primitive must have a unit test suite (see test/primitives/)

primitives = Dict{Symbol, Type}()

function register_primitive(shortname::Symbol, generator_type::Type)
    primitives[shortname] = generator_type

    eval(quote
        # define singleton generator (e.g. normal)
        $shortname = $generator_type()
        
        # override call syntax for the generator
        (g::$generator_type)(args...) = simulate(g, args, ())[2]
    end)

	eval(quote export $shortname end)

    # HACK to get 'export Normal' instead of 'export Gen.Normal'
    type_str = split(string(generator_type), '.')
    if length(type_str) > 1
        # Gen.Normal (defined within Gen)
	    eval(quote export $(Symbol(type_str[2])) end)
    else
        eval(quote export $(Symbol(type_str[1])) end)
    end
      
    
end

export register_primitive

include("simple.jl")
#include("crp.jl") # TODO
#include("nign.jl")
#include("niwn.jl")
