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
        (g::$generator_type)(args...) = generate!(g, args, empty_trace(g))[2]
    end)

	eval(quote export $shortname end)

    # HACK to get 'export Normal' instead of 'export Gen.Normal'
	eval(quote export $(Symbol(split(string(generator_type), '.')[2])) end)
end


include("simple.jl")
include("exchangeable_joint_generator.jl")
include("crp.jl")
include("nign.jl")
