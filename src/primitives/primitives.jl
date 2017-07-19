# each primitive is defined in a separate source file
# each source file exports its own symbols
# NOTE: every new primitive must have a unit test suite (see test/primitives/)

primitives = Dict{Symbol, Type}()

function register_primitive(shortname::Symbol, generator_type::Type)
    primitives[shortname] = generator_type

    # define a shorthand function for an untraced invocation of the generator
    # that returns just the value 
    eval(quote
        function $shortname(args...)
            generator = $generator_type()
            (score, value) = generate!(generator, args, empty_trace(generator))
            value
        end
    end)
	eval(quote export $shortname end)
    # HACK to get 'export Normal' instead of 'export Gen.Normal'
	eval(quote export $(Symbol(split(string(generator_type), '.')[2])) end)
end


include("simple.jl")

#include("exchangeable_joint_generator.jl")
#include("crp.jl")
#include("nign.jl")
