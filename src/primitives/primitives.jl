# each primitive is defined in a separate source file
# each source file exports its own symbols
# NOTE: every new primitive must have a unit test suite (see test/primitives/)

function register_primitive(shortname::Symbol, generator_type::Type)
	eval(quote $shortname = $generator_type() end)
	eval(quote (g::$generator_type)(args...) = (g, args) end)
	eval(quote export $shortname end)
	eval(quote export $(Symbol(generator_type)) end)
end

include("simple.jl")
include("crp.jl")
include("nign.jl")
