# each primitive is defined in a separate source file
# each source file exports its own symbols
# NOTE: every new primitive must have a unit test suite (see test/primitives/)
include("flip.jl")
include("uniform.jl")
include("nil.jl")
include("normal.jl")
include("gamma.jl")
include("mvnormal.jl")
include("crp.jl")
include("nign.jl")
