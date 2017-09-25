using Gen
using Base.Test

include("ad.jl")
include("primitives/primitives.jl")
include("math.jl")
include("address_trie.jl")
include("program.jl")
include("nested.jl")
include("replicated.jl")
include("aide.jl")
include("inference/state_space_smc.jl")
include("dag.jl")
include("opt/adam.jl")

nothing
