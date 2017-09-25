module Gen

    # automatic differentiation
    include("ad/types.jl")
    include("ad/operators.jl")

    # basic math operations
    include("math.jl")

    # simple trace and generator types
    include("address_trie.jl")
    include("generator.jl")
    include("nested.jl")
    include("replicated.jl")

    # networks of generators with dependency tracking
    # (incomplete feature)
    include("dag.jl")

    # probabilistic program trace and generator types
    include("dict_trace.jl")
    include("probabilistic_program.jl")

    # custom features for jupyter notebooks
    include("notebook.jl")

    # built-in probabilistic primitives
    include("primitives/primitives.jl")
   
    # AIDE algorithm for measuring divergences
    include("aide.jl")

    # inference algorithms
    include("inference/state_space_smc.jl")

    # numerical optimization
    include("opt/adam.jl")

end
