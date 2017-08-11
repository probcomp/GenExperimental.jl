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

    # probabilistic program trace and generator types
    include("hierarchical_trace.jl")
    include("probabilistic_program.jl")

    # custom features for jupyter notebooks
    include("notebook.jl")

    # built-in probabilistic primitives
    include("primitives/primitives.jl")
   
    # AIDE algorithm for measuring divergences
    include("aide.jl")

    # inference algorithms
    #include("sir.jl")
    include("inference/state_space_smc.jl")

end
