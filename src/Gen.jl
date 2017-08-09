module Gen

    # automatic differentiation
    include("ad/types.jl")
    include("ad/operators.jl")

    # basic math operations
    include("math.jl")

    # simple trace and generator types
    include("generator.jl")
    include("address_trie.jl")

    # probabilistic program trace and generator types
    include("program.jl")

    # custom features for jupyter notebooks
    include("notebook.jl")

    # built-in probabilistic primitives
    include("primitives/primitives.jl")
   
    # AIDE algorithm for measuring divergences
    include("aide.jl")

    # sampling importance sampling
    include("sir.jl")

end
