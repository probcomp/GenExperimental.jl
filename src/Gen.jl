module Gen

    # automatic differentiation
    include("ad/types.jl")
    include("ad/operators.jl")

    # basic math operations
    include("math.jl")

    # trace interface and probabilistic programming
    include("trace.jl")

    # custom features for jupyter notebooks
    include("notebook.jl")

    # built-in probabilistic primitives
    include("primitives/primitives.jl")

end
