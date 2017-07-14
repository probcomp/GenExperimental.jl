module Gen

    # automatic differentiation
    include("ad/types.jl")
    include("ad/operators.jl")

    # basic math operations
    include("math.jl")

    # simple trace and generator types
    include("generator.jl")

    # probabilistic program trace and generator types
    include("program.jl")

    # custom features for jupyter notebooks
    include("notebook.jl")

    # built-in probabilistic primitives
    include("primitives/primitives.jl")

end
