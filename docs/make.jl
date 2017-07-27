using Documenter, Gen

makedocs(
    modules = [Gen],
    clean = true,
    authors = "Marco Cusumano-Towner, Iris Seaman, and MIT Probabilistic Computing Project",
    format = :html,
    sitename = "Gen.jl",
    pages = [
        "Gen.jl" => "index.md",
        "Inference Programming" => "inference.md",
        "Automatic Differentiation" => "ad.md",
        "Built-in Generators" => "builtin_generators.md"
    ]
)

deploydocs(
    # TODO the repository is currently private
    repo   = "github.com/probcomp/Gen.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)
