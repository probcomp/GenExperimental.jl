# Gen

Gen is a featherweight embedded probabilistic programming language and
compositional inference programming library

## Installing

First, install the Julia packages that Gen depends on:
```
julia> Pkg.add("Distributions")
julia> Pkg.add("PyPlot")
julia> Pkg.add("IJulia")
```

To make Gen available for import with `using Gen`, add the following to your
`~/.juliarc.jl` file:
```
push!(LOAD_PATH, "absolute-path-to-parent-directory-of-Gen.jl")

```

## Run tests


```
using Gen
julia> include("test/runtests.jl")

```
