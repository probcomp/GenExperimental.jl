# Gen

Gen is a featherweight embedded probabilistic programming language and
compositional inference programming library

## Installing

First, install the Julia packages that Gen depends on:
```
julia> Pkg.add("Distributions")
julia> Pkg.add("DataStructures")
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

## Getting started

To launch a tutorial IJulia notebook that introduces you to Gen, open a Julia console:

```
julia
using IJulia
notebook()
```
This should open a browser window with a Jupyter directory navigator. Navigate
and open the tutorial notebook at `examples/tutorial.ipynb`.

Note that Gen is a regular Julia package. You can use Gen through an IJulia
notebook, or from the command-line and Julia console. The other examples in
`examples/` are written as regular Julia programs.
