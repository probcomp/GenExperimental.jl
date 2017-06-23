# Gen

Gen is a featherweight embedded probabilistic programming language and
compositional inference programming library


## Installing

First, install the Julia packages that Gen depends on:
```
julia> Pkg.add("Distributions")
julia> Pkg.add("DataStructures")
```

To make Gen available for import with `using Gen`, add the following to your
`~/.juliarc.jl` file:
```
push!(LOAD_PATH, "<path>")
```
where `<path>` is the absolute path to the parent directory of the Gen.jl directory.


## Run tests


```
julia test/runtests.jl

```

## Installing support for IJulia notebooks

To install the Gen.jl notebook extension for Jupyter notebooks, which provides
a simple API for Javascript-based trace renderings in Jupyter noteoboks, use:

```
cd jupyter/
./install_notebook_extensions.sh
```


## Getting started

Examples can be found in the [gen-examples repository](https://github.com/probcomp/gen-examples).
