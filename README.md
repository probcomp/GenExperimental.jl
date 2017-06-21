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
using Gen
julia> include("test/runtests.jl")

```

## Installing support for IJulia notebooks

Most of the tutorials take the form of [IJulia](https://github.com/JuliaLang/IJulia.jl) notebooks.
Before running these tutorials you will need to install IJulia and its
dependencies.

You will also need to install a few extra dependencies of the Gen tutorial notebooks:

```
cd examples/
./install_notebook_extensions.sh
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
