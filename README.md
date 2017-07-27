# Gen

Gen is a featherweight embedded probabilistic programming language and compositional inference programming library.

WARNING: Gen is unsupported and highly unstable research software.

## Installing
Gen.jl is not a a publicly registered Julia package.
To use or develop Gen.jl, clone the repository with:
```
julia> Pkg.clone("git@github.com:probcomp/Gen.jl.git")
```
You can find the location on your filesystem where Julia placed the clone using:
```
julia> Pkg.dir("Gen")
```

## Run tests
```
julia> Pkg.test("Gen")
```

## Installing support for IJulia notebooks

To install the Gen.jl notebook extension for Jupyter notebooks, which provides
a simple API for Javascript-based trace renderings in Jupyter noteoboks, use:

```
$ cd jupyter/
$ ./install_notebook_extensions.sh
```
