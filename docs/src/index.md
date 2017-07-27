# Gen.jl


!!! warning

    This is unsupported research software provided for educational purposes.
    The interfaces and syntax are very unstable.

Gen is an extensible, low-level Julia library for probabilistic modeling and inference.
Gen facilitates construction of high-performance implementations of custom hybrid approximate inference algorithms, using new probabilistic programming abstractions and compositions.

Gen uses two core abstractions: the [`Trace`](@ref) and the [`Generator`](@ref), described below.

# Traces

A `Trace` is an abstract mutable data structure that stores a record of a generative process.
Each `Trace` has a set of **addresses** at which values are recorded.
The word `Trace` derives from the 'execution trace' of a probabilistic program, which is one type of generative process.

```@docs
Trace
```
The basic interface to a `Trace` consists of the following functions:

```@docs
value
```

```@docs
constrain!
```

```@docs
intervene!
```

```@docs
propose!
```

```@docs
release!
```

```@docs
mode
```


# Generators

In Gen, all generative processes, including probabilistic generative models and stochastic inference algorithms, are instances of the abstact type `Generator`.
```julia
abstract type Generator{T <: Trace} end
```

```@docs
Generator
```

The key function implemented by a `Generator` is `generate!`:
```@docs
generate!
```
The score returned by `generate!` is a stochastic estimate of the probability (density) of the addresses marked with `constrain!` and `propose!` taking the values present in the trace.
The precise mathematical specification of this estimate may depend on the type of `Generator`.

Two important subtypes of `Generator` are [`ProbabilisticProgram`](@ref) and [`AtomicGenerator`](@ref).
Custom `Generator` types and corresponding `Trace` types can be implemented.


# Probabilistic Programs

Gen contains an embedded probabilistic programming language, following the 'lightweight implementation' pattern introduced in [Wingate et al. 2011](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf) with some key differences.
Probabilistic programs in Gen are Julia functions, except instead of defining the function with Julia's `function` keyword, they are defined using Gen's `@program` macro.
Unlike most other probabilistic programming languages, random choices in the probabilistic program are explicitly annotated with addresses by the programmer, using the [`@g`](@ref) macro (which stands for 'generator invocation').
Not all random choices need to be annotated---the programmer only annotates random choices that are of domain interest, or are useful targets for custom inference algorithms.

Here is an example:
```julia
@program my_model(std::Float64, n::Int) begin
    mu = @g(normal(0, 1), "mu")
    for i=1:n
        @g(normal(mu, std), "x$i")
    end
end
```

This is equivalent to the alternate syntax:
```julia
my_model = @program (std::Float64, n::Int) begin
    mu = @g(normal(0, 1), "mu")
    for i=1:n
        @g(normal(mu, std), "x$i")
    end
end
```

In both cases, `my_model` is a Julia record of type `ProbabilisticProgram`, which wraps a Julia `function` object:

```@docs
ProbabilisticProgram
```

```@docs
@program
```

Within a probabilistic program, we annotate random choices produced by `Generator` invocations with the `@g` macro.
Only random choices annotated as such can be used as constraints (i.e. observed data), as targets of custom inference proposals, or as proposed values themselves (when the program being written is a proposal program).

```@docs
@g
```

In addition to annotating generator invocations, we can annote arbitary other expressions using the [`@e`](@ref) macro.
This is useful for recording arbitrary program state for [Trace Rendering](@ref).
Addresses annotated with `@e` cannot be constrained or proposed.

```@docs
@e
```

The score returned by [`generate!`](@ref) for a `ProbabilisticProgram` is computed recursively from the scores returned by generator invocations annotated with `@g`.
Specifically, `constrain!` and `propose!` operations on the trace are propagated to the sub-traces of sub-generators.
The score returned by `generate!` is the sum of the scores returned by the recursive calls to `generate!` that are created for each generator invocation encountered during the program's execution that was annotated with `@g`.

In the special case when only constraints are applied to the trace, the score is the log of an unbiased estimate of the probability (density) of the constraints.


# Atomic Generators

Probabilistic programs are fairly 'transparent' generators---they typically expose a number of different addresses.
Depending on the context, some addresses may be constrained, intervened, or proposed.
In contrast, some generative processes have a clearly defined 'atomic output' that cannot, or does not need to be, broken up into finer-grained pieces.

In Gen, these processes are subtypes of the `AtomicGenerator` subtype.

```julia
AtomicGenerator{T} = Generator{AtomicTrace{T}}
```

```@docs
AtomicGenerator
```

```@docs
AtomicTrace
```

The score returend by [`generate!`](@ref) for a `AtomicGenerator` follows the specification given in:
[Cusumano-Towner, Mansinghka. 2016.](https://arxiv.org/abs/1612.04759)
Specifically, let $Z$ denote the output of the generator, and let $X$ denote the parameters of the generator and $x$ denote particular values of the parameters (e.g. arguments).
Then, if $Z$ is **constrained** to value $Z = z$ in the trace (i.e. the address `()` was marked with [`constrain!`](@ref)), then the score $S$ is the log of an unbiased estimate of the probability (density) of the constrained value:

```math
\mathbb{E} \left[ S \right] = p(z; Xx)
```

Equivalently, for the internal random variables $U$, let $p(u, z; x)$ denote the joint probability (density) of the internals $u$ and the output $z$ under the generative (i.e. forward) distribution.
Then, for some density on the internals $u$ and parameterized by $z$, denoted $q(u; z, x)$, the score is given by:

```math
\log s(u, x, z) = \frac{p(u, z; x)}{q(u; z, x)} \mbox{ for } u \sim q(\cdot; z, u)
```

Alternatively, $Z$ is **proposed** (i.e. if the address `()` was marked with [`propose!`](@ref)), then the score $S$ is the inverse log of an unbiased estimate of the inverse probability (density) of the constrained value:

```math
\mathbb{E} \left[ \frac{1}{S} \right] = \frac{1}{p(z; x)}
```

Equivalently, for some density on the internals $u$ and parameterized by $z$, denoted $q(u; z, x)$, the score is given by:

```math
\log s(u, x, z) = \frac{p(u, z; x)}{q(u; z, x)} \mbox{ for } u, z \sim p(u, z; x)
```

# Assessable Atomic Generators

Sometimes, the output density of stochastic computation can be computed exactly and efficiently.
Such a generator can be implemented as an `AssessableAtomicGenerator`:

```@docs
AssessableAtomicGenerator
```

# Generator Combinators

Given one Generator, we can construct another generator with the same forward sampling distribution, but with scores that are more accurate estimates of the log probability density of the constrained or proposed addresses.

There are currently two built-in mechanisms for this:

## Nested inference

```@docs
PairedGenerator
```

```@docs
compose
```

## Replication

```@docs
ReplicatedAtomicGenerator
```

```@docs
replicate
```
