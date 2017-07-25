# Inference Programming

Inference algorithms can be implement by hand in Julia.
There are some common motifs, discussed below.

## Metropolis-Hastings

## Importance Sampling

## Sequential Monte Carlo

## Variational inference

## Generator as proposal

The usual approach to MCMC, importance sampling, and SMC use proposal distributions whose density or probability mass function can be efficiently evaluated.
This restricts the modularity of inference algorithms, because complex proposal samplers, whose density functions are not known, cannot have their internal details abstracted away.
This limitation on the use of black box abstraction in inference limits the complexity of inference algorithms.

It turns out we can actually relax the requirement that proposal samplers have a known density function.
Instead, we ask that proposals conform to the less stringent [`AtomicGenerator`](@ref) interface.
When we use the scores returned by the proposal generator in place of the true proposal density, we can still obtain valid Metropolis-Hastings and importance sampling algorithms on the extended auxiliary space that includes the internal random choices of the generator as additional random variables.

Consider first independent proposals, that don't depend on the previous value of the random choice they are proposing to.
Let $\pi(z)$ denote a possibly unnoralized target density on random choice $Z$.
Let $p(u, z; x)$ be the joint density of a proposal sampler, where $u$ are the internal random choices and $z$ is the proposed value.
Let $q(u; z, x)$ be the auxiliary inference density described in the Atomic Generator specification.
The ideal importance weight is:

```math
\frac{\pi(z)}{p(z; x)}
```

If we use the score of the generator in place of the true proposal density $p(z;x)$ when computing the importance weight, we get instead:
```math
\frac{\pi(z)}{\frac{p(u, z; x)}{q(u; z, x)}} = \frac{\pi(z) q(u; z, x)}{p(u, z; x)}
```

This is equivalent to an importance weight on the extended space of tuples $(u, z)$, where the proposal density is $p(u, z; x)$ and the target density is $\pi(z) q(u; z, x)$.
Therefore, we can inherit the theoretical guarantees associated with importance sampling.


## Inference as generator
