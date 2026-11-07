# Design and audit contract

## A normalized supplied target, not a learned posterior

The input is a finite Ising log potential `E(z) = sum(h_i*z_i) + sum(J_ij*z_i*z_j)`, with spins -1/+1. Target `p(z)=exp(E(z))/Z`. The fields/couplings are supplied assumptions. There is no observed data, parameter estimation, implicit temperature, hidden likelihood or inferred causal interpretation. A normalized target is enough to demonstrate variational inference and its approximation limits.

Models are compiled into immutable local tuples after strict field/index validation. Edges are sorted by `(i,j)` and stored once, with `i<j`; sparse adjacency lists are built for coordinate updates. Zero edges remain in the canonical model and work count. Public outputs contain fresh containers. No function uses a random generator or writes the input.

## Full-state reference

Enumerate `itertools.product((-1,1), repeat=n)`, so the rightmost spin changes fastest. Let `a=max_z E(z)`. Compute `logZ=a+log(fsum(exp(E(z)-a)))`, then log probabilities `E(z)-logZ`. Moments are full-state weighted sums, using `fsum`, not samples. Return every state, probability/log probability, means, second moments and covariance. All input potentials are finite, so every mathematical target state has positive mass; a float probability underflow keeps its finite log probability.

"Exact" here means enumeration of the complete finite support, not exact real arithmetic. Summed probabilities may differ from one by floating rounding, covariance diagonals may be slightly above one and a directly summed KL may be slightly negative at roundoff scale. Do not silently relabel these as rational certificates. No probability cutoff drops states or changes normalization.

For independent spins, `logZ=sum(log(2*cosh(h_i)))`, mean `tanh(h_i)` and cross-covariance zero. For two zero-field spins with coupling J, `Z=4*cosh(J)`, both means zero and cross-covariance `tanh(J)`. Tests use these separate analytic references, and unshifted direct exponentials over 216 moderate three-spin models.

## Product family and objective

Use `q(z)=product_i (1+m_i*z_i)/2`, `m_i in [-1,1]`. Independence yields:

```text
ELBO(m) = sum(h_i*m_i) + sum(J_ij*m_i*m_j)
          + sum(H_Bernoulli((1+m_i)/2))
```

Compute entropy from both probabilities `(1+m)/2` and `(1-m)/2`, ignoring the conventional zero-times-log-zero terms. Endpoints are valid distributions, not invalid logarithms. Differentiating at an interior mean gives `h_i + sum_j J_ij*m_j - atanh(m_i)`; gradients reject endpoints because the derivative there is not finite. Tests finite-difference every coordinate, not only one parameter block.

With other factors fixed, write the local field `a_i=h_i+sum_j J_ij*m_j`. Coordinate objective is `a_i*m_i+H((1+m_i)/2)+constant`, whose interior stationary point is `m_i=tanh(a_i)` and whose second derivative is `-1/(1-m_i*m_i)`. Thus each exact coordinate update maximizes that one-dimensional objective. Apply it sequentially in a requested permutation, using already changed earlier means. Synchronous updates would be a different algorithm and are not implemented.

The history includes a separate copied means vector and recomputed ELBO at initialization and after every coordinate, not only after a sweep. The maximum single-coordinate absolute change during each full sweep is the stopping statistic. Final coordinate residuals are recomputed from the final means. A stopping rule is reported literally, not called a global or local optimum; the fixture demonstrates why that distinction matters. Floating-point `tanh` can saturate at -1/+1 and tiny nonmonotonic ELBO differences can appear at roundoff scale. Tests allow 1e-12 for ascent comparisons; no line search silently changes the algorithm.

## Two independent ways to audit reverse KL

By definition, `KL(q || p)=sum_z q(z)*(logq(z)-logp(z))`. `compare` enumerates the product probability independently from the analytic ELBO, using `log1p(m_i*z_i)-log(2)` per nonsingular factor. If a factor is mathematically zero for the stored endpoint mean, that state's logq is null and probability zero; its KL contribution is zero. A target state's log probability is always finite, including when its displayed float probability underflows.

Since `logp=E-logZ`, algebra also gives `KL(q || p)=logZ-ELBO`. Return both quantities without forcing them to match or clipping one to zero. Tests reconstruct the direct sum separately with moderate, representable state probabilities, including product endpoint starts and coupled models. This tests the objective arithmetic, not the expressivity of q.

## A fixed point can stop without maximizing ELBO

For two spins with h=(0,0), J=3, the exact distribution has two almost equal aligned modes and means (0,0). At m=(0,0), both coordinate updates equal zero. The sweep-change stopping rule and final residual are both zero. The interior gradient is also zero. Yet the ELBO Hessian there has diagonal -1 and off-diagonal 3; along the joint `(1,1)` direction its quadratic curvature is positive. A small joint positive perturbation raises ELBO. This is a coordinate fixed point, not a local maximum.

Starts (+.1,+.1) and (-.1,-.1) lead to opposite near-point-mass product distributions, with means about +/-0.994902. Each has ELBO 3.005027 and reverse KL 0.690596, versus 1.386294 and 2.309329 for the symmetric start. Their marginal means are worse despite the better objective. The exact cross-covariance is 0.995055; every product result has off-diagonal covariance zero. The independently factorized control is representable and matches exact means. Both mean-error gate directions remain explicit.

The entire fixture was specified directly: J=3, fields zero, starts 0,+.1,-.1, fixed order and tolerance. There was no seed, parameter or checkpoint search and no hidden selection of a preferred mode. These comparisons describe this constructed target, not empirical uncertainty calibration or a general ranking of inference methods.

## Bounds, costs and failure behavior

Dimension 1..12 and magnitudes <=100 keep state enumeration and log energies finite. Model JSON files are at most 262,144 bytes, with duplicate-key and nonfinite-constant rejection. API booleans are not numbers, versions or indices. Bounds apply to user requests before history or state enumeration:

- Reference work: `2**n*(n+E+n*n) <= 786432`, where E is the stored edge count. The terms cover energy and moment work up to constant factors. The exactly executed boundary test uses n=12 and E=36; E=37 is refused before `_energy` runs.
- Fit work: `requested_sweeps*n*(n+E) <= 500000`. The conservative check uses the requested cap, even when a run might stop early. The boundary request n=10, E=40, 1,000 sweeps is accepted; E=41 is refused before ELBO evaluation. Max sweeps is 1,000.
- `compare` checks both limits before running either branch. Invalid settings fail through the same fit validation; error output is never a partial result.

The reference takes O(2^n*(n+E+n^2)) arithmetic and retains O(2^n*n+n^2) data. Fitting recomputes a full ELBO after each update, so its conservative arithmetic is O(S*n*(n+E)); history storage is O(S*n^2). Adjacency and model storage add O(n+E). `compare` also retains all q state rows, so enumeration-sized output is intentional. Resource counts are operational limits, not timing guarantees. Error or work-cap rejection is not a truncated approximation.

## Tests, figure and source

24 core tests combine independent analytic references, 216 direct three-spin distributions, all covariance entries, direct product ELBO/KL sums, every gradient coordinate, exhaustive one-coordinate objective scans, all-update ascent and snapshot reconstruction, symmetry/order equivalence, the stopping nonmaximum, lost covariance, extremes/point masses, isolation, strict validation and work boundaries. Six CLI tests exercise API equality, strict file/flag errors and JSON-finite saturated outputs. Four figure tests inspect all 16 state-probability bars, nine objective/error values, complete unchanged metadata, byte regeneration and rejection of changed fixtures. Total is 34 tests.

The figure plots all four states of the same target and three product approximations on one 0..1 scale. It does not hide the symmetric run's better mean error, the broken-symmetry runs' better ELBO/KL, the missing target correlation or the independent-control pass. Full model, state probabilities, coordinate histories, settings, gates and selection statement are embedded in metadata. It is generated from the complete unchanged demo.

Standard ELBO/KL and mean-field principles follow Blei, Kucukelbir and McAuliffe (2017), sections 2.2-2.4: https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf . The Ising algebra, code, constructed examples, test oracles and diagrams are independent implementations. No paper code, prose or dataset is bundled. No observation-conditioning, richer variational family, general-purpose probabilistic programming, optimizer convergence proof or production uncertainty claim is supplied.
