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

The reference takes `O(2^n*(n+E+n^2))` arithmetic and retains `O(2^n*n+n^2)` data. Fitting recomputes a full ELBO after each update, so its conservative arithmetic is `O(S*n*(n+E))`; history storage is `O(S*n^2)`. Adjacency and model storage add `O(n+E)`. `compare` also retains all q state rows, so enumeration-sized output is intentional. Resource counts are operational limits, not timing guarantees. Error or work-cap rejection is not a truncated approximation.

## Tests, figure and source

24 core tests combine independent analytic references, 216 direct three-spin distributions, all covariance entries, direct product ELBO/KL sums, every gradient coordinate, exhaustive one-coordinate objective scans, all-update ascent and snapshot reconstruction, symmetry/order equivalence, the stopping nonmaximum, lost covariance, extremes/point masses, isolation, strict validation and work boundaries. Six CLI tests exercise API equality, strict file/flag errors and JSON-finite saturated outputs. Four figure tests inspect all 16 state-probability bars, nine objective/error values, complete unchanged metadata, byte regeneration and rejection of changed fixtures. Total is 34 tests.

The figure plots all four states of the same target and three product approximations on one 0..1 scale. It does not hide the symmetric run's better mean error, the broken-symmetry runs' better ELBO/KL, the missing target correlation or the independent-control pass. Full model, state probabilities, coordinate histories, settings, gates and selection statement are embedded in metadata. It is generated from the complete unchanged demo.

Standard ELBO/KL and mean-field principles follow Blei, Kucukelbir and McAuliffe (2017), sections 2.2-2.4: https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf . The Ising algebra, code, constructed examples, test oracles and diagrams are independent implementations. No paper code, prose or dataset is bundled. No observation-conditioning, richer variational family, general-purpose probabilistic programming, optimizer convergence proof or production uncertainty claim is supplied.

## Rational curvature, separate from floating-point inference

The analytic interior Hessian is `H_ii=-1/(1-m_i^2)` and `H_ij=J_ij`. Fields affect the gradient but not this Hessian. `curvature` first applies the existing strict model/mean validation and binary-float compilation, then converts each represented mean/coupling to `Fraction`. The denominator `1-m_i^2` is formed rationally, not rounded as a float. Endpoint means are rejected. This is exact curvature at the represented point, not an interval statement about uncertain decimal inputs. The separately returned gradient remains floating point.

Maintain a symmetric rational residual and a full column transform `T`, initially the identity. A nonzero diagonal can be moved to the next pivot with simultaneous row/column permutations and the same column permutation in `T`. Partition the current residual as `[B C; C^T A]`. The unit triangular congruence `[I -B^-1 C; 0 I]` gives `[B 0; 0 A-C^T B^-1 C]`. Accumulate this transform into `T`. A 1x1 pivot contributes its exact sign. If all residual diagonals are zero but an off-diagonal `b` is nonzero, move that pair to the front. Its invertible pivot block is `[0 b; b 0]`, with inverse `[0 1/b; 1/b 0]` and exactly one positive and one negative eigenvalue. If the remaining block is zero, all its remaining eigenvalues are zero. Thus no numerical pivot threshold, square root or eigenvalue approximation is needed. Every permutation and unit triangular step is invertible, including the singular residual branch.

The returned `T^T H T = D` is verified exactly in tests. Block positions make `D` independently inspectable. Sylvester's law preserves inertia under this invertible congruence; pivot values do not preserve eigenvalue magnitudes. The first positive/negative pivot supplies a coordinate vector (or sum/difference for a zero-diagonal pair); map it through the accumulated `T` to give a rational original-coordinate direction `v` and exact `v^T H v`. Later steps leave earlier pivot columns unchanged. Positive and negative witness forms are recomputed against the original Hessian.

At `J=3,m=(0,0)`, elimination returns `D=diag(-1,8)`, a negative direction `(1,0)` of form `-1`, and positive direction `(3,1)` of form `8`. The exact mathematical gradient here is zero by the formula, and the original finite-perturbation test independently increases ELBO. At `J=1,m=0`, inertia is `(0,1,1)`. Direct `nextafter(1,0)` and `nextafter(1,2)` couplings straddle that singular point with negative-definite and indefinite inertia. The existing broken-symmetry fit has negative-definite curvature at its represented final point. Its near-zero float gradient is not an exact stationarity certificate. No maximum or higher-order conclusion is inferred from curvature alone.

Dimension 1..12 bounds matrices and outputs. The algorithm uses cubic rational-operation work and quadratic storage, without exponential enumeration or coordinate fitting. Fractions have variable bit cost, and their exact numerators/denominators can be large; this is not a fixed-time bound. The 20 added curvature tests cover exact transform reconstruction, invertibility, signed witnesses, every symmetric 3x3 matrix with six independent entries in `{-1,0,1}` (729 cases) against an independent characteristic-polynomial/sign oracle, singular/zero/2x2 pivots, binary-neighbor signs, analytic formulas, Hessian finite differences, permutation invariance, full 12-spin size, near-endpoint represented values, isolation, API/CLI equality and errors. Four added figure tests verify all 15 inertia counts and positions, full certificates, unchanged regeneration and rejection of changed data. Total: 58 tests. The original demo and SVG remain byte-identical.

The inertia/congruence principle follows [Nick Higham, "What Is the Inertia of a Matrix?" (2022)](https://nhigham.com/2022/12/06/what-is-the-inertia-of-a-matrix/). Rational elimination, serialized certificates, witnesses, boundary fixtures and test oracles are independent code. No external text or implementation is bundled. There is no spectral condition-number estimate, exact gradient, optimizer convergence proof, general-purpose matrix API, or calibration claim.
