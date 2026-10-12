# Coupling and interpolation controls

Original finite-dimensional illustrations for Section 6 of the Functional Learning blog. These computations do not train a neural network or benchmark PDE forecasts.

## Reproduce

Use Python 3.9+ with NumPy, SciPy, and Matplotlib:

```sh
python -m pip install numpy scipy matplotlib
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py
```

Run from this folder, or provide the script's full path. The script writes `results.json` here and two PNGs to the article's `figures/` folder. The recorded run used Python 3.9, NumPy 2.0.2, SciPy 1.13.1, and Matplotlib 3.9.4. Seed: 20261012. No external data are required.

## Question

Does choosing closer source–target pairs reduce marginal drift when the source law and target law stay fixed? Compare independent pairing and optimal-transport-based pairing under two schedules on a shared generation interval `t in [0,1]`:

- Affine: `U_t = t Y + (1-t) Z`.
- Cosine: `U_t = sin(pi t/2) Y + cos(pi t/2) Z`.

The metric is Euclidean and the diagnostic is the integral of the expected squared conditional-mean velocity. It is not the expected squared paired target, a training loss, or a sampling-cost estimate. Appendix A.8 gives the formulas.

## Exact Gaussian control

Source and target are both standard normal. Compare independent draws with the identity coupling `Y=Z`. Identity coupling is the unregularized quadratic-OT optimum, not a positive-entropy plan. Gaussian conditioning gives exact marginal velocities.

| Schedule | Independent action | Identity action |
| --- | ---: | ---: |
| Cosine | 0 | `pi^2/4 - pi/2 = 0.896605` |
| Affine | `2 - pi/2 = 0.429204` | 0 |

Numerical quadrature with 8,193 points agrees with each formula to below `1e-7`.

## Smooth spiral control

Both endpoints are mixtures of 24 isotropic Gaussians with within-component standard deviation 0.30. Target centers lie on a stretched spiral. Source centers are centered Gaussian draws transformed to match the target centers' covariance exactly. The resulting source is a fixed Gaussian-like mixture, not an exact Gaussian. The endpoint laws are identical across all six trials.

Couplings act on component labels. Each label has mass `1/24`. Compare the product label plan with log-domain Sinkhorn plans for squared Euclidean center distance and its bounded RBF-kernel transform. The RBF bandwidth squared is the median squared center distance. Each cost matrix is divided by its own median, then uses entropy parameter 0.05. The label plan is sampled; endpoints are not replaced by barycenters. Residual Gaussian noises in each paired component remain independent.

This restricts the continuous coupling class: it is not unrestricted continuous OT and is not a reproduction of kFFM training. Positive component variance makes the conditional-mean velocity an analytic, smooth Gaussian-mixture expression.

| Coupling | Cosine action ± MC standard error | Affine action ± MC standard error |
| --- | ---: | ---: |
| Independent | 1.3123 ± 0.0281 | 2.9205 ± 0.0488 |
| Quadratic label OT | 3.9040 ± 0.0794 | 0.3348 ± 0.0049 |
| Bounded-kernel label OT | 3.9063 ± 0.0796 | 0.3365 ± 0.0050 |

Each estimate uses 2,048 endpoint pairs and 129 time points. Common random numbers are used across choices. Standard errors are computed across entire per-pair time integrals, preserving temporal dependence. Plot error bars are ±1.96 standard errors and express Monte Carlo uncertainty, not variation across training runs. The paired-target actions are saved separately to prevent conflating them with marginal drift.

## Verification and limits

- Maximum plan marginal error: below `1e-12`.
- Source–target covariance discrepancy: below `1e-12`.
- Finite-difference continuity-equation residual: below `1e-7` at the tested points and times.
- Changing quadrature from 65 to 129 times changes every mean action by less than `0.001`.
- An additional algebra check verifies the lift isometry, Wasserstein coupling costs, and reconstruction floor; see `verify_geometry.py` and `geometry-checks.json`.

All assertions execute before the result report is written. No ODE solver is needed to estimate these prescribed marginal-path actions. The checks validate formulas and numerical integration, not a neural approximation or the accuracy of a generated PDE law. This example shows that pairing and schedule interact; it does not establish a universal preferred coupling or schedule. Its actions use unit generation time, unlike the angle-time action in the earlier source-covariance toy.

## Related source

Xu, Markovich, Barancikova, and Sun (2026), *Improving Function Space Flow Matching with Kernel Optimal Transport*, arXiv:2609.38049v1. The paper motivates the endpoint-coupling comparison. The Gaussian counterexample, smooth-mixture controls, and figures here are original calculations.
