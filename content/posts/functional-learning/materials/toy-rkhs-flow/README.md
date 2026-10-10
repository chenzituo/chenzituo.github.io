# Covariance-matching toy for the RKHS Flow outlook

Executed 2026-10-10. This is a finite-dimensional illustration of the source-covariance mechanism, not an RKHS model comparison or a neural-network benchmark.

## Reproduce

Run `python run.py` in an environment with NumPy 2.x and Matplotlib. The recorded execution used NumPy 2.2.3 and Matplotlib 3.10.1. Seed: 20261010. The script writes `results.json`, two PDF figure exports here, and two PNG figures under the article's `figures/` folder. It needs no downloaded data, GPU, or training.

## Construction and exact velocity

The target is an equal mixture of 96 two-dimensional Gaussians with means along a stretched spiral and component covariance `sigma² I`, with `sigma=0.12`. The mean is exactly centered. Let its covariance be `Sigma`. Compare source covariance `G=Sigma` with `G=trace(Sigma)/2 I`; their total variance is identical. A Gaussian target with covariance `Sigma` is the analytic control. Both source choices use the same independent target draws and underlying standard-normal source draws.

For angle `theta`, set `a=cos(theta)`, `s=sin(theta)`, `U=aY+sZ`, and `V=aZ-sY`. In mixture component j:

- `E[U|j]=a mu_j` and `Q=Cov(U|j)=a² sigma² I+s² G`.
- `E[V|j]=-s mu_j` and `Cov(V,U|j)=as(G-sigma² I)`.
- Thus `E[V|U=u,j]=-s mu_j+B(u-a mu_j)`, where `B=as(G-sigma² I)Q^-1`.
- Mixture responsibilities are proportional to `exp(-0.5 (u-a mu_j)^T Q^-1 (u-a mu_j))`.

Averaging those conditional means gives the exact marginal angular velocity. The Gaussian control replaces `sigma² I` by `Sigma` and has no mixture means. When `G=Sigma`, its marginal velocity is identically zero. All covariance matrices are positive definite; the positive component variance avoids a singular curve endpoint.

Generation integrates the velocity from angle pi/2 to zero with RK4. The plotted midpoint is angle pi/4, not halfway in physical time. The figures use 700 particles per source and independent target samples. Shared axes preserve the comparison of spread.

## Measurement and scope

The reported quantity is the integral of `E[||E[V|U]||_2²]` over angle `[0,pi/2]`, in common Euclidean coordinates. It is estimated using 4,096 independent source-target pairs and trapezoidal quadrature at 129 angles. Standard errors are computed across the independent per-pair integrals, retaining the dependence across angles. Error bars show 1.96 standard errors; they describe Monte Carlo uncertainty, not uncertainty about the mechanism or neural learning. The 65-angle calculation uses every other angle from the same sample paths.

Results: isotropic/matched energies are 0.806159/0 for the Gaussian control and 1.116021/0.437754 for the spiral. These are measurements of a prescribed probability path, not minima over paths or couplings. They are schedule-dependent. Lower energy does not establish reduced stiffness, faster training, fewer solver evaluations, or improved forecasting. The matched source uses exact target covariance; learning that covariance from finite data is outside this experiment. The spiral is deliberately anisotropic and is one example, not a representative dataset collection.

Checks executed by the script: exact zero Gaussian drift for matched covariance; Gaussian matrix conditional-mean identity; finite-difference verification of the mixture continuity equation at 20 fixed points and three interior angles for both sources (tolerance 1e-5); quadrature sensitivity; and a matched-source RK4 endpoint comparison at 120 versus 240 steps on 100 common particles. Full values are saved in `results.json`. The endpoint comparison checks numerical sensitivity, not distributional accuracy against an independent population metric.

The blog's lift and regression arguments are separate algebraic results from `../rkhs_dfot_architecture_20261007/theory/03_theory.md`, Sections 2–3. This toy isolates the source choice and introduces no additional observations, field lift, functional-gradient descent, or learned loss weighting.
