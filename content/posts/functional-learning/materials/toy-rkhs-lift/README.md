# Two-mode RKHS lift toy

Executed 2026-10-10. This example isolates a field lift, independently of source covariance matching or neural training.

On [0,1], use `phi_1=sin(2 pi x)` and `phi_3=sin(6 pi x)`. The finite RKHS has kernel `K_rho(x,q)=phi_1(x)phi_1(q)+rho phi_3(x)phi_3(q)` and squared norm `a²+b²/rho` for the field `a phi_1+b phi_3`. Both rho choices use the same function space but different inner products.

A point velocity reading of one at x=1/4 gives `a-b=1`. The minimum-norm lift has coefficients `(1/(1+rho),-rho/(1+rho))`. Its single kernel coefficient is `1/(1+rho)`; kernel coefficients and Fourier coefficients are different coordinates. The code calculates the lift by `C O^T (O C O^T)^-1 y` and checks this against the closed form.

The figure has four panels:

- A: two lifts at rho=0.1 and rho=1, fitting exactly the same velocity reading. The declared reference `phi_1` is withheld from the lifting calculation.
- B: apply each velocity to the same initial field `0.3 phi_1` with step 0.5. Both change the observed field value from 0.3 to 0.8. This is a demonstration of a field increment, not a claim that the velocity is a functional gradient. The common initial field belongs to both RKHSs; it is not asserted to be each kernel's minimum-norm lift of its single reading.
- C: add `0.6 (phi_1+phi_3)` to the rho=0.1 lift. This addition vanishes at the sensor, is RKHS-orthogonal to the lift, and increases its squared RKHS norm. Both functions remain compatible with the observation.
- D: add a second observation from the reference, at x=1/12 with value 1/2. The two observation rows are independent, so both kernels recover the reference's two coefficients. This identifiability statement is restricted to the declared two-mode space and exact observations.

## Reproduce and checks

Run `python run.py` with NumPy and Matplotlib (recorded environment: NumPy 2.2.3, Matplotlib 3.10.1). It produces `results.json` and `../../figures/rkhs-lift-two-mode-toy.png`. No data download or training is needed.

Assertions check the closed-form coefficients, exact re-observation, the induced-norm identity, the null direction and its orthogonality, Pythagoras, linearity of observation lifting under an update, and two-sensor recovery. Results also include analytic L2 and derivative-L2 errors relative to the declared reference; these are illustrative errors, not evidence that one kernel is universally superior. The RKHS norms for different rho values are different metrics and must not be compared as a common accuracy score.

The example uses direct point evaluation, whereas the DynaBench package uses released RBF observation functionals composed with a finite Fourier field. Both instantiate the same minimum-norm lifting algebra. Neither the toy nor exact observation recovery validates unseen field values without assumptions or independent measurements.
