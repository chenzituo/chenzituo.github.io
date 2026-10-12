"""Analytic Gaussian and smooth-mixture coupling/schedule controls, no training.

Run with Python 3.9+, numpy, scipy, matplotlib. All actions use t in [0, 1].
The spiral source is a fixed Gaussian-like mixture, not an exact Gaussian.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent.parent / "figures"
SEED = 20261012


def schedule(t, kind):
    if kind == "Affine":
        return t, 1-t, 1., -1.
    angle = np.pi*t/2
    return np.sin(angle), np.cos(angle), np.pi/2*np.cos(angle), -np.pi/2*np.sin(angle)


def gaussian(t, kind, correlation):
    a, b, da, db = schedule(t, kind)
    q = a*a+b*b+2*correlation*a*b
    cross = a*da+b*db+correlation*(da*b+a*db)
    return q, cross*cross/q


def sinkhorn(cost, epsilon=.05):
    cost = cost/np.median(cost)
    logk = -cost/epsilon
    n = len(cost)
    logmass = -np.log(n)
    u, v = np.zeros(n), np.zeros(n)
    for iteration in range(20000):
        u = logmass-logsumexp(logk+v[None, :], axis=1)
        v = logmass-logsumexp(logk+u[:, None], axis=0)
        if iteration % 20 == 0:
            plan = np.exp(logk+u[:, None]+v[None, :])
            if max(abs(plan.sum(0)-1/n).max(), abs(plan.sum(1)-1/n).max()) < 1e-12:
                return plan, iteration+1
    raise AssertionError("Sinkhorn failed marginal tolerance")


def mixture_field(points, t, kind, plan, zmeans, ymeans, sigma, density=False):
    a, b, da, db = schedule(t, kind)
    centers = (b*zmeans[:, None, :]+a*ymeans[None, :, :]).reshape(-1, 2)
    dcenters = (db*zmeans[:, None, :]+da*ymeans[None, :, :]).reshape(-1, 2)
    q = (a*a+b*b)*sigma*sigma
    beta = (a*da+b*db)/(a*a+b*b)
    d2 = np.maximum((points*points).sum(1)[:, None]+(centers*centers).sum(1)[None, :]
                    -2*points@centers.T, 0.)
    logits = np.log(plan.ravel())[None, :]-d2/(2*q)
    normalizer = logsumexp(logits, axis=1)
    if density:
        return normalizer-np.log(2*np.pi*q)
    weights = np.exp(logits-normalizer[:, None])
    return beta*points+weights@(dcenters-beta*centers)


def main():
    rng = np.random.default_rng(SEED)
    times = np.linspace(0, 1, 129)
    exact = {"Cosine / Independent": 0., "Cosine / Identity": np.pi**2/4-np.pi/2,
             "Affine / Independent": 2-np.pi/2, "Affine / Identity": 0.}
    gaussian_results = {}
    for kind in ["Cosine", "Affine"]:
        for coupling, correlation in [("Independent", 0), ("Identity", 1)]:
            key = kind+" / "+coupling
            _, energy = gaussian(np.linspace(0, 1, 8193), kind, correlation)
            estimate = float(trapezoid(energy, dx=1/8192))
            assert abs(estimate-exact[key]) < 1e-7
            gaussian_results[key] = {"action_exact": float(exact[key]), "quadrature": estimate}

    n, sigma, samples = 24, .30, 2048
    phase = np.linspace(.4, 3.6*np.pi, n)
    radius = np.linspace(.3, 2.5, n)
    ymeans = np.column_stack((2*radius*np.cos(phase), .55*radius*np.sin(phase)))
    ymeans -= ymeans.mean(0)
    zmeans = rng.normal(size=(n, 2))
    zmeans -= zmeans.mean(0)
    # Match the center covariances exactly; adding sigma^2 I preserves equality.
    zmeans = zmeans@np.linalg.inv(np.linalg.cholesky(zmeans.T@zmeans/n)).T
    zmeans = zmeans@np.linalg.cholesky(ymeans.T@ymeans/n).T
    covariance_error = float(abs(zmeans.T@zmeans/n-ymeans.T@ymeans/n).max())
    assert covariance_error < 1e-12
    distance = ((zmeans[:, None, :]-ymeans[None, :, :])**2).sum(2)
    bounded = 2*(1-np.exp(-distance/(2*np.median(distance))))
    quadratic_plan, quadratic_iterations = sinkhorn(distance)
    bounded_plan, bounded_iterations = sinkhorn(bounded)
    plans = {"Independent": np.ones((n, n))/(n*n), "Quadratic OT": quadratic_plan,
             "Bounded-kernel OT": bounded_plan}
    mass_error = max(max(abs(p.sum(0)-1/n).max(), abs(p.sum(1)-1/n).max()) for p in plans.values())
    uniforms = rng.uniform(size=samples)
    znoise, ynoise = rng.normal(size=(samples, 2)), rng.normal(size=(samples, 2))
    points = rng.normal(size=(12, 2))
    continuity_error = 0.
    results = {}
    for coupling, plan in plans.items():
        indices = np.searchsorted(np.cumsum(plan.ravel()), uniforms)
        zi, yi = indices//n, indices % n
        z, y = zmeans[zi]+sigma*znoise, ymeans[yi]+sigma*ynoise
        for kind in ["Cosine", "Affine"]:
            energies, paired_energies = [], []
            for t in times:
                a, b, da, db = schedule(t, kind)
                points_t = a*y+b*z
                velocity = mixture_field(points_t, t, kind, plan, zmeans, ymeans, sigma)
                energies.append((velocity*velocity).sum(1))
                paired_energies.append(((da*y+db*z)**2).sum(1))
            integrals = trapezoid(np.asarray(energies), times, axis=0)
            coarse = trapezoid(np.asarray(energies)[::2], times[::2], axis=0)
            quadrature_change = float(abs(integrals.mean()-coarse.mean()))
            assert quadrature_change < .005
            results[kind+" / "+coupling] = {
                "action": float(integrals.mean()),
                "mc_standard_error": float(integrals.std(ddof=1)/np.sqrt(samples)),
                "quadrature_65_to_129_change": quadrature_change,
                "paired_target_action": float(trapezoid(np.mean(paired_energies, axis=1), times)),
            }
            # Independent finite-difference continuity equation test.
            step = 1e-5
            for t in [.17, .51, .83]:
                def logp(x, time):
                    return mixture_field(x, time, kind, plan, zmeans, ymeans, sigma, True)
                dt = (logp(points, t+step)-logp(points, t-step))/(2*step)
                grad, div = np.zeros_like(points), np.zeros(len(points))
                for axis in range(2):
                    shift = np.eye(2)[axis]*step
                    grad[:, axis] = (logp(points+shift, t)-logp(points-shift, t))/(2*step)
                    vp = mixture_field(points+shift, t, kind, plan, zmeans, ymeans, sigma)
                    vm = mixture_field(points-shift, t, kind, plan, zmeans, ymeans, sigma)
                    div += (vp[:, axis]-vm[:, axis])/(2*step)
                velocity = mixture_field(points, t, kind, plan, zmeans, ymeans, sigma)
                continuity_error = max(continuity_error, float(abs(dt+div+(velocity*grad).sum(1)).max()))
            print(kind, coupling, results[kind+" / "+coupling], flush=True)
    assert continuity_error < 2e-5, continuity_error

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = {"Cosine": "#c26336", "Affine": "#286980"}
    for kind in ["Cosine", "Affine"]:
        for coupling, corr, style in [("Independent", 0, "-"), ("Identity", 1, "--")]:
            q, _ = gaussian(times, kind, corr)
            axes[0].plot(times, np.sqrt(q), style, color=colors[kind], linewidth=2.5,
                         label=kind+", "+coupling.lower())
    axes[0].set(xlabel="Generation time t", ylabel="Standard deviation",
                title="A. Same Gaussian at both endpoints")
    axes[0].legend(fontsize=9, frameon=False)
    keys = list(exact)
    bars = axes[1].bar(range(4), [exact[k] for k in keys],
                       color=[colors[k.split(" / ")[0]] for k in keys], width=.62)
    axes[1].set_xticks(range(4), [k.replace(" / ", "\n") for k in keys], fontsize=9)
    axes[1].set(ylabel="Integrated squared marginal drift", title="B. Pairing and schedule interact", ylim=(0, 1.05))
    for bar, key in zip(bars, keys):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                     f"{exact[key]:.3f}", ha="center")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/"rkhs-coupling-schedule-gaussian.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 4.5), layout="constrained")
    for j, kind in enumerate(["Cosine", "Affine"]):
        keys = [kind+" / "+name for name in plans]
        x = np.arange(3)+(j-.5)*.34
        ax.bar(x, [results[k]["action"] for k in keys], width=.32, color=colors[kind], label=kind,
               yerr=[1.96*results[k]["mc_standard_error"] for k in keys], capsize=3)
    ax.set_xticks(np.arange(3), list(plans))
    ax.set(ylabel="Integrated squared marginal drift", title="Fixed smooth spiral target and fixed source mixture")
    ax.legend(frameon=False)
    fig.savefig(FIG/"rkhs-coupling-schedule-spiral.png", dpi=190)
    plt.close(fig)
    report = {"seed": SEED, "time_interval": [0, 1], "gaussian": gaussian_results,
              "spiral": results, "protocol": {"components_per_endpoint": n, "within_component_sigma": sigma,
              "samples": samples, "time_points": len(times), "epsilon_after_cost_median_normalization": .05,
              "source": "fixed covariance-matched Gaussian-like mixture, not an exact Gaussian",
              "coupling_scope": "OT on component labels; residual Gaussian endpoint noises are independent"},
              "checks": {"max_plan_marginal_error": float(mass_error), "source_target_covariance_error": covariance_error,
              "max_continuity_residual": continuity_error, "sinkhorn_iterations_quadratic": quadratic_iterations,
              "sinkhorn_iterations_bounded": bounded_iterations}}
    (HERE/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
