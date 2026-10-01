---
title: "How and why we redefine latent representations for PDE solutions as functional learning?"
date: 2026-10-01T10:47:44-04:00
weight: -1
draft: false
tags: [functional-learning, pdes, neural-networks, flow-matching]
summary: "Why PDE learning needs a functional view of latent codes, and how that view connects solvers, physical constraints, uncertainty, and adaptive transport."
showtoc: true
tocopen: false
math: true
ShowReadingTime: true
publishDate: 2026-10-01T10:47:44-04:00
completed: true
---

*Click any figure to open its full-resolution image.*

Neural PDE models often compress a field into a latent vector and learn what happens to that vector. But what does the vector represent? With a coordinate-query decoder, fixing the code \(z\) defines a whole function \(u_z(x)\). The code and its decoder together describe the field. Querying another mesh gives another set of observations of that same field.

For a PDE, that distinction matters. We may need the field's derivatives, boundary values, fluxes, or response to a change of initial conditions. We may also need a distribution of fields when observations leave the solution uncertain. Accuracy in a vector of sampled values does not, by itself, establish accuracy in these operations.

**The case for functional learning is to make these operations part of the learning problem.** We specify what a function means, how its errors are measured, and how it should change; latent coordinates then provide a way to compute those changes. This becomes especially useful when a PDE model must work across discretizations, support physical calculations, or serve several tasks.

Three papers give us a mathematical starting point. **Functional Gradient Descent** (FGD; [Csillag et al., 2026](https://arxiv.org/abs/2606.16926)) asks how to update a function to reduce an objective. **Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations** ([Furuya et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4b6ccf3acd6ccbc1093e093df345ba2-Abstract-Conference.html)) asks how a neural operator can approximate an entire PDE solution map. **Functional Flow Matching** (FFM; [Kerrigan et al., 2024](https://proceedings.mlr.press/v238/kerrigan24a.html)) asks how to move a distribution of functions toward a target distribution. We will develop these three objects—function, operator, and law—then use [FunDiff](https://doi.org/10.1038/s41467-026-72292-0) and related PDE models to see what their shared building elements enable. The later questions concern learned solvers, inverse inference, and transport that can adapt its representation.

## Functions and latent codes

Consider a coordinate-query network \(F(z,x)\). Its two inputs have different roles: \(z\) selects a function, and \(x\) selects where to evaluate it. For example, a simple decoder could be

$$
F(z,x)=z_1\sin x+z_2\cos x.
$$

The code \((1,0)\) describes the same sine function whether we query ten locations or a thousand. This is the interpretation we want to retain for a learned decoder.

Write \(H\) for the function space, equipped with an inner product and the resulting norm. We use a real separable Hilbert space for the gradient and probability constructions below. An element \(u\in H\) can be a spatial field or an entire space-time solution \(u(x,t)\). The decoder is a map into that space:

$$
D:Z\to H,\qquad z\mapsto u_z,
\qquad
D(z)(x)=F(z,x).
$$

Thus \(z\) is a coordinate description of \(u_z\) through \(D\). Its meaning depends on that decoder: two codes can describe the same function, and changing \(D\) can change the function described by a fixed code.

[CORAL](https://arxiv.org/abs/2306.07266) makes this interpretation explicit. In Figure 1, input observations are encoded into \(z_a\); a processor predicts an output code \(\widehat z_u\); and the output decoder evaluates the predicted function at query locations \(\mathcal X\). The locations enter the decoder separately from the code. Changing them changes the returned samples of the function.

{{< figure src="figures/coral-function-codes.png" link="figures/coral-function-codes.png" alt="CORAL diagram: sampled input values are encoded into a code, a processor maps it to an output code, and a coordinate-query decoder evaluates the output function." caption="**Figure 1.** CORAL maps an input function code to an output code, then evaluates the output function at query coordinates. Source: [Serrano et al., Figure 2](https://arxiv.org/abs/2306.07266v2)." >}}

A fixed decoder selects a family \(\mathcal M=D(Z)\subset H\). Training the decoder changes the family; updating a code moves within it. [Autoencoders in Function Space](https://www.jmlr.org/papers/v26/25-0035.html) uses this separation to define reconstruction at the function level.

There are several things we might learn with such a representation:

| Object | Mathematical question | Example |
| --- | --- | --- |
| One function | \(\min_{u\in H}\mathcal L(u)\) | Solve a PDE by residual minimization |
| A map between functions | Learn \(\mathcal S:a\mapsto u\) | Map coefficients or initial data to solutions |
| A law over functions | Generate \(u\sim\mu\) | Sample a family of physically admissible fields |

For the first problem, we need an update that reduces a functional \(\mathcal L(u)\). For the second, we need a shared map that takes different input functions to their corresponding solutions. For the third, we need dynamics that produce the intended distribution. These lead to functional gradient descent, operator learning, and functional flow matching, respectively.

**Notation.** We use \(t\) for physical time, \(s\) for optimization time, and \(\tau\) for generative time. A space-time field \(u(x,t)\) can be one state in an optimization or generative process; updating it changes the entire field.

## How we have been modeling PDE solutions

The functional viewpoint has a long numerical history. A finite-element approximation \(u_h(x)=\sum_jc_j\phi_j(x)\) already interprets coefficients as coordinates of a function. Neural methods change the representation and which parts of the solution process we learn. The following comparison separates those choices; the families can overlap.

| Approach | What it computes or learns | Limitation relevant here |
| --- | --- | --- |
| Conventional discretized solvers | Values, cell averages, or basis coefficients satisfying a discrete PDE | Resolving small scales can be expensive; changing problem data generally requires another solve |
| Neural predictors on a fixed grid | A map between sampled input and output arrays | Grid transfer needs an explicit mechanism; sample accuracy alone does not control derivatives or physical constraints |
| Coordinate networks and PINNs | A function \(u_\theta(x,t)\), fitted to observations and often a PDE residual | Optimization and residual conditioning can be difficult; a basic per-problem fit must be repeated for a new problem |
| Neural operators | A solution map \(\mathcal S:a\mapsto u\) across a family of inputs | Mesh flexibility does not ensure accuracy on new physical regimes, constraints, or derivative-based quantities |
| Reduced and latent models | A basis or decoder plus dynamics or a processor in its coordinates | Compression can discard important scales; reconstruction accuracy does not establish accuracy of the represented updates |
| Generative field models | A conditional distribution of solutions or corrections | Distributional accuracy, physical validity, and consistency across resolutions need separate checks |

[PINNs](https://arxiv.org/abs/1711.10561) make the function and its differentiated residual explicit. [DeepONet](https://www.nature.com/articles/s42256-021-00302-5) and [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) learn solution maps across problem families, within the broader [operator-learning framework](https://jmlr.org/papers/v24/21-1524.html). CORAL, DINo, and function-space autoencoders connect functions to learned coordinates; FFM and FunDiff add generative modeling. Many of the functional ingredients are already present in these methods.

The limitations in the table identify requirements to examine, rather than failures of every method in a family. One documented example is [Krishnapriyan et al.'s PINN study](https://arxiv.org/abs/2109.01050): its convection and reaction–diffusion examples expose optimization difficulties despite adequate network expressivity. Simply making the network larger does not address every source of error.

Our question is how to carry the functional interpretation through the whole learning system: the norm, physical operations, update directions, solution map, and probability law. We now examine the three learning objects in that order.

## Functional gradient descent

### From a directional derivative to a gradient

Start with a loss \(\mathcal L(u)\) and perturb the function by a small amount \(\epsilon h\). For a Fréchet differentiable functional,

$$
\mathcal L(u+\epsilon h)
=\mathcal L(u)+\epsilon D\mathcal L(u)[h]+o(\epsilon).
$$

The derivative \(D\mathcal L(u)\) tells us how the loss changes in each direction \(h\). To turn it into an update, we need a function \(g\) whose inner product with every \(h\) reproduces this first-order change. The Riesz representation theorem gives exactly this function:

$$
D\mathcal L(u)[h]
=\langle\nabla_H\mathcal L(u),h\rangle_H.
$$

The gradient is therefore defined through the inner product. Changing that inner product changes which direction counts as steepest descent. This is the geometric starting point of FGD ([Csillag et al., Section 2](https://arxiv.org/abs/2606.16926)).

Moving in the negative gradient direction gives the function-space analogue of ordinary gradient flow:

$$
\frac{du_s}{ds}=-\nabla_H\mathcal L(u_s),
\qquad
\frac{d}{ds}\mathcal L(u_s)
=-\|\nabla_H\mathcal L(u_s)\|_H^2.
$$

The second identity follows by substituting the first into the chain rule. Along a sufficiently regular trajectory, the loss decreases at a rate equal to the squared gradient norm. Convergence to a global minimum additionally depends on the objective.

### The same PDE energy under two geometries

Consider the Dirichlet energy on a bounded domain with homogeneous boundary conditions:

$$
\mathcal E(u)=\frac12\int_\Omega|\nabla u|^2\,dx
-\int_\Omega fu\,dx,
\qquad u\in H_0^1(\Omega).
$$

For \(f\in L^2(\Omega)\) on a suitable domain, differentiating this energy in direction \(h\) gives

$$
D\mathcal E(u)[h]
=\int_\Omega\nabla u\cdot\nabla h\,dx
-\int_\Omega fh\,dx.
$$

Now compare two ways of turning this derivative into a function. In the \(L^2\) geometry, integration by parts gives the formal gradient \(-\Delta u-f\), on a domain where the required derivatives exist. In the energy geometry, with \(\langle g,h\rangle_{H_0^1}=\int_\Omega\nabla g\cdot\nabla h\,dx\), the gradient satisfies

$$
\int_\Omega\nabla g\cdot\nabla h\,dx
=D\mathcal E(u)[h].
$$

Writing \(A=-\Delta\) for the weak Dirichlet operator, the last equation says \(Ag=Au-f\), hence \(g=u-A^{-1}f\). The two gradients are related by an inverse elliptic operator. They describe descent for the same energy using different geometries, with different computational costs. The \(L^2\) expression needs the additional operator-domain regularity above; the energy itself is defined on \(H_0^1\).

### Adaptive approximation of the gradient

An ideal update is \(u_{k+1}=u_k-\eta\nabla_H\mathcal L(u_k)\). In practice, we have to represent both \(u_k\) and its gradient using finitely many degrees of freedom.

This is where adaptive FGD makes a specific choice. It refines the gradient representation until its approximation error is small relative to the update being taken. To see why this helps, consider the simpler Hilbert-space condition

$$
\|g_k-\nabla_H\mathcal L(u_k)\|_H
\leq\varepsilon\|g_k\|_H.
$$

For a \(K\)-smooth functional, the descent calculation gives

$$
\begin{aligned}
\mathcal L(u_k-\eta g_k)
&\leq\mathcal L(u_k)\\
&\quad-\eta\left(1-\varepsilon-\frac{K\eta}{2}\right)
\|g_k\|_H^2.
\end{aligned}
$$

The coefficient in parentheses determines whether the approximate update still decreases the loss. As the gradient becomes small, a fixed absolute approximation error can overwhelm it; a relative criterion tightens the accuracy accordingly. The short derivation is in Appendix A.

Figure 2 shows the numerical consequence in the paper's toy reconstruction example. The fixed representations eventually plateau, while the adaptive representation adds detail and continues reducing the loss. The figure compares particular methods on this example.

{{< figure src="figures/fgd-adaptive.png" link="figures/fgd-adaptive.png" alt="Three reconstruction sequences and training-loss curves comparing a neural network, fixed-resolution functional gradients, and adaptive functional gradients." caption="**Figure 2.** Reconstructions and loss curves in the FGD toy example, comparing neural, fixed, and adaptive representations. Source: [Csillag et al., Figure 1](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

The paper's algorithm uses a computable error bound for its refinement test. Its analysis also allows approximations in a larger Banach space than the Hilbert space defining the gradient. Under its extension, compatibility, smoothness, and step-size assumptions, the authors bound the minimum squared gradient norm over iterations; a Polyak–Łojasiewicz-type condition gives a geometric objective-gap bound. The full conditions are in [Section 3](https://arxiv.org/abs/2606.16926).

### Two applications: wave equations and inverse rendering

FGD's second experiment optimizes a space-time function through a wave-equation residual and initial-condition penalties. Its third optimizes density and view-dependent color through a differentiable rendering operator. Both fit the template

$$
\mathcal L(u)=\tfrac12\|\mathcal A(u)-b\|_Y^2+\lambda\mathcal R(u),
$$

where \(\mathcal A\) describes the task, \(Y\) supplies the mismatch norm, and \(\mathcal R\) is an optional regularizer. For sufficiently differentiable maps,

$$
\nabla_H\mathcal L(u)
=\mathcal A'(u)^*(\mathcal A(u)-b)
+\lambda\nabla_H\mathcal R(u).
$$

The task operator \(\mathcal A\) maps an unknown function to quantities we can compare with \(b\). Its adjoint sends the mismatch back into a function-space update. The choice of inner products determines this adjoint.

In the wave-equation experiment ([FGD, Section 4.2](https://arxiv.org/abs/2606.16926)), the unknown is the whole space-time solution. The loss combines its wave-equation residual with initial-condition penalties, much as a space-time PINN objective does. This solves a known initial-value problem through optimization. In Figure 3, each column is a physical-time slice of the resulting function; the middle row shows adaptive FGD and the bottom row the reference solution.

{{< figure src="figures/fgd-wave-solution.png" link="figures/fgd-wave-solution.png" alt="Six physical-time snapshots of a wave solution: neural network approximation, adaptive functional gradient descent, and reference solution in three rows." caption="**Figure 3.** Physical-time slices of the wave solution: neural approximation, adaptive FGD, and reference, from top to bottom. Source: [Csillag et al., Figure 3](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

Inverse rendering changes the task operator. The unknown functions describe scene density and view-dependent color, and a differentiable renderer predicts the images used in the loss ([FGD, Section 4.3](https://arxiv.org/abs/2606.16926)). Figure 4 now advances optimization iterations across columns. The adaptive representation in the bottom row resolves finer leaves as optimization proceeds; the test-loss plot reports the comparison for this Ficus scene. The rendering operator is nonlinear, so the function-space objective remains nonconvex.

{{< figure src="figures/fgd-inverse-rendering.png" link="figures/fgd-inverse-rendering.png" alt="Novel-view renderings of a potted plant through optimization iterations, comparing a neural network, fixed-grid FGD, and adaptive FGD, with test-loss curves." caption="**Figure 4.** Inverse rendering of the Ficus scene through optimization iterations, with test-loss curves for neural, fixed, and adaptive FGD representations. Source: [Csillag et al., Figure 4](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

Both applications specify an objective on functions and then approximate its updates. They seek an optimized function. If the initial conditions or coefficients change, however, the desired function changes too. This brings us to the second object: a map that serves an entire family of problems.

## Functional operator learning

A per-problem optimization produces one solution. Operator learning produces a rule for obtaining solutions when the problem data change. Let \(\mathcal A\) be an input function space and \(\mathcal U\) an output function space. The object is

$$
\mathcal S:\mathcal A\to\mathcal U,
\qquad a\mapsto u_a,
$$

and a typical training objective is

$$
\mathcal R(\theta)
=\mathbb E_{a\sim\rho}
\|\mathcal S_\theta(a)-\mathcal S(a)\|_{\mathcal U}^2.
$$

Here \(\rho\) describes the input problems we expect to encounter. The same parameters \(\theta\) serve every input. In FGD, the unknown being updated was a function \(u\); here, the unknown being fitted is a map between functions. Training that map can still use ordinary parameter-space gradient descent.

DeepONet and FNO are established examples. For a PDE-based mathematical counterpart to FGD and FFM, we will use **Furuya, Taniguchi, and Okuda's ICLR 2025 paper**. Its construction connects operator layers to a convergent solution procedure. This lets us ask what each layer computes and why errors need not accumulate without control.

### From a PDE to a fixed-point map

Consider a semilinear parabolic equation on a bounded domain \(\Omega\):

$$
\partial_tu+\mathscr A u=\mathcal N(u),
\qquad u(0)=a.
$$

The linear operator \(\mathscr A\), boundary conditions, and scalar pointwise nonlinearity \(\mathcal N\) are fixed; the initial function \(a\) varies. For the heat equation, \(\mathscr A=-\Delta\) with the chosen boundary conditions. Write \(E(t)=e^{-t\mathscr A}\) for its linear solution operator. Duhamel's formula expresses the nonlinear solution as

$$
u(t)=E(t)a+\int_0^t E(t-\zeta)\mathcal N(u(\zeta))\,d\zeta
=: \Phi_a(u)(t).
$$

Thus the solution \(\mathcal S(a)\) is a fixed point of \(\Phi_a\). Picard iteration starts with a candidate trajectory and repeatedly applies this map:

$$
u^{(0)}=0,
\qquad u^{(k+1)}=\Phi_a(u^{(k)}).
$$

Every \(u^{(k)}\) is a whole space-time function. Increasing \(k\) improves the candidate trajectory; increasing \(t\) moves along physical time within that trajectory.

The decisive condition is contraction on an invariant ball in \(\mathcal U\): for some \(0 < q < 1\), uniformly over the admitted inputs,

$$
\|\Phi_a(v)-\Phi_a(w)\|_{\mathcal U}
\le q\|v-w\|_{\mathcal U}.
$$

A short time interval can make this possible. For intuition, suppose \(\|E(t)\|\le M_E\), and \(\mathcal N\) has Lipschitz constant \(L_{\mathcal N}\) on the relevant bounded range. In a supremum-in-time norm, the integral contribution has Lipschitz bound \(M_E L_{\mathcal N}T\). Choosing \(T\) small enough makes it contractive, provided the map also stays inside that range. The paper uses semigroup smoothing estimates to work in mixed Lebesgue norms; Appendix I records its hypotheses.

### Turning the solution procedure into an operator network

Separate the initial trajectory and nonlinear correction:

$$
\Phi_a(u)=Ba+K\mathcal N(u),
$$

where

$$
(Ba)(t)=E(t)a,
\qquad
(Kg)(t)=\int_0^t E(t-\zeta)g(\zeta)\,d\zeta.
$$

A finite construction replaces the Green kernel in \(B,K\) by a truncated expansion and the scalar nonlinearity by a ReLU network. Its repeated block has the form

$$
\widehat u^{(k+1)}
=B_Na+K_N\mathcal N_\theta(\widehat u^{(k)}),
\qquad
\widehat{\mathcal S}_{N,J}(a)=\widehat u^{(J)}.
$$

The initial-data term is carried through the blocks. Kernel operations communicate across locations, while \(\mathcal N_\theta\) acts pointwise. The same block can be reused at every iteration. This is the construction behind the paper's neural-operator approximation, rather than a claim that every trained operator architecture performs Picard iteration ([Section 4.1 and Remark 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)).

{{< figure src="figures/functional-operator-picard.svg" link="figures/functional-operator-picard.svg" width="720" alt="Two parallel constructions map initial data to an entire trajectory. Exact Picard iteration converges to the PDE solution; finite kernel and neural approximations produce an operator network. A bound separates iteration error from block approximation error." caption="**Figure 5.** A solution map built from a repeated functional update. The upper row is exact Picard iteration; the lower row approximates its blocks. The same construction serves a family of initial functions. Original LaTeX/TikZ exposition of the mechanism in [Furuya et al., Section 4.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)." >}}

We can see the error mechanism without inspecting all the network weights. Suppose the approximate block has uniform defect at most \(\eta\) on the invariant ball, and its iterates remain there. With \(e_k=\|\widehat u^{(k)}-\mathcal S(a)\|_{\mathcal U}\), the triangle inequality gives

$$
e_{k+1}\le q e_k+\eta,
\qquad
\boxed{e_J\le q^J e_0+\frac{1-q^J}{1-q}\eta.}
$$

The first term is unfinished solution iteration. The second is the cost of approximating its blocks, amplified by stability. Greater depth reduces the first term; improving kernels, nonlinearities, or numerical quadrature reduces the second. **Depth alone cannot remove the approximation floor.** Appendix I derives this bound and distinguishes the continuum construction from its numerical evaluation.

### What the 2025 theorem guarantees

Under the paper's semigroup, nonlinearity, and kernel-expansion assumptions, each initial-data radius \(R\) admits a sufficiently short \(T\). For any \(\varepsilon\in(0,1)\), a ReLU neural operator exists with

$$
\sup_{\|a\|_{L^\infty(\Omega)}\le R}
\|\widehat{\mathcal S}(a)-\mathcal S(a)\|_{L^{q_t}(0,T;L^{q_x}(\Omega))}
\le\varepsilon.
$$

For small \(\varepsilon\), its depth and neuron count satisfy

$$
L_{\rm net}\le C_L\bigl(\log\varepsilon^{-1}\bigr)^2,
\qquad
N_{\rm neuron}\le C_H\varepsilon^{-1}
\bigl(\log\varepsilon^{-1}\bigr)^2.
$$

This is **uniform approximation of a solution map**, stronger than fitting one input. The construction combines logarithmically many Picard steps with approximation of a scalar nonlinearity ([Theorem 1 and its proof sketch](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)).

The kernel rank \(N\) is a separate cost, and the theorem supplies no general rate for \(N(\varepsilon)\). Its result is local in time and measured in the stated mixed Lebesgue norm. Derivative accuracy, long-time stability, generalization from finite data, and successful optimization require additional arguments. The theorem constructs approximating weights; it does not prove that training discovers them.

The choice of output norm is consequential. Operator approximation can use Banach spaces; it does not require the Hilbert structure used earlier for a Riesz gradient. If a downstream task needs a flux or a derivative, its norm and regularity requirements must enter the operator problem too.

### A contemporary architectural connection

[**Continuum Attention for Neural Operators** (Calvello et al., JMLR 2025)](https://www.jmlr.org/papers/v26/24-0879.html) gives a complementary example. Attention can be defined on functions through an integral:

$$
\operatorname{Att}(u)(x)
=\frac{\int_\Omega e^{\langle Qu(x),Ku(y)\rangle}Vu(y)\,dy}
{\int_\Omega e^{\langle Qu(x),Ku(y)\rangle}\,dy}.
$$

Here \(Q,K,V\) are pointwise linear query, key, and value maps. Finite attention approximates this operation numerically; quadrature weights matter on nonuniform nodes. The paper proves universality for a specified modified transformer operator on compact input sets, including results in differentiable-function and Sobolev norms. Those are existence guarantees, with assumptions on the spaces and architecture (Theorems 22–23), rather than an error rate for arbitrary meshes. Its spatial attention weights also differ from FFM's probability law over whole functions.

We now have a map \(\mathcal S\) that takes input functions to solution functions. Random inputs \(a\sim\rho\) induce a solution law \(\mathcal S_\#\rho\). But when a physical context leaves several fields possible, we may want to learn that conditional law directly. This is the third object, addressed by FFM.

## Functional flow matching

Suppose we have samples of functions: solution fields from a simulator, for example. We want to generate new functions from the same distribution. FFM starts with a reference distribution of random functions and learns a velocity that transports it toward the data distribution.

Figure 6 illustrates one conditional path. Each blue curve is a complete function. As generative time advances, the curve moves from noise toward a target function; the arrows show the required functional velocity. We will construct such paths first, then see how their velocities give a training objective.

{{< figure src="figures/ffm-function-flow.png" link="figures/ffm-function-flow.png" alt="Four stages of a noisy function evolving toward a sine curve, with arrows showing its function-space velocity." caption="**Figure 6.** A conditional function-space path from a noisy function toward a target curve. Arrows indicate its functional velocity. Source: [Kerrigan et al., Figure 1](https://arxiv.org/abs/2305.17209v2)." >}}

### Constructing a path between noise and a function

Draw a target function \(f\sim\nu\) and an independent Gaussian random function \(\xi\sim\mathcal N(0,C)\). Choose \(0 < \sigma_{\min} < 1\). One construction in [Kerrigan et al., Section 4](https://proceedings.mlr.press/v238/kerrigan24a.html) is

$$
\begin{aligned}
\sigma_\tau&=1-(1-\sigma_{\min})\tau,\\
u_\tau&=\tau f+\sigma_\tau\xi,\\
w_\tau&=f-(1-\sigma_{\min})\xi.
\end{aligned}
$$

At \(\tau=0\), the sample is pure reference noise. At \(\tau=1\), it is the target function plus a small residual noise term. Differentiating \(u_\tau\) gives \(w_\tau\), so we know the velocity of every sampled path without solving an ODE:

$$
\frac{du_\tau}{d\tau}
=f+\frac{d\sigma_\tau}{d\tau}\xi
=f-(1-\sigma_{\min})\xi.
$$

The positive \(\sigma_{\min}\) leaves a smoothed target law at the endpoint.

### Learning the velocity

Different pairs \((f,\xi)\) can lead to the same intermediate function. A learned velocity \(v_\theta(\tau,u)\) sees that intermediate function and generative time, rather than the particular pair used to construct it.

Sample \(\tau\sim\mathrm{Uniform}[0,1]\) independently and regress against the known conditional velocity:

$$
\mathcal J(\theta)
=\mathbb E\|v_\theta(\tau,u_\tau)-w_\tau\|_H^2.
$$

For square-integrable targets, squared-error regression learns their conditional mean:

$$
v^*_\tau(u)=\mathbb E[w_\tau\mid u_\tau=u].
$$

This averages the velocities of the conditional paths arriving at \(u\). The regression identity in Appendix B explains why fitting these sampled velocities also fits the marginal velocity, up to a parameter-independent variance term.

After training, generate a function by drawing \(u_0\sim\mu_0\) and integrating

$$
\frac{du_\tau}{d\tau}=v_\theta(\tau,u_\tau).
$$

If \(\Phi_\tau\) denotes the resulting flow map, then the evolved distribution is \(\mu_\tau=(\Phi_\tau)_\#\mu_0\): draw from the initial law and apply the map. Training uses directly sampled paths; generation uses an ODE solver. The transport interpretation requires the flow and measure assumptions developed in [Kerrigan et al., Sections 3–4](https://proceedings.mlr.press/v238/kerrigan24a.html), discussed further in Appendix C.

### Choosing a random function

The covariance \(C\) above is part of the model. To see its role, expand the random function in orthonormal modes \(e_j\), with \(Ce_j=\lambda_j e_j\):

$$
\xi=\sum_j\sqrt{\lambda_j}\epsilon_j e_j,
\qquad \epsilon_j\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
\qquad
\mathbb E\|\xi\|_H^2=\sum_j\lambda_j.
$$

A positive trace-class covariance, \(\sum_j\lambda_j<\infty\), gives an \(H\)-valued Gaussian. In \(L^2\), the eigenvalues allocate expected field energy among spatial modes. The choice \(\lambda_j=1\) for every mode has infinite total energy, so an identity-covariance Gaussian does not define a random element of infinite-dimensional \(H\). White noise can be defined in other ambient spaces, including generalized-function settings.

This makes source design a physical question: which modes should carry uncertainty, and how should their amplitudes decay as we increase resolution?

| | Functional gradient descent | Operator learning: the 2025 construction | Functional flow matching |
| --- | --- | --- | --- |
| Object | One function | A map between functions | A law over functions |
| Goal | Decrease a functional | Approximate a solution map uniformly over inputs | Transport a probability law |
| Update mechanism | Negative Riesz gradient | Repeated approximate fixed-point blocks | Learned generative velocity |
| Essential structure | Geometry and gradient accuracy | PDE well-posedness, contraction, kernel and nonlinear approximation | Measures, covariance, conditional paths |
| Iteration meaning | Optimization time | Refinement of a whole candidate trajectory | Generative time |

FGD obtains its velocity from an objective and an inner product. The operator construction obtains its blocks from a PDE integral equation. FFM obtains its velocity by regression against a probability path; that velocity need not be a gradient field. These supply three distinct learning objects, with shared questions about representation, approximation, and stability.

## The building elements of functional learning

Consider a PDE task with known problem data \(a\): coefficients, forcing, domain, and initial or boundary conditions. We want to construct a solution function, a map across these problems, or a law of possible functions conditioned on \(a\). Before choosing a neural architecture, we can specify six elements.

| Element | Mathematical form | What it determines |
| --- | --- | --- |
| Function and context | \(u\in H,\ a\in\mathcal A\) | Which physical state is represented, and for which problem |
| Operations | \(\mathcal O_hu,\ \mathcal A_a(u),\ Q(u)\) | How we observe the field, apply the governing model, and extract quantities |
| Geometry and regularity | \(\langle\cdot,\cdot\rangle_H\), domains of operators | Which errors and changes are small; which operations are defined |
| Target | \(u^\star(a),\ \mathcal S:a\mapsto u,\ \mu(du\mid a)\) | One solution, a solution map, or a conditional law |
| Construction or evolution | \(\widehat{\mathcal S}:a\mapsto u\), \(u^{(k+1)}=\Phi_a(u^{(k)})\), or \(\partial_r u_r=V_r(u_r;a)\) | How to obtain or approach that target |
| Numerical realization | \(u\approx D_m(z)\), quadrature, integration | How finite computations approximate the functional problem |

Here \(\mathcal O_h\) is an observation operator, \(\mathcal A_a\) is a physical or measurement model, and \(Q\) is a quantity such as drag or heat flux. The choice of \(H\) must support the required operations, possibly with additional regularity. For a probabilistic target, we must also specify a reference law and its covariance; the inner product alone does not determine a random function.

An operator can predict a solution directly or construct it through iterations such as Figure 5. When we use a continuous evolution, its parameter \(r\) has a meaning supplied by the task. In FGD it is optimization time \(s\), with \(V=-\nabla_H\mathcal L\). In FFM it is generative time \(\tau\), with a learned velocity chosen to transport laws. An evolution PDE supplies a third possibility: physical time \(t\), with a velocity specified by the governing equation. Figure 7 separates these continuous motions from the coordinates used to implement them.

{{< figure src="figures/functional-learning-elements.svg" link="figures/functional-learning-elements.svg" width="720" alt="A functional learning problem specifies a field, physical operations, geometry, and a target. Its motion can be optimization, physical evolution, or generative transport. A decoder and numerical solver realize the chosen motion in finite coordinates." caption="**Figure 7.** Different tasks supply different motions of a function. The decoder and numerical solver must realize the chosen motion and preserve the quantities the task needs. Original explanatory schematic, drawn in LaTeX/TikZ." >}}

This is what makes the functional viewpoint useful: the same field can be observed on several meshes, evaluated by a physical operator, optimized, or sampled from a law. Each operation has requirements that can be stated before we settle on its coordinates. A continuous decoder is one ingredient; the rest of the learning problem must respect the function it describes.

## What latent dynamics do to a function

### The geometry induced by a decoder

Let \(u=D(z)\), with differentiable \(D:\mathbb R^m\to H\). A small code change \(\delta z\) produces the first-order field change \(D'(z)\delta z\). The decoder derivative therefore determines both which directions are available and how large they are in the function norm.

For Euclidean gradient descent on the code, applying the chain rule twice gives

$$
\dot z=-D'(z)^*\nabla_H\mathcal L(u),
\qquad
\dot u=-D'(z)D'(z)^*\nabla_H\mathcal L(u).
$$

The first equation is the code update. The second is the resulting field update. Compared with \(-\nabla_H\mathcal L\), an extra operator \(D'D'^*\) appears. Its action depends on the decoder.

We can account for this geometry by measuring code changes through their decoded field changes:

$$
G(z)=D'(z)^*D'(z).
$$

For two small increments \(a,b\), \(\langle D'a,D'b\rangle_H=a^TG(z)b\). Thus \(G\) is the metric induced on the code space. When \(D'\) has full column rank it is positive definite.

To approximate a desired field velocity \(v(u)\), minimize \(\|D'(z)b-v(u)\|_H^2\) over code velocities \(b\). The normal equation is \(Gb=D'^*v\), which gives

$$
\dot z=G(z)^{-1}D'(z)^*v(D(z)).
$$

Decoding this update gives the orthogonal projection of \(v\) onto the tangent directions available through \(D\). Substituting \(v=-\nabla_H\mathcal L\) gives metric-aware descent. Substituting a generative velocity gives a projected transport update. Redundant coordinates require a pseudoinverse or a nonredundant chart.

The same construction appears in [manifold Galerkin reduced models](https://arxiv.org/abs/1812.08373): a decoder defines a trial manifold, and least-squares minimization projects the governing velocity onto its tangent space. The functional view connects latent learning to this established numerical question.

For a fixed linear synthesis \(D(a)=\sum_i a_i\psi_i\), this becomes particularly concrete:

$$
G_{ij}=\langle\psi_i,\psi_j\rangle_H,
\qquad
\|D(a)-D(b)\|_H^2=(a-b)^TG(a-b).
$$

This is the familiar Gram matrix of the basis. Overlapping basis functions create off-diagonal terms; orthonormal ones give \(G=I\). For an especially simple example, take \(D(a)=a_1e_1+2a_2e_2\) with orthonormal \(e_1,e_2\). Then \(G=\operatorname{diag}(1,4)\): equal code increments produce different field increments. For a loss depending on the second field coefficient, Euclidean code descent multiplies the decoded gradient by four, while the metric correction cancels that scaling.

Noise is affected by the same geometry. Centered Gaussian coefficients with covariance \(Q\) have expected squared field norm \(\operatorname{tr}(GQ)\). Unit code variance therefore need not mean equal variance in physical field directions. These identities explain what a decoder changes; whether accounting for them improves a trained model is an empirical question.

A controlled comparison can start with this linear dictionary, where the metric and projected velocity are known exactly, then move to nonlinear decoders. Compare Euclidean and induced-metric updates through field error, derivative error, conditioning, and computational cost.

### Decoding a flow of codes

If the code follows \(\dot z=b_\tau(z)\), then

$$
\dot u=D'(z)b_\tau(z),\qquad
\mu_\tau=D_\#\nu_\tau,
$$

where \(\nu_\tau\) is its code law. With a locally invertible chart on the represented family, this identifies a field velocity. If several codes represent the same field, their decoded velocities must agree for a single-valued deterministic field dynamics to be defined. Otherwise the code carries extra state beyond the function.

A decoder can reconstruct the endpoint fields well while giving poor tangent directions between them. Reconstruction error and velocity approximation consequently measure different aspects of the representation. This distinction will matter when we consider adaptation.

## Why PDE learning needs a functional interface

The building elements become practical when we work with simulation data. A solver stores arrays on a mesh, but the PDE and the quantities we care about act on fields. Functional learning supplies a common interface between these two levels.

[FunDiff](https://doi.org/10.1038/s41467-026-72292-0) learns a function representation and then a generative model of its latent codes. Its construction brings together three reasons for using this representation: data sampled at different resolutions, physical operations on the decoded function, and a common field description across tasks.

### Training across resolutions

Suppose one simulation stores a field on a \(64\times64\) mesh and another uses \(128\times128\). Their arrays have different sizes, but both sample a function on the physical domain. Write

$$
y_i=\mathcal O_i u_i+\varepsilon_i.
$$

Here \(\mathcal O_i\) records the sampling rule, such as point values or cell averages, and \(\varepsilon_i\) is observation error. For point values, choose a function space \(U\) with enough regularity to define evaluation; arbitrary \(L^2\) equivalence classes do not supply point values.

A numerical encoder \(E_h\) converts the samples at resolution \(h\) into a code. The decoder turns that code back into a function. Given two samplings of the same field, compare

$$
\|D(E_h(\mathcal O_hu))
-D(E_{h'}(\mathcal O_{h'}u))\|_U.
$$

This measures whether the reconstructed function changes when its sampling changes. Reconstruction error must also be measured, since two encodings can agree on an inaccurate field.

The training objective also matters. For example,

$$
\int_\Omega|\widehat u-u|^2\,dx
\approx\sum_j w_j|\widehat u(x_j)-u(x_j)|^2.
$$

The quadrature weights \(w_j\) connect the finite loss to a continuum norm. On a nonuniform mesh, an unweighted sum gives more influence to densely sampled regions. A function-space objective specifies the measure we intend to approximate.

The papers implement this connection differently. FunDiff uses random downsampling in flow-reconstruction training. [Autoencoders in Function Space](https://www.jmlr.org/papers/v26/25-0035.html) pairs function-space objectives with mesh-flexible encoders and decoders. CORAL obtains codes by fitting a coordinate-based representation to observations, as in Figure 1.

Evaluating at more coordinates samples the learned function more finely. Recovering finer physical features additionally requires the representation and data to identify them. The function-autoencoder paper discusses this distinction in its superresolution analysis. Changing the domain geometry or numerical fidelity brings further changes beyond the sampling layout.

### Computing derivatives and physical quantities

Once we have a smooth coordinate decoder \(u_z(x)=D(z)(x)\), automatic differentiation gives its derivatives. This lets a PDE residual act on the decoded function.

FunDiff also uses physical structure in the representation itself. For a sufficiently smooth two-dimensional streamfunction \(\psi\), define

$$
\mathbf v=(\partial_y\psi,-\partial_x\psi),
\qquad\nabla\cdot\mathbf v=0,
$$

The two mixed derivatives cancel, so this velocity is divergence-free by construction. A residual penalty takes another route: it penalizes violations at the points and with the weights used in the loss.

Figure 8 shows the FunDiff pipeline in the preprint. A Vision Transformer processes the sampled inputs, and a Perceiver encoder handles variable discretizations and maps them into a common latent representation. The decoder uses cross-attention between query coordinates and encoded features to evaluate the function. Physical constraints enter this function autoencoder through its architecture or training loss.

The second stage trains a Diffusion Transformer using rectified flow on the learned latent representation. Generation integrates a latent ODE from Gaussian noise and decodes its output into a function. Thus physical-prior enforcement is carried by the autoencoder and is decoupled from the generative model's training and inference.

{{< figure src="figures/fundiff-framework.png" link="figures/fundiff-framework.png" alt="FunDiff architecture: an encoder and coordinate-query decoder with physical priors, followed by latent diffusion training and inference for several field-reconstruction tasks." caption="**Figure 8.** FunDiff combines a function encoder and coordinate decoder, physical priors, and latent generation. Complete original diagram. Source: [Wang et al., Figure 1, preprint v2](https://arxiv.org/abs/2506.07902v2), [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)." >}}

The remaining issue is derivative accuracy. A decoder can approximate field values closely while having inaccurate derivatives. On \((0,2\pi)\), take the error function

$$
e_n(x)=\frac{\sin(nx)}n.
$$

Its norms are

$$
\|e_n\|_{L^2}=\frac{\sqrt\pi}{n},
\qquad
\|e_n'\|_{L^2}=\sqrt\pi,
\qquad
\|e_n''\|_{L^2}=n\sqrt\pi.
$$

The field error vanishes, the first-derivative error stays constant, and the second-derivative error grows. A PDE containing second derivatives would therefore need more than an \(L^2\) reconstruction criterion. Sobolev norms measure derivative errors directly; task-specific estimates can target the physical quantities we need. For shocks and interfaces, weak formulations may be more appropriate than high-order pointwise differentiation.

### Reusing the field across tasks

The same decoded function can serve several tasks, each with its own additional model:

| Task | What must be added to the representation |
| --- | --- |
| Reconstruction and denoising | An observation model and prior information |
| Forward PDE prediction | A map from problem data to solution fields |
| Physical forecasting | A dynamics model and time integration |
| Inverse inference | A likelihood or conditioning model and uncertainty assessment |
| Derived quantities | A functional such as an integral, flux, or gradient |

For example, [DINo](https://arxiv.org/abs/2209.14855) adds latent ODE dynamics to forecast the field in physical time. [GeoFunFlow](https://arxiv.org/abs/2509.24117) adds geometry-aware conditioning for posterior field generation. These models use a functional representation while supplying different dynamics and inference mechanisms; the task determines the needed trained components.

The relationship to downstream accuracy can be stated simply. If \(Q\) is \(L_Q\)-Lipschitz in the chosen space \(U\), then

$$
\|Q(\widehat u)-Q(u)\|
\leq L_Q\|\widehat u-u\|_U.
$$

The norm has to support the task. Differentiation is unbounded on \(L^2\), as the example above illustrates. For a derivative-based quantity \(Q\), we need a stronger norm or a separate error estimate. The benefit of the functional interface is that we can state these requirements on the field, independently of the mesh on which we inspect it.

These three demands explain when the functional view becomes important. Changing resolution asks for consistency of the reconstructed field; applying physics asks for accuracy of its operations; changing tasks asks for a common field on which new objectives or dynamics can act. A fixed-grid predictor can still be effective for a fixed-grid task. The broader demands require us to account for the underlying functions, whatever numerical architecture we use.

## Learning how a solution should change

### From a solution map to a learned correction

A [neural operator](https://jmlr.org/papers/v24/21-1524.html) learns a solution map \(\mathcal S:a\mapsto u\) between function spaces. This addresses a family of PDE problems: changing the coefficient field or initial data changes the output function. CORAL's code processor in Figure 1 is one realization of this input-to-output structure. The 2025 Picard construction gives another: approximate a stable solution procedure with shared operator blocks, as in Figure 5.

FGD suggests a complementary object to learn: an update of the current solution. We could use a learned solution map for initialization and then refine the field:

$$
u^{(0)}=\mathcal S_\theta(a),
\qquad
\frac{du_s}{ds}=V_\theta(u_s;a).
$$

This is a proposed solver design. Its update might approximate a functional gradient, a preconditioned residual correction, or another justified numerical step. A gradient approximation can inherit a descent guarantee only when the geometry, error tolerance, and step conditions needed by that guarantee hold. Small PDE residuals also need a problem-specific stability estimate before they imply small solution errors.

The interesting question is whether learning the update transfers better across a family of problems than learning only their endpoints. Can the same correction act on different mesh samplings or improve an imperfect initial prediction? The comparison should measure physical quantities and total cost at matched solution accuracy, including the correction steps.

### When the PDE supplies the velocity

For an evolution equation,

$$
\partial_tu=\mathcal F_a(u),
$$

the desired motion is already known. Restricting \(u\) to a differentiable representation \(D(z)\) gives the least-squares problem

$$
\dot z=\arg\min_b
\|D'(z)b-\mathcal F_a(D(z))\|_H^2.
$$

[Neural Galerkin](https://arxiv.org/abs/2203.01360) realizes this principle with a neural function ansatz. Its parameter evolution follows a Gram-matrix system obtained from the PDE residual; active sampling helps estimate the required integrals. The approximation now tracks the governing motion in physical time. FGD's wave example instead optimizes a whole space-time function.

This gives us three sources of functional dynamics: an objective supplies an optimization velocity, a PDE supplies a physical velocity, and a probability path supplies a generative velocity. A latent model can implement any of them, provided its decoded motion approximates the intended one.

### Representations that can express the next update

Suppose an autoencoder reconstructs every training snapshot well. That does not establish that its tangent directions can express the PDE velocity or a useful correction. A representation trained for a solver might therefore include both kinds of error:

$$
\mathbb E\left[
\|D(E(u))-u\|_H^2
+\gamma\inf_b
\|D'(E(u))b-V(u;a)\|_H^2
\right].
$$

This illustrative objective asks the representation to capture states and their required changes. The weight \(\gamma\) must account for units and scaling; the velocity must be available or estimated. Constraints such as boundary conditions may need to be built into the represented family as well.

The question becomes concrete: **can a representation with similar reconstruction error support more accurate or cheaper functional updates?** This connects representation learning to solver design, and sets up the adaptive question later.

## From finding a solution to learning a law

Inverse PDE and rendering problems often admit several fields consistent with the data. Functional optimization can find a candidate; functional transport can aim to describe their uncertainty. To connect them, we need to decide what probabilities the candidates should have.

### Descent already transports an ensemble

Initialize a random function and apply functional gradient flow to each realization. Under sufficient regularity, its law moves according to the same weak continuity equation as any other deterministic transport. Moreover,

$$
\frac{d}{ds}\mathbb E_{u\sim\mu_s}\mathcal L(u)
=-\mathbb E_{u\sim\mu_s}\|\nabla_H\mathcal L(u)\|_H^2.
$$

The ensemble's expected loss decreases. Its eventual distribution depends on the initial law and the attraction basins of the objective. If several solutions fit the observations, random initialization followed by optimization does not prescribe their relative probabilities.

### A physical objective can define a target law

One possible next step is to choose a reference law \(\mu_0\) and define

$$
\frac{d\pi_\beta}{d\mu_0}(u)
=Z_\beta^{-1}\exp[-\beta\mathcal L(u)].
$$

Assume the objective is measurable and \(0 < Z_\beta < \infty\). The reference law specifies the starting notion of plausible functions; the objective favors functions that fit the physical task. The scale \(\beta\) controls the strength of this preference. When the objective is a properly specified negative log likelihood, this has a Bayesian interpretation; an arbitrary residual penalty is a modeling choice. Densities are taken relative to a function-space reference measure, as in [Stuart's Bayesian formulation](https://doi.org/10.1017/S0962492910000061).

A transport model would seek \(T_\#\mu_0\approx\pi_\beta\). This supplies a distributional target for a PDE or rendering problem. Learning the transport requires target samples, as in ordinary sample-based flow matching, or another justified procedure for learning from the objective.

For inverse PDE inference, the unknown function might be a coefficient field whose forward solution must match observations. For inverse rendering, it might describe density and color whose rendered images must match photographs. The operator and the likelihood change; the construction still concerns a law over possible functions.

An appealing shortcut is to add a physical correction to an already trained generative velocity:

$$
v_\tau^{\mathrm{guided}}(u)
=v_\tau(u)-\lambda_\tau\nabla_H\mathcal L(u).
$$

This changes the transport and generally changes its endpoint law. It is a new sampling model whose target must be justified or evaluated. A more deliberate connection would specify the desired law first, then design or learn a velocity for it. The open question is how physical information can help learn that transport without losing the uncertainty we intended to represent.

Function-space Bayesian inversion and [functional normalizing flows](https://arxiv.org/abs/2411.13277) provide foundations for learning such transports. An analytically known target law would let us compare a descent ensemble with a sampler directly, checking their distributions as well as their losses.

### Transporting residual functions

| Intended result | Role of the physical model | What success means |
| --- | --- | --- |
| One solution | Define a residual or variational objective | Small solution error under suitable stability estimates |
| A distribution of solutions | Define a target law or reweight a reference law | Correct distribution as well as physical consistency |
| Improved approximate solutions | Define corrections between coarse and fine field laws | Reduced error and a preserved target distribution |

[Residual-augmented flow matching operators](https://arxiv.org/abs/2512.12749v3) uses the third route. A low-fidelity solver first predicts a field. The generative model learns a distribution of residual functions conditioned on that prediction and the problem input; adding a residual gives a corrected field.

In Figure 9, the upper branch supplies the low-fidelity context. The lower branch transports a Gaussian reference to residual functions. This changes what the generative model must produce: the uncertainty and structure of the correction, rather than the entire field.

{{< figure src="figures/residual-function-transport.png" link="figures/residual-function-transport.png" alt="A low-fidelity PDE solution conditions a functional flow that transports a Gaussian reference toward a distribution of residual functions, which correct the low-fidelity solution." caption="**Figure 9.** A Gaussian reference is transported to residual functions, conditioned on the problem input and a low-fidelity prediction. Source: [Bhola and Duraisamy, Figure 1](https://arxiv.org/abs/2512.12749v3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

Distributional accuracy and physical accuracy remain separate objectives. [Physics vs Distributions](https://arxiv.org/abs/2506.08604v3) studies their tension explicitly. A model can reduce a PDE residual while changing the distribution it was meant to reproduce.

We must also decide when a constraint should hold. For a linear constraint \(Au=0\), a source in its kernel and a velocity satisfying \(Av=0\) preserve the constraint along a sufficiently regular path. For a nonlinear constraint \(\mathcal C(u)=0\), the velocity must be tangent: \(D\mathcal C(u)[v]=0\). Requiring physical validity at every generative time restricts the source and path more strongly than requiring it only of final samples.

## Transporting with an adaptive representation

Some parts of a field need more degrees of freedom than others, and the required detail can change along a trajectory. Adaptive FGD refines a representation when it cannot approximate the required gradient accurately enough. Could a functional transport introduce detail when its motion requires it, rather than paying for the finest representation throughout? This section develops that research question.

### Adapt the velocity as well as the field

Let \(\mathcal M_m=D_m(Z_m)\) be a represented family, with a well-defined tangent space at \(u\). A natural measure of missing dynamics is

$$
\eta_m(\tau,u)
=\|(I-\Pi_{T_u\mathcal M_m})v_\tau(u)\|_H.
$$

The quantity \(\eta_m\) measures the velocity lost by projecting onto the available tangent directions. It can be large even when the current field itself is reconstructed well.

This is related to [dynamical low-rank approximation](https://doi.org/10.1137/050639703), which projects derivatives onto the tangent space of a moving approximation manifold. For learned transport, the additional question is how this projection changes the generated distribution.

FGD has a computable error test in its analyzed settings. For a learned transport, the exact population velocity is usually unavailable. A richer model, hierarchical detail modes, or an analytic reference problem can estimate missing directions, but disagreement alone is not a certified error bound.

### Deciding when to refine

For descent, we can ask whether an approximate gradient still reduces the loss. For generation, the quantity we ultimately care about is the error in the endpoint law.

A direct stability calculation separates three contributions: error in the initial functions, accumulated velocity error, and jumps caused by changing representations. The flow's sensitivity amplifies these errors. Appendix D gives the Grönwall bound under a globally state-Lipschitz reference velocity and finite-second-moment assumptions.

This suggests refining where the remaining contribution to endpoint error is large. Velocity-learning error, tangent-projection error, and numerical integration error should be measured separately. A relative velocity tolerance also needs an absolute tolerance near zero velocity. The generative path has no requirement to reduce a scalar objective at every step.

### Changing coordinates must preserve the sample and its law

A transfer \(R_m^{m'}\) between representations should first aim to preserve the existing field:

$$
D_{m'}(R_m^{m'}z)\approx D_m(z).
$$

This controls the change in an existing sample. There is a second issue: how to populate the newly introduced random directions.

Take a two-mode field \(U=Xe_1+Ye_2\), with orthonormal \(e_1,e_2\). A coarse representation stores only \(X\). Lifting it to \((X,0)\) preserves the coarse field exactly, but gives the second mode zero variance. If the desired fine law satisfies

$$
Y=\rho X+\sigma\epsilon,
\qquad \epsilon\sim\mathcal N(0,1)
$$

with \(\epsilon\) independent of \(X\), the new coefficient must instead be drawn from \(Y\mid X\sim\mathcal N(\rho X,\sigma^2)\). Adding independent noise would miss the correlation when \(\rho\ne0\). During a generative flow, it is the conditional law at the refinement time that matters.

A deterministic smooth lift therefore does not generally recover a full fine-dimensional law. One possible design keeps a common fine-scale random seed and reveals its modes progressively. Their distribution must be consistent with the current transport marginal, and the coarse dynamics must account for their effects.

For fixed orthogonal projection \(P_m\), the projected marginal of an exact fine flow has the formal effective velocity

$$
\bar v_m(\tau,w)
=\mathbb E[P_mv_\tau(U_\tau)\mid P_mU_\tau=w],
$$

under suitable integrability, where \(w\) is a resolved field state. This need not equal \(P_mv_\tau(w)\). Unresolved modes can influence resolved motion. Consequently, removing modes, learning their average effect, and later restoring them is a closure problem as well as a representation problem.

If the decoder itself changes continuously with generative time, the chain rule also requires

$$
\frac{du_\tau}{d\tau}
=\partial_\tau D_\tau(z_\tau)
+D_\tau'(z_\tau)\dot z_\tau.
$$

The first term is motion caused by the changing decoder itself. It must be included in the intended field velocity.

A tractable starting experiment would use fixed nested spaces and known field velocities. Compare adaptive refinement with a fixed fine representation at matched endpoint accuracy and total cost, separating projection, transfer, and integration errors. Learned velocities and a moving basis can then be studied once the transfer and marginal-consistency questions are understood.

### Source spectra, paths, and representations

| What adapts | Purpose | What must be checked |
| --- | --- | --- |
| Source covariance | Align stochastic scales with the field geometry | The continuum law and transport regularity |
| Interpolation schedule | Allocate generative time across scales | Endpoint law and integration error |
| Representation | Add or move directions available to the dynamics | Velocity defect, transfer error, and new-mode law |
| Solver step size | Integrate a given represented velocity accurately | Local and accumulated numerical error |

[Scale-Adaptive Generative Flows](https://arxiv.org/abs/2509.02971v2) studies the source-spectrum and interpolation-schedule choices. Their effect is visible in Figure 10. For the Gaussian-field experiment shown, a spectrum-matched source tracks the target spectrum with five RK4 steps. The white-source curves retain excess energy at finer frequencies at the displayed integration budgets, across the three resolutions.

{{< figure src="figures/scale-adaptive-spectra.png" link="figures/scale-adaptive-spectra.png" alt="Energy spectra at 32 by 32, 64 by 64, and 128 by 128 resolutions, comparing target Gaussian fields with generated fields using spectrum-matched or white noise and different integration budgets." caption="**Figure 10.** Gaussian-field energy spectra at 32², 64², and 128² resolution, comparing spectrum-matched and white sources at the displayed RK4 budgets. Source: [Chen and Vanden-Eijnden, Figure 2](https://arxiv.org/abs/2509.02971v2)." >}}

[Adaptive Flow Matching for Resolving Small-Scale Physics](https://proceedings.mlr.press/v267/fotiadis25a.html) uses an encoded base distribution and adaptive noise scaling. Representation refinement adds another choice: which directions are available to the velocity at a given time. Source covariance, interpolation, representation, and solver steps can each affect accuracy and cost, through the different mechanisms in the table.

These mechanisms raise a common theoretical question: can a sequence of finite models approach the intended solution or solution law while keeping its dynamics under control? Expressing a field, identifying it from measurements, learning its velocity, and integrating that velocity are different parts of the answer.

## Building a theory of functional solution learning

We can now assemble the argument from the object being learned to the error of the final computation. The construction below develops the accompanying theory draft. Projection, conditional regression, and stability supply standard ingredients; the source-conditioning comparison is a proof draft whose independent review and priority remain open. The aim here is to make the dependencies explicit.

### Start with the operations that accuracy must control

Suppose the solutions have one square-integrable spatial derivative. A possible physical space is \(H=H^1(\Omega;\mathbb R^c)\), with fixed channel scaling and a fixed length scale \(\ell>0\):

$$
\|u\|_H^2=\|u\|_{L^2}^2+\ell^2\|\nabla u\|_{L^2}^2.
$$

For homogeneous Dirichlet conditions, use the corresponding closed subspace. The same norm must be used at every resolution. If the problem contains shocks or requires higher derivatives, choose an appropriate space and weak formulation instead. In particular, \(H^1\) does not make point evaluation bounded in dimensions two and above.

This choice has an immediate consequence: \(\|\nabla u-\nabla v\|_{L^2}\le\ell^{-1}\|u-v\|_H\). For a Lipschitz observable \(Q:H\to Y\), its error is likewise bounded by \(L_Q\|u-v\|_H\). A norm chosen around the task makes these implications available.

There is still a PDE-specific requirement. A small residual proves solution accuracy only when the equation is stable in the chosen norms. For example, let \(\mathcal A:H\to H^*\) satisfy

$$
\langle\mathcal A(u)-\mathcal A(v),u-v\rangle
\ge\alpha\|u-v\|_H^2,\qquad \alpha>0,
$$

and let \(\mathcal A(u^\star)=b\). Duality gives

$$
\|u-u^\star\|_H
\le\alpha^{-1}\|\mathcal A(u)-b\|_{H^*}.
$$

Together with FGD's descent estimate, this connects optimization progress to physical solution error. Strong monotonicity is a sufficient example, rather than an assumption shared by all PDEs. Nonunique or ill-conditioned inverse problems need a different stability statement, or a target law that retains their uncertainty.

### Approximate the function before approximating its law

Let \((e_j)\) be an orthonormal basis of a real separable Hilbert space \(H\). Define nested spaces \(H_m=\operatorname{span}\{e_1,\ldots,e_m\}\), orthogonal projections \(P_m\), and a decoder

$$
D_m(z)=\sum_{j=1}^m z_j e_j.
$$

Because \(P_m u\to u\) in \(H\), these decoders become dense as their capacity grows. A nonlinear decoder family needs its own approximation result. Smooth coordinate queries alone do not prove this density, and adding query points to a fixed decoder does not add representable directions.

Now fix a solution law \(\mu\in\mathcal P_2(H)\): if \(U\sim\mu\), then \(\mathbb E\|U\|_H^2<\infty\). For a conditional problem, hold the physical context \(a\) fixed and apply the argument to \(\mu(\,\cdot\,|a)\). Uniform claims across contexts require uniform assumptions.

Measure law error in the same physical geometry:

$$
W_{2,H}(\rho,\mu)
=\inf_{(X,U)}\bigl(\mathbb E\|X-U\|_H^2\bigr)^{1/2},
$$

where the infimum ranges over couplings with the prescribed laws. Set \(\mu_m=(P_m)_\#\mu\). Orthogonality gives an exact representation floor:

$$
\delta_m^2=\mathbb E\|(I-P_m)U\|_H^2,\qquad
W_{2,H}(\mu,\mu_m)=\delta_m\longrightarrow0.
$$

Indeed, for any generated law \(\rho_m\) supported on \(H_m\),

$$
W_{2,H}(\rho_m,\mu)^2
=W_{2,H}(\rho_m,\mu_m)^2+\delta_m^2.
$$

Every coupling pays the same orthogonal tail cost; a coupling to \(\mu_m\) can be lifted using the conditional law of \(U\) given \(P_mU\). For a learned autoencoder, the more general statement is

$$
W_{2,H}\bigl(\operatorname{Law}(D(E(U))),\mu\bigr)
\le\bigl(\mathbb E\|D(E(U))-U\|_H^2\bigr)^{1/2}.
$$

Thus field reconstruction controls decoded-law approximation in that norm. It still does not establish the tangent accuracy needed by an evolving model.

### Construct a regular transport to any prescribed accuracy

The next question is whether the projected law can be generated by a regular flow. Choose **one source before choosing the target**:

$$
\gamma=\mathcal N_H(0,C),\qquad
Ce_j=q_j e_j,\qquad q_j>0,\quad \sum_jq_j<\infty.
$$

This is a full-support, finite-energy Gaussian law on \(H\). Its projected source is \(\gamma_m=\mathcal N_{H_m}(0,C_m)\).

Finite prototype laws are dense in \(\mathcal P_2(H_m)\). Choose

$$
\nu=\sum_{j=1}^K p_j\delta_{a_j},\qquad a_j\in H_m,
$$

with small \(W_{2,H}(\nu,\mu_m)\). Couple an independent prototype \(A\sim\nu\) to \(Z_m\sim\gamma_m\), using the regularized FFM path

$$
X_\tau=b_\tau Z_m+\tau A,\qquad
b_\tau=1-(1-\varepsilon)\tau,\qquad 0<\varepsilon\le1.
$$

The conditional mean velocity is globally Lipschitz for this finite construction. Its ODE has the path's one-time marginals and ends at
\(\rho_{m,\varepsilon}=\nu*\mathcal N_{H_m}(0,\varepsilon^2C_m)\). The sampled straight paths used for regression generally differ from the ODE's individual trajectories; their marginal laws agree. Appendix F gives the velocity and its regularity bound.

Coupling each prototype to itself plus the remaining Gaussian noise gives

$$
W_{2,H}(\rho_{m,\varepsilon},\mu)
\le
\underbrace{\delta_m}_{\text{representation}}
+
\underbrace{W_{2,H}(\nu,\mu_m)}_{\text{prototype approximation}}
+
\underbrace{\varepsilon\sqrt{\operatorname{tr}C_m}}_{\text{smoothing}}.
$$

For every finite-second-moment target and every positive tolerance, we can choose \(m\), \(K\), and \(\varepsilon\) so that this error is below the tolerance. This is a constructive approximation statement. It does not promise universality at fixed capacity, exact transport to a singular endpoint, or an efficient construction.

There is also a concrete neural realization: the posterior prototype probabilities are a softmax of affine measurements of the field. Their weighted sum, together with a known linear drift, implements the velocity. The construction establishes representability; learning those quantities from data remains a separate problem. In a nonorthogonal implementation, use the decoder's Gram matrix to preserve the stipulated \(H\) geometry.

### Turn velocity-learning error into endpoint error

Let \(v^*\) denote the population conditional velocity for this path, and let
\(\widehat v\) be a learned velocity. Define its **population excess risk**

$$
\mathcal E=
\mathcal J(\widehat v)-\mathcal J(v^*)
=\int_0^1
\mathbb E\|\widehat v(\tau,X_\tau)-v^*(\tau,X_\tau)\|_H^2\,d\tau.
$$

Appendix B supplies the equality. Suppose the learned field has an integrable global state-Lipschitz envelope \(\widehat L(\tau)\), with \(\widehat\Lambda=\int_0^1\widehat L(\tau)d\tau<\infty\), and the growth conditions needed for its ODE. Couple the learned and reference flows from the same initial draw. Then

$$
W_{2,H}(\widehat\rho_{m,\varepsilon},\rho_{m,\varepsilon})
\le e^{\widehat\Lambda}\sqrt{\mathcal E}.
$$

Here the Lipschitz constant belongs to the **learned velocity**: the regression error is measured along reference marginals, so we use learned-field stability to control movement away from them. Appendix G derives the bound. [Shikhman's August 2026 preprint, Theorem 23](https://arxiv.org/html/2608.04531v1), gives a related result using a superposition measure, without requiring a unique population ODE.

Finally, let \(E_{\rm num}\) bound the Wasserstein difference between the numerical sampler and the exact learned ODE. Figure 11 organizes the resulting error budget:

$$
\begin{aligned}
W_{2,H}(\widehat\mu_{\rm num},\mu)
\le{}&
\delta_m+W_{2,H}(\nu,\mu_m)
+\varepsilon\sqrt{\operatorname{tr}C_m}\\
&+e^{\widehat\Lambda}\sqrt{\mathcal E}
+E_{\rm num}.
\end{aligned}
$$

{{< figure src="figures/functional-theory-error-chain.svg" link="figures/functional-theory-error-chain.svg" width="720" alt="A LaTeX diagram tracing a target function law through projection, prototype approximation, Gaussian smoothing, a learned flow, and numerical sampling. Each transition is labeled by its contribution to the physical Wasserstein error." caption="**Figure 11.** From a target law to a computed sample law: each approximation introduces an error measured in the same physical geometry. Original LaTeX/TikZ diagram. The learning term assumes a stable learned velocity and population excess risk." >}}

The budget makes a consistency claim testable. Refinement must remove the representation floor; prototype and smoothing errors must vanish; learning must achieve \(e^{\widehat\Lambda}\sqrt{\mathcal E}\to0\); and integration error must vanish. A low empirical training loss alone establishes none of these limits. Statistical estimation, model approximation, and optimization determine the excess risk; solver accuracy determines the final term.

The result also transfers to physical outputs: for an \(L_Q\)-Lipschitz observable, \(W_2(Q_\#\widehat\mu,Q_\#\mu)\le L_QW_{2,H}(\widehat\mu,\mu)\). With the Sobolev norm above, it controls gradient-law error in \(L^2\). PDE satisfaction and conservation still need their own operator assumptions.

### Why source geometry enters the theory

The exponential factor suggests asking whether regularity deteriorates as resolution grows. Broadly, approximation capability and conditioning are already known to be distinct in normalizing-flow theory; see [Koehler, Mehta, and Risteski](https://proceedings.mlr.press/v139/koehler21a.html) and [Verine et al.](https://proceedings.mlr.press/v189/verine23a.html). The latter's expressivity bounds use total variation, whereas the comparison below uses physical Wasserstein distance.

Let \(U\sim\mu\) have centered RMS norm \(R\). Choose a fixed unit observable direction \(h\) whose standard deviation is \(s>0\), retained in every sufficiently large \(H_M\). A globally bi-Lipschitz map \(T_M\) taking Gaussian source covariance \(S_M\) to within \(\eta < s\) of \(\mu_M\) must satisfy the draft's bound

$$
\boxed{
\kappa_H(T_M)\ge
\sqrt{d_{\rm eff}(S_M)}\,\frac{s-\eta}{R+\eta},
\qquad
d_{\rm eff}(S_M)=\frac{\operatorname{tr}S_M}{\|S_M\|}.
}
$$

Here \(\kappa_H(T)=\operatorname{Lip}_H(T)\operatorname{Lip}_H(T^{-1})\) measures global distance distortion. The proof combines two demands: contract total source variance enough to match the target, while retaining variability in the observable. Gaussian Poincaré bounds the latter demand; Appendix E gives the argument and its error-floor consequence.

For \(H\)-isotropic noise, \(S_M=\sigma_M^2I_M\), so \(d_{\rm eff}=M\). Scalar normalization cancels, and the required distortion grows at least as \(\sqrt M\). A regular ODE endpoint has \(\kappa_H\le e^{2\Lambda_M}\), implying a logarithmically growing lower bound on its integrated Lipschitz envelope.

The fixed functional source admits a different construction. **Fix the target and a positive tolerance first.** Choose finitely many prototypes in \(H_m\) and a positive smoothing scale. For every ambient resolution \(M\ge m\), the same prototype-dependent velocity acts on the first \(m\) modes; the remaining modes undergo the known linear contraction. Its integrated Lipschitz bound depends on \(m\), the prototypes, \(\varepsilon\), and \(C_m^{-1}\), but not on \(M\). Thus the draft gives a resolution-independent distortion bound at that fixed accuracy. Its constant may be enormous and may diverge when the requested tolerance tends to zero.

Exact convergence with one fixed distortion budget needs stronger structure. For example, consider a Gaussian target with covariance \(r_j\) in the source eigenbasis. If
\(0 < c_-\le r_j/q_j\le c_+ < \infty\), then

$$
T_M\!\left(\sum_{j=1}^M z_j e_j\right)
=\sum_{j=1}^M\sqrt{r_j/q_j}\,z_j e_j,
\qquad
\kappa_H(T_M)\le\sqrt{c_+/c_-}.
$$

This generates \(\mu_M\) exactly, while its full-law error \(\delta_M\) tends to zero. It is a transparent positive example of compatible source and target spectra, rather than evidence that any chosen functional source fits any PDE law.

These comparisons concern regular invertible transports in a fixed physical norm. A known covariance-coloring layer can reproduce the functional source from finite white noise; a fair comparison must disclose whether its distortion lies inside the transport budget. Noninvertible decoders, dimension changes, and stochastic samplers require other arguments. Growing global distortion also does not prove slower training or more solver steps: a large, known contraction may be cheap to integrate.

### What finite observations leave unresolved

The construction above starts with full functions or their exact coefficients. Real training data provide \(Y=\mathcal O_hU+\xi\). Even an unlimited deterministic reconstructor has an information floor:

$$
\inf_{\mathcal R}\mathbb E\|U-\mathcal R(Y)\|_H^2
=\mathbb E\|U-\mathbb E[U\mid Y]\|_H^2.
$$

If two admissible fields have identical observations, querying a decoder more densely cannot identify which one was observed. A conditional generative model can instead represent their conditional law. That is a different target from exact recovery of each individual field.

There is an analogous issue for velocities. The resolved regression target is
\(\mathbb E[P_m w_\tau\mid P_mX_\tau]\), which averages over unresolved information. It need not equal a continuum velocity evaluated on a truncated field. This is the closure issue encountered in adaptive transport. [Shikhman's Theorem 13](https://arxiv.org/html/2608.04531v1) exhibits a Lipschitz continuum velocity whose finite conditional targets lack a uniform Lipschitz envelope; the example still has convergent flows. Uniform Grönwall bounds are therefore sufficient tools whose failure does not by itself prove failure of convergence.

The theory has now located the missing work precisely. We need observation consistency, approximation of the correct conditional velocity, control of learned dynamics, and accurate numerical realization. Functional learning makes these requirements refer to one common object. It gives a route from representation to PDE operations and solution laws; realizing that route requires checking each link.

## Outlook

The examples began with three objects: FGD improves a function, operator learning shares a solution map across inputs, and FFM learns to sample a function law. The wider opportunity is to build PDE models whose coordinates support the physical operations, corrections, and uncertainty that the problem calls for. Four directions seem especially useful.

### Learn a solver whose correction has a physical meaning

A learned solution map can supply a good initial field; a functional correction can then respond to new boundary conditions, parameters, or measurements. The useful question is whether that correction preserves a descent or stability property and reduces solution error at a measured cost. FGD suggests controlling gradient approximation; the Picard construction suggests controlling each block's defect and its amplification by the reference solver; Neural Galerkin suggests controlling the part of the PDE velocity that the representation can express.

A first study could compare a frozen decoder with one trained on both states and update directions. Measure residuals, field and derivative errors, and correction cost. This would test whether a representation built for motion improves the solve beyond a representation built for reconstruction.

### Let the dynamics decide when the representation grows

Adaptive transport could spend capacity where a velocity develops unresolved structure. The theory suggests a refinement criterion based on accumulated dynamical defect and endpoint sensitivity. For deterministic evolution, transfer must preserve the current function. For sampling, new directions also need the appropriate conditional distribution.

Nested spaces with known velocities provide a clean first test. Then one can study learned local bases, moving decoders, or geometry-dependent coordinates. The central question is whether these choices reduce total cost at a fixed physical endpoint accuracy, while controlling representation jumps and the law of newly introduced modes.

### Design uncertainty for the physical context

A source covariance describes which functions are plausible before transport. Compatible spectra can keep a transport regular, but a marginal field covariance may be poorly suited to a conditional inverse problem. Boundaries, forcing, and measurements can change both the uncertain scales and their correlations.

This points toward context-dependent functional sources, constraint-preserving velocities, and transports informed by PDE stability. Compare white, scalar-normalized, and spectrally colored sources with the same architecture before attributing an additional benefit to the architecture. Evaluate the generated law alongside physical residuals and observables; accurate-looking samples do not determine uncertainty calibration.

### Make the theory useful at finite accuracy

Existence and asymptotic consistency leave open the constants that govern an actual computation. A useful theory would estimate representation tails from data, relate velocity error to physical quantities of interest, and allocate error between learning, refinement, and integration. For operator learning, it should also estimate the kernel rank needed at a chosen accuracy and connect constructive approximability to finite-data learning. Local or one-sided stability may give sharper estimates than a global Lipschitz envelope.

Gaussian laws and finite mixtures offer analytically checkable starting points. The next step is to identify PDE families where the required regularity and stability can be established, then test the resulting predictions under refinement and changing observations.

The title's reinterpretation is therefore consequential: a latent code is a way to compute with a function through a decoder. Carrying that meaning through the whole model lets the same solution be observed, differentiated, corrected, and sampled. Functional learning is especially valuable when a PDE application needs several of these capabilities together. The research opportunity is to make their mathematical compatibility translate into reliable computation.

## Appendix

### A. A descent bound with an approximate gradient

Let \(a=\nabla_H\mathcal L(u)\) and suppose \(\|g-a\|_H\leq\varepsilon\|g\|_H\). Then

$$
\begin{aligned}
\langle a,g\rangle_H
&=\|g\|_H^2+\langle a-g,g\rangle_H\\
&\geq(1-\varepsilon)\|g\|_H^2.
\end{aligned}
$$

Combining this with \(K\)-smoothness gives the descent inequality in the FGD section. A positive decrease coefficient requires \(\varepsilon < 1\) and \(0 < \eta < 2(1-\varepsilon)/K\). This calculation is the Hilbert-space special case; the adaptive FGD paper's larger-space analysis needs additional compatibility assumptions.

### B. Why conditional regression works

Set \(v^*=\mathbb E[w\mid u_\tau,\tau]\). For square-integrable targets and predictions, conditional expectation makes the cross term vanish:

$$
\begin{aligned}
\mathbb E\|v_\theta-w\|_H^2
&=\mathbb E\|v_\theta-v^*\|_H^2\\
&\quad+\mathbb E\|w-v^*\|_H^2.
\end{aligned}
$$

The last term is independent of the model parameters. This is a regression identity; establishing the associated transport of measures additionally requires appropriate flow and measure assumptions.

### C. Probability transport in function space

Let \(\Phi_\tau\) be the flow of a velocity \(v_\tau:H\to H\):

$$
\partial_\tau\Phi_\tau(u)=v_\tau(\Phi_\tau(u)),
\qquad \Phi_0(u)=u,
\qquad \mu_\tau=(\Phi_\tau)_\#\mu_0.
$$

For suitable smooth cylindrical test functionals \(\psi\), the weak continuity equation is

$$
\frac{d}{d\tau}\int_H\psi(u)\,d\mu_\tau(u)
=\int_H D\psi(u)[v_\tau(u)]\,d\mu_\tau(u).
$$

It describes probability moving among functions, on \(H\). A physical conservation equation instead acts on quantities over the spatial domain \(\Omega\). [Kerrigan et al., Section 3](https://proceedings.mlr.press/v238/kerrigan24a.html) gives the function-space transport formulation.

There is no nontrivial locally finite translation-invariant Lebesgue measure on infinite-dimensional \(H\). Probability constructions therefore use reference measures rather than a continuum version of a finite-dimensional Lebesgue density.

FFM's shared-covariance Gaussian construction also uses absolute-continuity assumptions. A sufficient condition places the relevant mean shifts in the Cameron–Martin space \(\operatorname{Range}(C^{1/2})\). The authors acknowledge that verifying the data-support condition in practice is difficult. These measure assumptions accompany the regression identity when identifying a valid transport.

### D. A stability bound for changing representations

Suppose a reference trajectory satisfies \(\dot u_\tau=v_\tau(u_\tau)\), with global state-Lipschitz envelope \(L(\tau)\). Between representation changes, define the computed trajectory's defect as

$$
r_\tau=\dot{\widetilde u}_\tau-v_\tau(\widetilde u_\tau).
$$

Couple the initial states and assume finite second moments and integrable defect norms. If \(J_j\) is a field jump introduced at a representation change, Grönwall's inequality bounds the coupled endpoint difference. Using this coupling to bound the Wasserstein distance gives

$$
\begin{aligned}
W_{2,H}(\operatorname{Law}(u_1),\operatorname{Law}(\widetilde u_1))
\le e^{\int_0^1L(\tau)d\tau}\Big[&\|u_0-\widetilde u_0\|_{L^2(\mathbb P;H)}\\
&+\int_0^1\|r_\tau\|_{L^2(\mathbb P;H)}d\tau\\
&+\sum_j\|J_j\|_{L^2(\mathbb P;H)}\Big].
\end{aligned}
$$

The three terms measure initial-state error, accumulated dynamical defect, and representation jumps. The defect can include velocity-learning, tangent-projection, and numerical errors. This is a direct stability calculation under the stated assumptions; turning it into a computable refinement rule requires estimates of those errors and the reference flow's stability.

### E. Source geometry: a theory draft

Use the same fixed physical Hilbert norm, nested spaces \(H_M\), and projected target \(\mu_M\). Let \(R\) be the full target's centered RMS norm and \(s>0\) the standard deviation of a unit linear observable retained in \(H_M\). Let \(Z\sim\mathcal N_{H_M}(0,S_M)\), \(Y=T(Z)\), and suppose \(W_{2,H}(\operatorname{Law}(Y),\mu_M)\le\eta < s\).

Write \(A=\operatorname{Lip}_H(T)\) and \(B=\operatorname{Lip}_H(T^{-1})\). Independent copies and the inverse Lipschitz bound give

$$
\bigl(\mathbb E\|Y-\mathbb EY\|_H^2\bigr)^{1/2}
\ge B^{-1}\sqrt{\operatorname{tr}S_M}.
$$

Centered RMS norms differ by at most the Wasserstein distance, so
\(B\ge\sqrt{\operatorname{tr}S_M}/(R+\eta)\). For the scalar function
\(g(z)=\langle T(z),h\rangle_H\), Gaussian Poincaré gives

$$
(s-\eta)^2\le\operatorname{Var}(g(Z))
\le\|S_M\|A^2.
$$

The inequality follows by applying the standard-Gaussian result to
\(g(S_M^{1/2}\,\cdot\,)\); see [Bandeira, Singer, and Strohmer, Proposition 8.8](https://www.math.ucdavis.edu/~strohmer/papers/2025/MDS_Book.pdf#page=145), including its Lipschitz extension. Multiplication yields

$$
\kappa_H(T)=AB\ge
\sqrt{\frac{\operatorname{tr}S_M}{\|S_M\|}}
\frac{s-\eta}{R+\eta}.
$$

**Error floor.** If \(\kappa_H(T_M)\le K\), set \(D_M=\sqrt{d_{\rm eff}(S_M)}\). Rearranging the bound when the error is below \(s\), and using the trivial case otherwise, gives

$$
W_{2,H}((T_M)_\#\operatorname{Law}(Z),\mu_M)
\ge\frac{(D_Ms-KR)_+}{D_M+K}.
$$

If \(D_M\to\infty\) with \(K\) fixed, the lower limit of the error is at least \(s\). Orthogonal projection shows that the full-target error is no smaller. A deterministic target has no such \(s>0\) and is excluded.

**ODE consequence.** An integrable global state-Lipschitz envelope gives forward and inverse flow constants at most \(e^{\Lambda_M}\). Hence

$$
\Lambda_M\ge
\max\left\{0,\frac14\log d_{\rm eff}(S_M)
+\frac12\log\frac{s-\eta}{R+\eta}\right\}.
$$

**A matching fixed-accuracy example.** Let the projected Gaussian target have covariance \(R_{0,M}\), with positive eigenvalues \(r_j\) and \(r_*=\sup_jr_j<\infty\). From an \(H\)-isotropic source \(\mathcal N(0,\sigma_M^2I)\), the linear map

$$
T_M=\sigma_M^{-1}\left(R_{0,M}+\frac{\eta^2}{M}I\right)^{1/2}
$$

generates the target plus independent noise of RMS norm \(\eta\). Its target error is at most \(\eta\), while

$$
\kappa_H(T_M)\le\sqrt{1+Mr_*/\eta^2}.
$$

Thus the square-root lower rate is order-sharp for these Gaussian targets at fixed positive projected-law accuracy. This does not give a sharp joint limit as accuracy tends to zero.

**Status: theory in progress.** These are derivations from standard inequalities, with independent review and priority still open. They concern global transport distortion. They supply neither training-time nor solver-cost bounds; covariance coloring must be included or disclosed in a matched comparison.

### F. A constructive finite-prototype flow

Fix prototypes \(a_j\in H_m\), positive weights \(p_j\), covariance \(C_m>0\), and \(0<\varepsilon\le1\). Write \(\lambda=1-\varepsilon\), \(b_\tau=1-\lambda\tau\). Bayes' rule for \(X_\tau=b_\tau Z_m+\tau A\) gives posterior weights

$$
\begin{aligned}
\omega_j(\tau,x)&=\operatorname{softmax}_j(g_1,\ldots,g_K),\\
g_j(\tau,x)&=\log p_j
+\frac{\tau}{b_\tau^2}\langle C_m^{-1}a_j,x\rangle_H
-\frac{\tau^2}{2b_\tau^2}\langle C_m^{-1}a_j,a_j\rangle_H,\\
d_\tau(x)&=\sum_j\omega_j(\tau,x)a_j.
\end{aligned}
$$

The common Gaussian factor cancels. All covariance inverses here act on the finite space \(H_m\). Since \(A-\lambda Z_m=(A-\lambda X_\tau)/b_\tau\), the population conditional velocity is

$$
v^*(\tau,x)=\frac{d_\tau(x)-\lambda x}{b_\tau}.
$$

Let \(R_A=\max_j\|a_j\|_H\). Differentiating the posterior mean produces a covariance of the prototypes, whose operator norm is at most \(R_A^2\). Consequently,

$$
\operatorname{Lip}_H(v^*_\tau)
\le\frac{\lambda}{b_\tau}
+\frac{\tau R_A^2\|C_m^{-1}\|}{b_\tau^3},
$$

and

$$
\int_0^1\operatorname{Lip}_H(v^*_\tau)d\tau
\le\log(1/\varepsilon)
+\frac{R_A^2\|C_m^{-1}\|}{2\varepsilon^2}.
$$

The field is globally Lipschitz with linear growth. The path marginals solve its finite-dimensional continuity equation; uniqueness for this regular field identifies them with the ODE marginals.

For an ambient \(H_M\), \(M\ge m\), replace \(d_\tau(x)\) by \(d_\tau(P_mx)\) and use source \(C_M\). The remaining coordinates solve
\(\dot x_\perp=-\lambda x_\perp/b_\tau\), hence \(x_\perp(\tau)=b_\tau x_\perp(0)\). Independence of Gaussian eigenmodes gives endpoint law \(\nu*\mathcal N_{H_M}(0,\varepsilon^2C_M)\), with the same Lipschitz bound for all \(M\). This proves the fixed-accuracy positive construction used in the main text. The possibly large inverse-covariance factor explains why existence alone is insufficient for efficiency.

### G. Population excess risk and learned-flow stability

Let \(Y_\tau\) follow the regular reference flow from Appendix F, and let \(\widehat Y_\tau\) follow the learned velocity from the same initial state. Their difference obeys

$$
\begin{aligned}
\frac{d}{d\tau}\|\widehat Y_\tau-Y_\tau\|_H
\le{}&
\widehat L(\tau)\|\widehat Y_\tau-Y_\tau\|_H\\
&+\|\widehat v(\tau,Y_\tau)-v^*(\tau,Y_\tau)\|_H.
\end{aligned}
$$

The norm inequality holds almost everywhere, with the usual interpretation at zero. Grönwall, Minkowski, and Cauchy–Schwarz on the unit time interval give

$$
\|\widehat Y_1-Y_1\|_{L^2(\mathbb P;H)}
\le e^{\widehat\Lambda}
\left(\int_0^1
\mathbb E\|\widehat v(\tau,Y_\tau)-v^*(\tau,Y_\tau)\|_H^2d\tau
\right)^{1/2}.
$$

The reference marginals equal the training-path marginals, so Appendix B identifies the integral with \(\mathcal E\). The coupling bounds endpoint Wasserstein distance. Adding approximation, smoothing, and numerical error by the triangle inequality yields Figure 11's budget.

This bound uses the learned envelope, unlike Appendix D's defect bound along computed trajectories, which uses the reference envelope. Neither follows solely from small empirical loss.

### H. A consistency construction with complete field data

For iid \(U_i\sim\mu\in\mathcal P_2(H)\), let
\(\mu_n=n^{-1}\sum_{i=1}^n\delta_{U_i}\). Choose \(m_n\to\infty\) and \(\varepsilon_n\to0\), and apply Appendix F to the projected empirical prototype law
\(\nu_n=(P_{m_n})_\#\mu_n\). Its exact endpoint is
\(\rho_n=\nu_n*\mathcal N_{H_{m_n}}(0,\varepsilon_n^2C_{m_n})\). Projection is a contraction, so

$$
W_{2,H}(\rho_n,\mu)
\le W_{2,H}(\mu_n,\mu)+\delta_{m_n}
+\varepsilon_n\sqrt{\operatorname{tr}C_{m_n}}.
$$

Empirical laws converge almost surely in \(W_2\) under the finite-second-moment assumption; the other terms vanish. This is an existence-level statistical consistency example with growing prototype storage. It supplies no dimension-independent sample rate, trained-network guarantee, or recovery theorem from finite noisy observations. For learned and numerical flows, the remaining two terms of the main error budget must also vanish.

### I. Operator approximation through a stable fixed point

The ICLR 2025 result concerns a fixed semilinear parabolic PDE and a varying initial function. In our notation, its assumptions include:

- A linear solution semigroup with smoothing estimate
  \(\|E(t)\|_{L^{b_1}\to L^{b_2}}\le C_E t^{-\nu(1/b_1-1/b_2)}\)
  for \(1\le b_1\le b_2\le\infty\) and \(0 < t\le 1\).
- A scalar \(C^1\) nonlinearity with \(\mathcal N(0)=0\) and
  \( |\mathcal N(z)-\mathcal N(w)|\le C_{\mathcal N}\max(|z|,|w|)^{p-1}|z-w|\), with \(p>1\).
- Output exponents \(q_t,q_x\in[p,\infty]\) satisfying
  \(\nu/q_x+1/q_t<1/(p-1)\).
- Convergence of both the initial-data and Duhamel Green-kernel expansions in the mixed norms specified in Assumption 3.

These yield a common sufficiently short interval and a contraction ball for the bounded initial-data family. The theorem controls error in \(\mathcal U=L^{q_t}(0,T;L^{q_x}(\Omega))\). Neither this norm alone nor pointwise differentiability of a network establishes convergence of spatial derivatives. The time-local argument also needs further bounds to continue over successive intervals ([Sections 2–4 and Appendix D](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)).

The following abstract estimate explains the mechanism. Let \(\Phi_a\) be \(q\)-contractive on a closed invariant ball \(\mathcal B\subset\mathcal U\), uniformly for \(a\in\mathcal A_R\), with fixed point \(\mathcal S(a)\). Suppose an approximate block obeys

$$
\sup_{a\in\mathcal A_R,\,v\in\mathcal B}
\|\widehat\Phi_a(v)-\Phi_a(v)\|_{\mathcal U}\le\eta,
$$

and its iterates remain in \(\mathcal B\). Then

$$
\begin{aligned}
e_{k+1}
&=\|\widehat\Phi_a(\widehat u^{(k)})-\Phi_a(\mathcal S(a))\|_{\mathcal U}\\
&\le\|\widehat\Phi_a(\widehat u^{(k)})-\Phi_a(\widehat u^{(k)})\|_{\mathcal U}
+\|\Phi_a(\widehat u^{(k)})-\Phi_a(\mathcal S(a))\|_{\mathcal U}\\
&\le\eta+q e_k.
\end{aligned}
$$

Induction gives

$$
e_J\le q^J e_0+\eta\sum_{j=0}^{J-1}q^j
=q^J e_0+\frac{1-q^J}{1-q}\eta.
$$

This uses contraction of the reference block and a bound on the approximate iterates. It does not assume the learned block itself is contractive, and it gives no stability guarantee for arbitrary unbounded iterates. The estimate is a standard perturbed-contraction argument; the paper's proof bounds its particular kernel and nonlinear defects and ensures the required ball membership.

For an engineering error budget, choose a forcing space \(\mathcal V\) on which \(K:\mathcal V\to\mathcal U\) is bounded. If \(\|\mathcal N(v)\|_{\mathcal V}\le M_{\mathcal N}\) on the ball and \(\|\mathcal N_\theta(v)-\mathcal N(v)\|_{\mathcal V}\le\delta_{\mathcal N}\), then the decomposition \(\Phi_a(v)=Ba+K\mathcal N(v)\) gives

$$
\eta\le
R\|B_N-B\|_{L^\infty\to\mathcal U}
+M_{\mathcal N}\|K_N-K\|_{\mathcal V\to\mathcal U}
+\|K_N\|_{\mathcal V\to\mathcal U}\delta_{\mathcal N}.
$$

If numerical evaluation introduces an additional uniform block defect \(\delta_{\rm quad}\), add it to this budget, while checking that numerical iterates stay in the same ball. This separates kernel truncation, nonlinear approximation, and quadrature. The forcing space and operator bounds must be established for the particular PDE; this abstract decomposition is not an extra theorem about every implementation.

Uniform approximation also bounds population risk on the admitted family: if \(\rho(\mathcal A_R)=1\) and \(\sup_a\|\widehat{\mathcal S}(a)-\mathcal S(a)\|_{\mathcal U}\le\varepsilon\), then \(\mathcal R(\widehat{\mathcal S})\le\varepsilon^2\). The converse fails in general: a small average error can hide poorly approximated inputs. Finite training data and an optimizer need their own analysis to connect the constructed operator to a trained one.

## References

1. Csillag, D., et al. [Functional Gradient Descent with Adaptive Representations](https://arxiv.org/abs/2606.16926). arXiv:2606.16926, 2026.
2. Kerrigan, G., Migliorini, G., and Smyth, P. [Functional Flow Matching](https://proceedings.mlr.press/v238/kerrigan24a.html). AISTATS, PMLR 238:3934–3942, 2024.
3. Bunker, J., et al. [Autoencoders in Function Space](https://www.jmlr.org/papers/v26/25-0035.html). JMLR 26(165):1–54, 2025.
4. Wang, S., et al. [FunDiff: diffusion models over function spaces for physics-informed generative modeling](https://doi.org/10.1038/s41467-026-72292-0). Nature Communications 17, 5749, 2026. Figure 8 uses the 2025 preprint version.
5. Serrano, L., et al. [Operator Learning with Neural Fields: Tackling PDEs on General Geometries](https://arxiv.org/abs/2306.07266). arXiv:2306.07266, 2023.
6. Yin, Y., et al. [Continuous PDE Dynamics Forecasting with Implicit Neural Representations](https://arxiv.org/abs/2209.14855). ICLR, 2023.
7. Serrano, L., et al. [AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields](https://arxiv.org/abs/2406.02176). arXiv:2406.02176, 2024.
8. Wang, S., Wu, Z., van Dijk, D., and Lu, L. [GeoFunFlow: Geometric Function Flow Matching for Inverse Operator Learning over Complex Geometries](https://arxiv.org/abs/2509.24117). arXiv:2509.24117, 2025.

9. Stuart, A. M. [Inverse problems: A Bayesian perspective](https://doi.org/10.1017/S0962492910000061). Acta Numerica, 2010.
10. Chen, Y., and Vanden-Eijnden, E. [Scale-Adaptive Generative Flows for Multiscale Scientific Data](https://arxiv.org/abs/2509.02971v2). arXiv:2509.02971v2, 2026 revision.
11. Fotiadis, S., et al. [Adaptive Flow Matching for Resolving Small-Scale Physics](https://proceedings.mlr.press/v267/fotiadis25a.html). ICML, 2025.
12. Koch, O., and Lubich, C. [Dynamical Low-Rank Approximation](https://doi.org/10.1137/050639703). SIAM Journal on Matrix Analysis and Applications, 2007.
13. Baldan, G., et al. [Physics vs Distributions: Pareto Optimal Flow Matching with Physics Constraints](https://arxiv.org/abs/2506.08604v3). arXiv:2506.08604v3.
14. Bhola, S., and Duraisamy, K. [Residual-augmented flow matching operators for probabilistic partial differential equations](https://arxiv.org/abs/2512.12749). arXiv:2512.12749.
15. Zhao, Y., Lu, H., Jia, J., and Zhou, T. [Functional normalizing flow for statistical inverse problems of partial differential equations](https://arxiv.org/abs/2411.13277). arXiv:2411.13277.
16. Kovachki, N., et al. [Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs](https://jmlr.org/papers/v24/21-1524.html). JMLR 24(89):1–97, 2023.
17. Bruna, J., Peherstorfer, B., and Vanden-Eijnden, E. [Neural Galerkin Schemes with Active Learning for High-Dimensional Evolution Equations](https://doi.org/10.1016/j.jcp.2023.112588). Journal of Computational Physics 496, 112588, 2024.
18. Lee, K., and Carlberg, K. T. [Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders](https://doi.org/10.1016/j.jcp.2019.108973). Journal of Computational Physics 404, 108973, 2020.
19. Raissi, M., Perdikaris, P., and Karniadakis, G. E. [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561). arXiv:1711.10561, 2017.
20. Lu, L., et al. [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://doi.org/10.1038/s42256-021-00302-5). Nature Machine Intelligence 3, 218–229, 2021.
21. Krishnapriyan, A. S., et al. [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050). NeurIPS, 2021.
22. Li, Z., et al. [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895). ICLR, 2021.
23. Shikhman, L. J. [Discretization and Statistical Consistency of Functional Flow Matching](https://arxiv.org/abs/2608.04531v1). arXiv:2608.04531v1, August 2026 preprint.
24. Koehler, F., Mehta, V., and Risteski, A. [Representational aspects of depth and conditioning in normalizing flows](https://proceedings.mlr.press/v139/koehler21a.html). ICML, PMLR 139:5628–5636, 2021.
25. Verine, A., et al. [On the expressivity of bi-Lipschitz normalizing flows](https://proceedings.mlr.press/v189/verine23a.html). PMLR 189:1054–1069, 2023.
26. Bandeira, A. S., Singer, A., and Strohmer, T. [Topics in Mathematics of Data Science](https://www.math.ucdavis.edu/~strohmer/papers/2025/MDS_Book.pdf). Preprint v1.0, 2025, Proposition 8.8.

27. Furuya, T., Taniguchi, K., and Okuda, S. [Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4b6ccf3acd6ccbc1093e093df345ba2-Abstract-Conference.html). ICLR, 2025; preprint first posted in 2024.
28. Calvello, E., Kovachki, N. B., Levine, M. E., and Stuart, A. M. [Continuum Attention for Neural Operators](https://www.jmlr.org/papers/v26/24-0879.html). JMLR 26(300):1–52, 2025.

## Cite this note

{{< cite-note >}}
