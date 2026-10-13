---
title: "How and why we redefine latent representations for PDE solutions as functional learning?"
date: 2026-10-01T10:47:44-04:00
draft: false
tags: [functional-learning, pdes, neural-networks, flow-matching]
summary: "Why PDE learning needs a functional view of latent codes, and how that view connects solvers, physical constraints, uncertainty, and adaptive transport."
showtoc: true
tocopen: false
UseHugoToc: true
math: true
ShowReadingTime: true
publishDate: 2026-10-01T10:47:44-04:00
completed: true
---

*Click any figure to open its full-resolution image.*

Suppose we learn temperature fields from simulations. Different simulations store different meshes, but we want to describe the same physical quantity. We may need its gradient to compute heat flux, predict its response to new initial conditions, or reconstruct it from incomplete measurements. Each use asks something different of the learned field.

Neural PDE models often compress these observations into latent vectors. With a coordinate-query decoder, fixing the code \(z\) defines a whole function \(u_z(x)\). Its meaning comes from the code and decoder together. The important question is what this represented function lets us compute accurately.

**Functional learning makes the field, its physical operations, and its required accuracy part of the learning problem.** We begin with those requirements, explain how arrays and codes describe functions, and then develop three learning tasks: finding a function, learning a map between functions, and generating a law over functions. Examples from functional gradient descent (FGD), functional operator learning (FOL), and functional flow matching (FFM) make these tasks concrete. Their shared ingredients lead to model design, accuracy requirements, and the open questions at the end.

## 1. What do we need from a learned PDE solution?

A solver returns a numerical description of a field. Our learning task may need to change how that field is sampled, apply physical operators to it, or use it under new conditions. [FunDiff](https://doi.org/10.1038/s41467-026-72292-0) and related function-representation models bring together three requirements: heterogeneous observations, physical operations, and a common field across tasks. Incomplete observations add a fourth requirement: an explicit account of uncertainty.

### Learning from simulations on different meshes

Suppose one simulation stores a field on a \(64\times64\) mesh and another uses \(128\times128\). Their arrays have different sizes, but both sample a function on the physical domain. Write

$$
y_i=\mathcal O_i u_i+\varepsilon_i.
$$

Here \(\mathcal O_i\) records the sampling rule, such as point values or cell averages, and \(\varepsilon_i\) is observation error. For point values, choose a function space \(U\) with enough regularity to define evaluation; arbitrary \(L^2\) equivalence classes do not supply point values.

### Computing derivatives, residuals, and physical quantities

A PDE acts on functions through operations such as differentiation, integration, and boundary evaluation. For a temperature field with thermal conductivity \(k\), the heat flux \(-k\nabla u\) depends on a derivative. A small error in stored field values need not imply a small error in that flux.

This distinction has a simple mathematical example. On \((0,2\pi)\), take the error function

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

### Reusing a field across prediction, reconstruction, and inference

A representation of a field can be reused when the task changes, provided it supports the operations the new task requires. The representation supplies a common object; each task still needs its own trained map, dynamics, or inference model.

### Describing uncertainty when observations are incomplete

If two admissible fields produce the same measurements, a deterministic reconstruction cannot identify which one was observed. A prior and an observation model can instead specify a conditional distribution over possible fields. This is uncertainty about the physical state given the available information, even when a fully specified forward PDE has a unique solution.

These requirements explain the three learning tasks ahead. We may need to fit one field, predict fields as problem data change, or sample possible fields given a context. To state their accuracy consistently, we first need to separate the function from its numerical coordinates.

## 2. From arrays and latent codes to functions

The physical state is a function; samples, basis coefficients, and latent codes describe it for computation. Many established numerical and neural methods already use this interpretation. The aim is to carry it through the operations and accuracy requirements of the whole model.

### What existing PDE models already provide

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

The limitations in the table identify requirements to examine, rather than failures of every method in a family.

### How a code and decoder define a field

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

[CORAL](https://arxiv.org/abs/2306.07266) makes this interpretation explicit. In Figure 1, a processor maps input function codes to output codes, and the decoder evaluates the output function at query locations. Changing those locations changes the returned samples of the function.

{{< figure src="figures/coral-function-codes.png" link="figures/coral-function-codes.png" alt="CORAL diagram: sampled input values are encoded into a code, a processor maps it to an output code, and a coordinate-query decoder evaluates the output function." caption="**Figure 1.** CORAL maps an input function code to an output code, then evaluates the output function at query coordinates. Source: [Serrano et al., Figure 2](https://arxiv.org/abs/2306.07266v2)." >}}

A fixed decoder selects a family \(\mathcal M=D(Z)\subset H\). Training the decoder changes the family; updating a code moves within it. [Autoencoders in Function Space](https://www.jmlr.org/papers/v26/25-0035.html) uses this separation to define reconstruction at the function level.

The code need not be a learned embedding. **The finite values or coefficients we store are also \(z\)**: sampled values with an interpolator, finite-element coefficients, tree-leaf values, Fourier-bin values, and voxel attributes all describe functions through a synthesis rule. For an adaptive representation, write

$$
z_m\in Z_m=\mathbb R^{d_m},
\qquad u_m=D_m(z_m).
$$

Here \(m\) identifies the partition, basis, or decoder, and \(d_m\) counts the real coordinates; complex coefficients can be stored as real and imaginary parts. Refinement can increase \(d_m\), so the code itself grows. Increasing the number of query locations only returns more samples of the existing \(u_m\). We will see three concrete ways to expand a representation in FGD.

### Three learning objects: a function, an operator, and a law

There are several things we might learn with such a representation:

| Object | Mathematical question | Example |
| --- | --- | --- |
| One function | \(\min_{u\in H}\mathcal L(u)\) | Solve a PDE by residual minimization |
| A map between functions | Learn \(\mathcal S:a\mapsto u\) | Map coefficients or initial data to solutions |
| A law over functions | Generate \(u\sim\mu\) | Sample a family of physically admissible fields |

For the first problem, we need an update that reduces a functional \(\mathcal L(u)\). For the second, we need a shared map that takes different input functions to their corresponding solutions. For the third, we need dynamics that produce the intended distribution. These lead to functional gradient descent, operator learning, and functional flow matching, respectively.

**Notation.** We use \(t\) for physical time, \(s\) for optimization time, and \(\tau\) for generative time. A space-time field \(u(x,t)\) can be one state in an optimization or generative process; updating it changes the entire field.

## 3. Three tasks of functional learning

We now develop the mathematics of these three targets: a function specified by an objective, a solution map specified by a PDE family, and a law specified by data and conditioning information.

### Functional optimization (FGD): finding one solution {#functional-gradient-descent}

Functional optimization treats the candidate function as the unknown. [Functional Gradient Descent](https://arxiv.org/abs/2606.16926) connects its updates to geometry and representation error. We derive the gradient, compare it with PINN parameter updates, and examine how its representation adapts.

#### From a directional derivative to a gradient

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

A neural PDE model also describes a function, but its optimizer usually updates network weights. A single PDE loss lets us compare that familiar gradient with the functional gradient above.

#### PINN parameter updates versus functional updates {#the-same-pde-energy-under-two-geometries}

The neural-network representation in the [FGD paper's introduction](https://arxiv.org/html/2606.16926v1#S1) is a familiar starting point: represent the unknown by \(u_\theta\), then optimize the weights \(\theta\). A PINN does this with a loss built from network values and PDE derivatives at training points. We can compare that parameter update with a functional update while keeping the equation and loss the same.

Suppose we want to find a scalar field \(u^*\) on a bounded domain \(\Omega\), given a source \(f\), by solving

$$
\begin{cases}
-\Delta u^*(x)=f(x), & x\in\Omega,\\
u^*(x)=0, & x\in\partial\Omega.
\end{cases}
$$

This Poisson problem can describe a steady temperature field with unit conductivity and zero boundary temperature. For simplicity, let the representation enforce the boundary condition exactly. At interior collocation points \(x_i\), define the residual and sampled loss

$$
r_i(u)=-\Delta u(x_i)-f(x_i),
\qquad
\mathcal L_N(u)=\frac12\sum_{i=1}^N w_i r_i(u)^2,
\qquad w_i>0.
$$

A PINN evaluates \(u_\theta\) and its derivatives by automatic differentiation, assembles \(\mathcal L_N(u_\theta)\), and backpropagates to the weights. Boundary penalties or field-data terms can be included in the same calculation; we omit them here to keep one mechanism visible ([Raissi et al., Section 2.1](https://arxiv.org/abs/1711.10561)).

To define a functional gradient of exactly this sampled loss, choose a Hilbert space \(H\) of fields satisfying the boundary condition, with enough regularity that \(h\mapsto-\Delta h(x_i)\) is bounded. A sufficiently regular Sobolev space or smooth RKHS can provide this; bare \(L^2\) does not. Perturbing the field gives

$$
D\mathcal L_N(u)[h]
=\sum_i w_i r_i(u)\,[-\Delta h(x_i)].
$$

Let \(q_i\in H\) be the Riesz representative of the residual evaluation: \(\langle q_i,h\rangle_H=-\Delta h(x_i)\). The functional gradient is therefore the whole correction field

$$
g_N(u):=\nabla_H\mathcal L_N(u)
=\sum_i w_i r_i(u)q_i.
$$

Now insert a differentiable neural representation \(u_\theta\in H\), with \(\theta\in\mathbb R^p\). Its Jacobian maps a small weight change into a field change:

$$
J_\theta:\mathbb R^p\to H,
\qquad
J_\theta\delta\theta
=\sum_{j=1}^p\frac{\partial u_\theta}{\partial\theta_j}\delta\theta_j.
$$

With \(J_\theta^*\) the adjoint for the chosen field inner product and Euclidean parameter inner product, the chain rule gives

$$
\boxed{\nabla_\theta\mathcal L_N(u_\theta)
=J_\theta^*\nabla_H\mathcal L_N(u_\theta).}
$$

This is the connection between the two gradients. To isolate the effect of the representation, compare ordinary Euclidean gradient descent on \(\theta\) with ideal functional descent:

| Same PDE and sampled loss | Neural representation / PINN | Ideal functional descent |
| --- | --- | --- |
| Unknown being updated | Weights \(\theta\) in \(u_\theta\) | The field \(u\) |
| Gradient | \(J_\theta^*g_N\) | \(g_N=\nabla_H\mathcal L_N\) |
| Descent step | \(\theta^+=\theta-\eta J_\theta^*g_N\) | \(u^+=u-\eta g_N\) |
| Field motion in gradient flow | \(\dot u=-J_\theta J_\theta^*g_N\) | \(\dot u=-g_N\) |
| Finite computation | Differentiate through the network | Represent and approximate the correction field |

Each weight affects a whole function \(\partial_{\theta_j}u_\theta\). Parameter descent measures the alignment of \(g_N\) with these available directions, then combines them to move the field. It can suppress directions outside their span and rescale directions within it. **\(J_\theta J_\theta^*\) is generally neither the identity nor an orthogonal projector.** The resulting kernel-mediated dynamics are also studied in [PINN neural tangent kernel analysis, Section 3.1](https://arxiv.org/abs/2007.14527); the Jacobian identity itself does not require an infinite-width limit.

Thus a zero parameter gradient \(J_\theta^*g_N=0\) need not imply a zero functional gradient \(g_N=0\). Functional descent still requires its correction field to be computed and represented. Both methods inherit the information limits of the sampled loss; matching finitely many residual values does not establish accuracy everywhere.

#### Adaptive approximation of the gradient

An ideal update is \(u_{k+1}=u_k-\eta\nabla_H\mathcal L(u_k)\). Numerically, we use a finitely represented approximation \(\widehat g_k\) to that gradient.

Adaptive FGD refines this representation until gradient error is small relative to the update. When the iterate and approximate gradient both lie in \(H\), a sufficient condition is

$$
\|\widehat g_k-\nabla_H\mathcal L(u_k)\|_H
\leq\varepsilon\|\widehat g_k\|_H.
$$

Write \(L_{\rm sm}\) for the loss smoothness constant—the paper's \(K\), distinct from a kernel. It bounds the quadratic remainder of the loss in the chosen norm. In the Hilbert-space case, the descent calculation gives

$$
\begin{aligned}
\mathcal L(u_k-\eta\widehat g_k)
&\leq\mathcal L(u_k)\\
&\quad-\eta\left(1-\varepsilon-\frac{L_{\rm sm}\eta}{2}\right)
\|\widehat g_k\|_H^2.
\end{aligned}
$$

The positive coefficient ensures descent. As the gradient shrinks, a fixed approximation error can overwhelm it; the relative criterion tightens accuracy accordingly. The estimate follows from \(\langle\nabla_H\mathcal L,\widehat g_k\rangle_H\ge(1-\varepsilon)\|\widehat g_k\|_H^2\) and the smoothness inequality.

For the toy reconstruction loss \(\mathcal L(u)=\frac12\|u-u^\star\|_{L^2}^2\), the exact gradient is \(u-u^\star\). Its closed form can still require approximation in the chosen finite representation. Fixed representations plateau while adaptation adds detail (Figure 2). This loss differs from the sampled RKHS loss below.

{{< figure src="figures/fgd-adaptive.png" link="figures/fgd-adaptive.png" alt="Three reconstruction sequences and training-loss curves comparing a neural network, fixed-resolution functional gradients, and adaptive functional gradients." caption="**Figure 2.** Reconstructions and loss curves in the FGD toy example, comparing neural, fixed, and adaptive representations. Source: [Csillag et al., Figure 1](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

The paper also allows approximations in a larger Banach space \(B\). At a nonstationary iterate, Algorithm 1 holds the current function fixed, fits \(\widehat g_m\), and computes

$$
U_m\ge\|\widehat g_m-\nabla\mathcal L(u)\|_B,
\qquad S_m=\|\widehat g_m\|_B.
$$

It refines, transfers the current function, and refits until

$$
\boxed{(1+\epsilon)U_m<\epsilon S_m,\qquad 0<\epsilon<1,}
$$

then applies \(u^+=u-\eta\widehat g_m\). This compares gradient error with gradient size; multiplying both by \(\eta\) gives the same relative update error. Smoothness controls the step size, while this test controls representation accuracy. The Banach-space theorem additionally needs the paper's gradient-compatibility assumptions ([Algorithm 1 and Section 3](https://arxiv.org/html/2606.16926v1#S3)).

Functional gradients and inexact-gradient methods predate this paper. Its distinctive contribution is adaptive gradient representations with computable error tests and convergence guarantees under its stated assumptions.

#### Three representations for functional updates {#two-applications-wave-equations-and-inverse-rendering}

FGD's three main experiments use different coordinates for their functions. **Only the wave-equation example uses a Fourier representation.** The following descriptions combine the paper's [Section 4 and Appendix B](https://arxiv.org/html/2606.16926v1#S4) with static inspection of the authors' [released implementation](https://github.com/dccsillag/experiments-adaptive-fgd/tree/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc); they do not reproduce the reported runs.

##### Case 1: RKHS regression with a tree representation

The first experiment fits observations \((X_i,Y_i)\). Its numerical representation partitions the input-feature domain into rectangular tree leaves \(C_j\), storing one value \(z_j\) per leaf:

$$
D_m(z)(x)=\sum_{j=1}^{d_m}z_j\mathbf1_{C_j}(x).
$$

An **RKHS**, or reproducing kernel Hilbert space, is a Hilbert space of functions in which evaluating a function at a fixed input is a bounded linear operation. By the Riesz theorem, evaluation therefore has a representative:

$$
u(X)=\langle u,K(X,\cdot)\rangle_{\mathcal H_K},
\qquad
|u(X)|\le\sqrt{K(X,X)}\,\|u\|_{\mathcal H_K}.
$$

The kernel function \(K(X,\cdot)\) reproduces the value at \(X\) through an inner product. In particular, \(K(X,Y)=\langle K(X,\cdot),K(Y,\cdot)\rangle_{\mathcal H_K}\): kernel values measure alignment between evaluation directions. The bound follows from Cauchy–Schwarz.

This makes a loss on observed point values compatible with the function-space calculus. Bare \(L^2\) does not provide this: changing a value at one point leaves its norm unchanged, and point evaluation is not well-defined on its equivalence classes.

For the RBF kernel below, \(K(X,\cdot)\) is a smooth bump centered at \(X\); \(\gamma\) controls its width. An observation's loss derivative therefore contributes a bump to the whole correction field. Overlapping bumps make observations influence updates away from their own locations.

The kernel specifies this gradient geometry. The tree separately stores the current function and approximates the correction. With residuals \(r_i=u_m(X_i)-Y_i\), the half-MSE loss is

$$
\mathcal L(u)=\frac1{2N}\sum_i(u(X_i)-Y_i)^2.
$$

For an arbitrary perturbation \(h\in\mathcal H_K\),

$$
\begin{aligned}
D\mathcal L(u_m)[h]
&=\frac1N\sum_i r_i h(X_i)\\
&=\left\langle
\frac1N\sum_i r_iK(X_i,\cdot),h
\right\rangle_{\mathcal H_K}.
\end{aligned}
$$

This holds for every \(h\), so no directions need to be enumerated. The residuals are scalar weights; the kernel functions supply the directions. Riesz representation identifies the correction field

$$
g(x)=\frac1N\sum_i r_i K(X_i,x),
\qquad K(X_i,x)=\exp(-\gamma\|X_i-x\|^2).
$$

This kernel sum is **exact**. At a sample, \(g(X_j)=N^{-1}\sum_i r_iK(X_i,X_j)\): residuals elsewhere contribute too. The identity \(\nabla_H\mathcal L(u)=u-u^\star\) instead holds for \(\mathcal L=\frac12\|u-u^\star\|_H^2\), even in an RKHS. Approximation error arises when the tree fits \(g\) as a piecewise-constant \(\widehat g\).

This is differentiation with respect to the **function's values**, not spatial differentiation of the tree. No \(\partial_xu_m\) is required. For cross-entropy, the scalar loss derivative replaces \(r_i\). The paper permits the tree and approximate gradient to live in a larger space of bounded functions, even though they generally do not belong to the RKHS defining the gradient ([Section 4.1 and Appendix A.1.2](https://arxiv.org/html/2606.16926v1#S4.SS1)).

Half-MSE has \(L_{\rm sm}=1\) in the supremum norm: its quadratic remainder is \((2N)^{-1}\sum_i h(X_i)^2\le\frac12\|h\|_\infty^2\). In the RKHS norm its sharp constant is instead \(\lambda_{\max}(G)/N\), where \(G_{ij}=K(X_i,X_j)\). The code's logit cross-entropy has a supremum-norm bound \(L_{\rm sm}\le1/4\); probability-valued cross-entropy needs a range away from zero and one for a finite uniform bound.

The tree fits \(b_j=g(c_j)\) at each leaf center. With \(R_j\) its radius, the RBF kernel gives a spatial Lipschitz bound and a computable error bound:

$$
L_g^{\rm space}=
\frac1N\sum_i|r_i|\sqrt{\frac{2\gamma}{e}},
\qquad
U_m=L_g^{\rm space}\max_jR_j,
\qquad S_m=\max_j|b_j|.
$$

This spatial bound differs from loss smoothness. On the represented domain, \(U_m\) bounds the sup-norm gradient error. The code accepts \(U_m/S_m<1/3\), or exits at depth 16. Otherwise it splits **every leaf** along its widest interval, copies parent values, and refits before updating \(z_j^+=z_j-\eta b_j\) ([regression code](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/kernel-regression/ours.py#L154-L287)).

{{< figure src="figures/fgd-tree-representation.svg" link="figures/fgd-tree-representation.svg" width="720" alt="A tree partition with two constant leaf values is split into four leaves by copying parent values. The same observation locations remain in place. A subsequent kernel-gradient update changes the four values independently." caption="**Figure 3.** What grows in regression: the vector of leaf values. Copying preserves the current function; a newly fitted gradient changes it. The numerical values illustrate the two stages and are not experimental results. Original LaTeX/TikZ schematic based on the [released regression implementation](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/kernel-regression/ours.py)." >}}

Direct Euclidean descent on those same leaf values would instead use \(\partial\mathcal L/\partial z_j=N^{-1}\sum_{X_i\in C_j}r_i\). FGD first forms the kernel-weighted correction field from all observations, then approximates it with the tree. This is another concrete instance of the distinction between a function-space gradient and a coefficient gradient.

##### Case 2: a wave solution from Fourier-bin values

The second experiment optimizes the whole space-time solution \(u(t,x,y)\) of

$$
\partial_t^2u=c^2\Delta u,
\qquad u(0,x,y)=h(x,y),
\qquad \partial_tu(0,x,y)=0.
$$

The loss combines its wave-equation residual with initial-condition penalties, much as a space-time PINN objective does ([FGD, Section 4.2](https://arxiv.org/html/2606.16926v1#S4.SS2)).

For the paper's loss, collect the wave residual and two initial traces into a bounded linear operator \(A:H^2\to Y\). Then \(\mathcal L(u)=\frac12\|Au-b\|_Y^2\), so \(L_{\rm sm}=\|A\|^2\) is a smoothness bound. Determining its numerical value requires the operator norms and the chosen normalizations.

Here the stored values describe the Fourier transform, not physical grid-point values. For frequency cells \(Q_j\), write

$$
\widehat u_m(\kappa)=\sum_jz_j\mathbf1_{Q_j}(\kappa),
\qquad D_m(z)=\mathcal F^{-1}\widehat u_m,
$$

where \(\kappa\) contains one temporal and two spatial frequencies. The coefficients \(z_j\) are complex. An inverse-transformed frequency box is a modulated product of sinc functions, so constants in frequency space produce a continuous field in physical space. PDE derivatives can be evaluated through this synthesis; under the angular-frequency convention, for example, \(\mathcal F[\partial_t^2u]=-\kappa_0^2\widehat u\). The paper derives the \(H^2\)-gradient directly in Fourier coordinates.

The code fits the Fourier gradient at bin centers, then samples eight random offsets per bin to estimate

$$
\widehat r_m^2=
\frac{\sum_j|Q_j|\,\operatorname{mean}_{q}
 w(\kappa_{jq})|\widehat g_m(\kappa_{jq})-\widehat g(\kappa_{jq})|^2}
{\sum_j|Q_j|\,\operatorname{mean}_{q}
 w(\kappa_{jq})|\widehat g_m(\kappa_{jq})|^2},
\quad w(\kappa)=1+|\kappa|^2+|\kappa|^4.
$$

If the test fails, it doubles all three axes, copies parents into eight children, and refits. The frequency box remains fixed. The inspected script stops at \(\widehat r_m\le50\) or side length 128 and uses \(\eta=0.1\). **That threshold does not enforce Algorithm 1's small relative error.** The sampled test also omits an outside-box error bound; Appendix B.2 describes a stronger protocol ([fitting and test](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/wave-equation/sandbox_ours.py#L195-L292), [loop](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/wave-equation/sandbox_ours.py#L424-L433)).

{{< figure src="figures/fgd-fourier-representation.svg" link="figures/fgd-fourier-representation.svg" width="720" alt="A two-dimensional slice of a Fourier grid shows coarse constant bins, finer bins holding copied parent values, and independently changed fine-bin coefficients after descent. An inverse Fourier transform decodes these values into a physical field." caption="**Figure 4.** What grows in the wave solve: frequency-bin coefficients. The full representation has three frequency axes; the diagram shows a two-dimensional slice. Refinement preserves the transform before the gradient update. Original LaTeX/TikZ schematic based on the [released wave implementation](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/wave-equation/sandbox_ours.py)." >}}

Figure 5 shows physical-time slices of the optimized function. The plotted sample count is separate from its coefficient count.

{{< figure src="figures/fgd-wave-solution.png" link="figures/fgd-wave-solution.png" alt="Six physical-time snapshots of a wave solution: neural network approximation, adaptive functional gradient descent, and reference solution in three rows." caption="**Figure 5.** Physical-time slices of the wave solution: neural approximation, adaptive FGD, and reference, from top to bottom. Source: [Csillag et al., Figure 3](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

##### Case 3: radiance fields from voxels and spherical harmonics

Inverse rendering changes the task operator. The unknown functions describe scene density \(\sigma(x)\) and view-dependent RGB color \(c(x,\omega)\); a differentiable renderer predicts the images used in the loss ([FGD, Section 4.3](https://arxiv.org/html/2606.16926v1#S4.SS3)). The rendering operator is nonlinear, so the function-space objective remains nonconvex.

In the default piecewise-constant voxel representation, a spatial cell \(V_v\) stores a scalar density and RGB spherical-harmonic coefficients:

$$
\sigma(x)=\sigma_v,\qquad
c(x,\omega)=\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}
a_{v,\ell m}Y_{\ell m}(\omega),\qquad x\in V_v.
$$

Here \(a_{v,\ell m}\in\mathbb R^3\). The spherical harmonics encode variation with viewing direction; the voxel index encodes variation across space. Flattening these attributes gives \(z_m\). For an \(n^3\) grid, its real coordinate count is

$$
d_m=n^3\bigl[1+3(L+1)^2\bigr].
$$

Loss smoothness here requires a uniform bound on the rendering loss's Hessian over admissible states. Its curvature depends on the current fields; the inspected code does not compute a global \(L_{\rm sm}\).

The default code samples eight spatial points per voxel, averages density gradients, and fits color gradients in the fixed spherical-harmonic basis. It reports a volume-weighted fitting-loss estimate \(E_{\rm fit}\), then tests

$$
\widehat r_m=
\sqrt{\frac{E_{\rm fit}}{S_\sigma^2+S_c^2}},
\qquad
S_\sigma^2+S_c^2=
\sum_v|V_v|\left[
\widehat g_{\sigma,v}^2+
\sum_{\ell,m}\|\widehat a_{v,\ell m}\|_{\mathbb R^3}^2
\right].
$$

It accepts \(\widehat r_m<0.4\), or exits when a grid side reaches 128 or fixed-grid mode is enabled. Otherwise it doubles every spatial axis, copies attributes, and refits before applying separate density/color learning rates. **The angular degree stays fixed.** \(E_{\rm fit}\) uses sampled density and angular losses; it is not a certified bound on the full gradient error ([fitting loss](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/functional_radiance/models/our_model/representation.py#L617-L719), [training loop](https://github.com/dccsillag/experiments-adaptive-fgd/blob/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc/src/functional_radiance/models/our_model/train.py#L583-L647)).

{{< figure src="figures/fgd-voxel-representation.svg" link="figures/fgd-voxel-representation.svg" width="720" alt="A spatial voxel is divided into eight children that inherit its density and RGB spherical-harmonic coefficient tuple. Each voxel stores density plus fixed-degree angular coefficients; only spatial resolution increases." caption="**Figure 6.** What grows in inverse rendering: the number of voxel attribute tuples in z. The spherical-harmonic basis stays fixed. The diagram shows one parent; the implementation refines the whole grid. Copying alone preserves the default fields, while additional visibility masking in the training loop can change density. Original LaTeX/TikZ schematic based on the released implementation." >}}

Figure 7 shows finer plant leaves emerging during optimization and compares test losses for the Ficus scene.

{{< figure src="figures/fgd-inverse-rendering.png" link="figures/fgd-inverse-rendering.png" alt="Novel-view renderings of a potted plant through optimization iterations, comparing a neural network, fixed-grid FGD, and adaptive FGD, with test-loss curves." caption="**Figure 7.** Inverse rendering of the Ficus scene through optimization iterations, with test-loss curves for neural, fixed, and adaptive FGD representations. Source: [Csillag et al., Figure 4](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

These checked-in settings are not recovered configurations of the plotted runs. The paths do not compute \(L_{\rm sm}\) to choose learning rates; the paper reports tuning them ([Section 4](https://arxiv.org/html/2606.16926v1#S4)). Verified error bounds and suitable step sizes are still required by the theorem.

In each case, \(z_m\) describes one function and adaptation expands its possible updates. Changing the problem data changes the desired solution; the next task learns a map across problems.

### Functional operator learning (FOL): predicting across problems {#functional-operator-learning}

Operator learning fits a shared map between input and output function spaces \(\mathcal A\) and \(\mathcal U\):

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

Here \(\rho\) describes the expected input problems. The same \(\theta\) serves every input. FGD fits one function; FOL fits a map between functions, and can train it by ordinary parameter descent.

DeepONet and FNO are established examples of **functional operator learning (FOL)**; FOL names the task. Our mathematical example is **Furuya, Taniguchi, and Okuda's [Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4b6ccf3acd6ccbc1093e093df345ba2-Abstract-Conference.html)**. Its construction connects operator layers to a convergent solution procedure and controls their approximation errors.

#### From a PDE to a fixed-point map

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

A short time interval can make this possible. The paper uses semigroup smoothing estimates to establish contraction in mixed Lebesgue norms, under its nonlinearity and kernel-expansion assumptions.

#### Turning the solution procedure into an operator network

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

Each block carries the initial-data term, applies a kernel across locations, and evaluates \(\mathcal N_\theta\) pointwise. The construction reuses this block; general trained operators need not perform Picard iteration ([Section 4.1 and Remark 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)).

{{< figure src="figures/functional-operator-picard.svg" link="figures/functional-operator-picard.svg" width="720" alt="Two parallel constructions map initial data to an entire trajectory. Exact Picard iteration converges to the PDE solution; finite kernel and neural approximations produce an operator network. A bound separates iteration error from block approximation error." caption="**Figure 8.** A solution map built from a repeated functional update. The upper row is exact Picard iteration; the lower row approximates its blocks. The same construction serves a family of initial functions. Original LaTeX/TikZ exposition of the mechanism in [Furuya et al., Section 4.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)." >}}

We can see the error mechanism without inspecting all the network weights. Suppose the approximate block has uniform defect at most \(\eta\) on the invariant ball, and its iterates remain there. With \(e_k=\|\widehat u^{(k)}-\mathcal S(a)\|_{\mathcal U}\), the triangle inequality gives

$$
e_{k+1}\le q e_k+\eta,
\qquad
\boxed{e_J\le q^J e_0+\frac{1-q^J}{1-q}\eta.}
$$

Depth reduces unfinished iteration, the first term. Improving kernels, nonlinearities, or quadrature reduces the second term, the block-approximation error amplified by stability. **Depth alone cannot remove the approximation floor.**

#### What the FOL approximation theorem guarantees

Under the paper's semigroup, nonlinearity, and kernel-expansion assumptions, each initial-data radius \(R\) admits a sufficiently short \(T\). For any \(\varepsilon\in(0,1)\), a ReLU neural operator exists with

$$
\sup_{\|a\|_{L^\infty(\Omega)}\le R}
\|\widehat{\mathcal S}(a)-\mathcal S(a)\|_{L^{q_t}(0,T;L^{q_x}(\Omega))}
\le\varepsilon.
$$

**This is uniform approximation of a solution map**, stronger than fitting one input ([Theorem 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)).

The kernel rank \(N\) is a separate cost, and the theorem supplies no general rate for \(N(\varepsilon)\). Its result is local in time and measured in the stated mixed Lebesgue norm. Derivative accuracy, long-time stability, generalization from finite data, and successful optimization require additional arguments. The theorem constructs approximating weights; it does not prove that training discovers them.

Operator approximation can use Banach spaces. Its output norm must control the physical quantities we need; a Riesz gradient additionally requires Hilbert structure.

#### A contemporary architectural connection

**Neural parameters are compatible with functional learning; the architectural advance is defining what each layer does to a function, then making its token computation consistently approximate that operation.** [Continuum Attention for Neural Operators (Calvello et al., 2025)](https://www.jmlr.org/papers/v26/24-0879.html) applies this perspective to attention, independently of the Picard construction above.

With learned query, key, and value matrices \(Q,K,V\), set \(s_v(x,y)=\langle Qv(x),Kv(y)\rangle\). For a feature function \(v\),

$$
\begin{aligned}
\mathcal A(v)(x)
&=\frac{\int_\Omega e^{s_v(x,y)}Vv(y)\,dy}
{\int_\Omega e^{s_v(x,y)}\,dy},\\
\mathcal A_h(v)(x_i)
&=\frac{\sum_j w_j e^{s_v(x_i,x_j)}Vv(x_j)}
{\sum_j w_j e^{s_v(x_i,x_j)}}.
\end{aligned}
$$

The first line defines continuum attention; the second approximates it by quadrature. Equal positive \(w_j\) recover conventional softmax. On uneven meshes, appropriate weights approximate spatial volume; unweighted attention follows sampling density. This is a quadrature interpretation; the paper proves convergence for independent uniform samples ([Definitions 4–5 and Theorem 6](https://www.jmlr.org/papers/volume26/24-0879/24-0879.pdf)).

Figure 9 applies attention blocks to feature functions. Its output \(z(x)\) denotes a function; our \(z\) denotes a latent vector.

{{< figure src="figures/continuum-attention-architecture.png" link="figures/continuum-attention-architecture.png" alt="Transformer neural operator architecture: an input function and coordinates are lifted, processed by repeated attention encoder layers, and projected to an output function." caption="**Figure 9.** An implemented operator architecture with attention acting on feature functions. Source: [Calvello et al., Figure 1](https://www.jmlr.org/papers/v26/24-0879.html), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

#### Example: mapping a Darcy coefficient field to its solution

For a uniformly positive coefficient \(a(x)\), consider

$$
-\nabla\cdot(a(x)\nabla u(x))=1
\quad\text{in }(0,1)^2,
\qquad u|_{\partial\Omega}=0.
$$

Figure 10 shows two coefficient functions and their predicted solutions from the same model. Its columns show inputs, reference solutions, predictions, and logarithmic absolute errors. The rows are the test samples with median and maximum relative \(L^2\) error.

{{< figure src="figures/continuum-attention-darcy.png" link="figures/continuum-attention-darcy.png" alt="Darcy operator-learning examples with input fields, reference solutions, predictions, and log-scale pointwise error for the median and maximum relative-error samples." caption="**Figure 10.** Darcy predictions from the Fourier attention neural operator variant in the continuum-attention paper. These are experiments from a separate 2025 operator-learning source, rather than numerical results of the parabolic approximation theorem. Source: [Calvello et al., Figure 12](https://www.jmlr.org/papers/v26/24-0879.html), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

Random inputs \(a\sim\rho\) induce a solution law \(\mathcal S_\#\rho\). When observations leave several fields possible, we may instead want to learn a conditional law directly. This brings us to functional transport learning.

### Functional transport learning (FFM): generating solution distributions {#functional-flow-matching}

Suppose we have samples of functions: solution fields from a simulator, for example. We want to generate new functions from the same distribution. FFM starts with a reference distribution of random functions and learns a velocity that transports it toward the data distribution.

Figure 11 illustrates one conditional path. Each blue curve is a complete function. As generative time advances, the curve moves from noise toward a target function; the arrows show the required functional velocity. We will construct such paths first, then see how their velocities give a training objective.

{{< figure src="figures/ffm-function-flow.png" link="figures/ffm-function-flow.png" alt="Four stages of a noisy function evolving toward a sine curve, with arrows showing its function-space velocity." caption="**Figure 11.** A conditional function-space path from a noisy function toward a target curve. Arrows indicate its functional velocity. Source: [Kerrigan et al., Figure 1](https://arxiv.org/abs/2305.17209v2)." >}}

#### Constructing a path between noise and a function

Draw a target function \(f\sim\nu\) and an independent Gaussian random function \(\xi\sim\mathcal N(0,C)\). Choose \(0 < \sigma_{\min} < 1\). One construction in [Kerrigan et al., Section 4](https://proceedings.mlr.press/v238/kerrigan24a.html) is

$$
\begin{aligned}
\sigma_\tau&=1-(1-\sigma_{\min})\tau,\\
u_\tau&=\tau f+\sigma_\tau\xi,\\
w_\tau&=f-(1-\sigma_{\min})\xi.
\end{aligned}
$$

At \(\tau=0\), the sample is pure reference noise. At \(\tau=1\), it is the target function plus residual noise, so the endpoint law is smoothed. Differentiating \(u_\tau\) gives \(w_\tau\); the training velocity is known without solving an ODE.

#### Learning the velocity

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

This averages the velocities of the conditional paths arriving at \(u\). Conditional expectation makes the cross term vanish:

$$
\mathbb E\|v_\theta-w_\tau\|_H^2
=\mathbb E\|v_\theta-v^*\|_H^2
+\mathbb E\|w_\tau-v^*\|_H^2.
$$

The last term is independent of the model parameters. Fitting sampled path velocities therefore fits the marginal velocity as well.

After training, generate a function by drawing \(u_0\sim\mu_0\) and integrating

$$
\frac{du_\tau}{d\tau}=v_\theta(\tau,u_\tau).
$$

If \(\Phi_\tau\) denotes the resulting flow map, then the evolved distribution is \(\mu_\tau=(\Phi_\tau)_\#\mu_0\): draw from the initial law and apply the map. Training uses directly sampled paths; generation uses an ODE solver. The transport interpretation requires the flow and measure assumptions in [Kerrigan et al., Sections 3–4](https://proceedings.mlr.press/v238/kerrigan24a.html); the regression identity alone does not establish them.

#### Choosing a random function

The covariance \(C\) above is part of the model. To see its role, expand the random function in orthonormal modes \(e_j\), with \(Ce_j=\lambda_j e_j\):

$$
\xi=\sum_j\sqrt{\lambda_j}\epsilon_j e_j,
\qquad \epsilon_j\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
\qquad
\mathbb E\|\xi\|_H^2=\sum_j\lambda_j.
$$

A positive trace-class covariance, \(\sum_j\lambda_j<\infty\), gives an \(H\)-valued Gaussian. In \(L^2\), the eigenvalues allocate expected field energy among spatial modes. The choice \(\lambda_j=1\) for every mode has infinite total energy, so an identity-covariance Gaussian does not define a random element of infinite-dimensional \(H\). White noise can be defined in other ambient spaces, including generalized-function settings.

This makes source design a physical question: which modes should carry uncertainty, and how should their amplitudes decay as we increase resolution?

#### Example: generating Navier–Stokes fields

The FFM paper tests generation of \(64\times64\) Navier–Stokes fields using an FNO velocity model. Figure 12 compares independent data and generated samples, together with errors in density and spectral statistics. This is a distribution-learning experiment: the generated panels are new draws, rather than predictions paired with particular ground-truth fields.

{{< figure src="figures/ffm-navier-stokes-samples.png" link="figures/ffm-navier-stokes-samples.png" alt="Independent Navier–Stokes field samples from the dataset, FFM, DDPM, DDO, and GANO, with a table comparing density and spectrum errors." caption="**Figure 12.** Samples and distributional diagnostics in the FFM Navier–Stokes experiment. Matching a collection of field statistics differs from solving an individual initial-value problem. Source: [Kerrigan et al., Figure 2 and Table 2a](https://proceedings.mlr.press/v238/kerrigan24a.html)." >}}

The generative ODE advances \(\tau\), not the fluid's physical time. The experiment supports an empirical comparison of field distributions; it does not establish that each generated field satisfies a specified PDE trajectory.

#### Example: conditioning a function law on observations

The same transport idea can use side information \(c\). Draw training functions from the conditional data law and fit

$$
\mathcal J_{\rm cond}(\theta)
=\mathbb E\|v_\theta(\tau,u_\tau;c)-w_\tau\|_H^2.
$$

The population target becomes \(\mathbb E[w_\tau\mid\tau,u_\tau,c]\). Figure 13 illustrates this on the paper's AEMET temperature curves. Several generated curves share the conditioning information while varying elsewhere. This is a time-series example, separate from the Navier–Stokes experiment.

{{< figure src="figures/ffm-conditional-functions.png" link="figures/ffm-conditional-functions.png" width="720" alt="Conditional FFM temperature curves: dark generated curves, pale data curves, and black observations, comparing conditional training with additional conditional sampling." caption="**Figure 13.** Conditional function generation on AEMET. The left column uses conditional training; the right additionally modifies sampling to enforce the observations. Source: [Kerrigan et al., Figure 5](https://proceedings.mlr.press/v238/kerrigan24a.html)." >}}

Conditioning the learned velocity and enforcing observations during sampling are different mechanisms. The right column demonstrates the latter's effect; exact agreement at observed points does not by itself establish a correct posterior law ([Section 5 and Appendix A.4](https://proceedings.mlr.press/v238/kerrigan24a/kerrigan24a.pdf)). The paper's FNO implementation also uses uniform grids, so its continuum formulation should be distinguished from that numerical restriction.

| | FGD: functional gradient descent | FOL: the Picard construction | FFM: functional flow matching |
| --- | --- | --- | --- |
| Object | One function | A map between functions | A law over functions |
| Goal | Decrease a functional | Approximate a solution map uniformly over inputs | Transport a probability law |
| Update mechanism | Negative Riesz gradient | Repeated approximate fixed-point blocks | Learned generative velocity |
| Essential structure | Geometry and gradient accuracy | PDE well-posedness, contraction, kernel and nonlinear approximation | Measures, covariance, conditional paths |
| Iteration meaning | Optimization time | Refinement of a whole candidate trajectory | Generative time |

FFM's velocity need not be a gradient field. All three constructions face questions about representation, approximation, and stability.

## 4. Designing a functional PDE model

With the three learning targets in place, we can choose the representation and computations that realize them. The common requirements concern physical operations, available updates, and numerical accuracy.

### The shared building elements

Let \(a\) collect the coefficients, forcing, domain, and initial or boundary conditions. Before choosing a neural architecture, specify six elements.

| Element | Mathematical form | What it determines |
| --- | --- | --- |
| Function and context | \(u\in H,\ a\in\mathcal A\) | Which physical state is represented, and for which problem |
| Operations | \(\mathcal O_hu,\ \mathcal A_a(u),\ Q(u)\) | How we observe the field, apply the governing model, and extract quantities |
| Geometry and regularity | \(\langle\cdot,\cdot\rangle_H\), domains of operators | Which errors and changes are small; which operations are defined |
| Target | \(u^\star(a),\ \mathcal S:a\mapsto u,\ \mu(du\mid a)\) | One solution, a solution map, or a conditional law |
| Construction or evolution | \(\widehat{\mathcal S}:a\mapsto u\), \(u^{(k+1)}=\Phi_a(u^{(k)})\), or \(\partial_r u_r=V_r(u_r;a)\) | How to obtain or approach that target |
| Numerical realization | \(z_m\in\mathbb R^{d_m},\ u\approx D_m(z_m)\), quadrature, integration | How finite coordinates realize the functional problem; adaptation can grow \(d_m\) |

Here \(\mathcal O_h\) is an observation operator, \(\mathcal A_a\) is a physical or measurement model, and \(Q\) is a quantity such as drag or heat flux. The choice of \(H\) must support the required operations, possibly with additional regularity. For a probabilistic target, we must also specify a reference law and its covariance; the inner product alone does not determine a random function.

An operator can predict directly or iterate as in Figure 8. For continuous evolution, \(r\) denotes optimization time \(s\) in FGD, generative time \(\tau\) in FFM, or physical time \(t\) in an evolution PDE. Figure 14 separates these motions from their numerical coordinates.

{{< figure src="figures/functional-learning-elements.svg" link="figures/functional-learning-elements.svg" width="720" alt="A functional learning problem specifies a field, operations, geometry, and target. Optimization, physical evolution, or generative transport supplies the motion. A decoder realizes it using a finite code that can grow under refinement while a transfer preserves the field." caption="**Figure 14.** Different tasks supply different motions of a function. The finite code z contains the chosen representation's values or coefficients; its dimension can grow. Exact preservation under refinement is a design condition, satisfied by the copying operations in the three default FGD representations. Original LaTeX/TikZ schematic." >}}

### Geometry and available updates {#geometry-induced-by-the-representation}

The PINN Jacobian calculation extends to any differentiable decoder. Let \(u=D_m(z)\). Its derivative maps a small code change to a field change, \(D_m'(z)\delta z\). The decoder therefore determines which update directions are available and how large they are in the function norm.

To approximate a desired field velocity \(V(u)\), minimize the mismatch:

$$
\dot z=\arg\min_b\|D_m'(z)b-V(D_m(z))\|_H^2.
$$

The normal equation is

$$
G(z)\dot z=D_m'(z)^*V(D_m(z)),
\qquad G(z)=D_m'(z)^*D_m'(z).
$$

When \(D_m'\) has full column rank, \(G\) is positive definite and decoding this update gives the orthogonal projection of \(V\) onto the available tangent directions. Redundant coordinates require a pseudoinverse or a nonredundant chart. For a linear synthesis \(D_m(z)=\sum_jz_j\psi_j\), \(G_{ij}=\langle\psi_i,\psi_j\rangle_H\) is the familiar Gram matrix.

The same construction appears in [manifold Galerkin reduced models](https://arxiv.org/abs/1812.08373). A decoder can reconstruct endpoint fields well while giving poor tangent directions between them. Reconstruction accuracy and update accuracy consequently measure different aspects of the representation.

For a physical evolution \(\partial_tu=\mathcal F_a(u)\), choose \(V=\mathcal F_a\). [Neural Galerkin](https://arxiv.org/abs/2203.01360) realizes this principle with a neural function ansatz and a Gram-matrix system. This follows physical time; FGD's wave example instead optimizes an entire space-time solution.


### When the finite code grows

The FGD examples make representation adaptation concrete. While a representation is fixed, changing \(z_m\) changes the function within \(D_m(Z_m)\). Refinement changes the available coordinate space itself:

$$
z_m\in\mathbb R^{d_m}
\quad\xrightarrow{\ P_{m\to m'}\ }\quad
z_{m'}\in\mathbb R^{d_{m'}},
\qquad d_{m'}>d_m.
$$

The transfer \(P_{m\to m'}\) must account for the new decoder. In the nested tree, Fourier-bin, and default voxel examples, copying values gives

$$
\boxed{D_{m'}(P_{m\to m'}z_m)=D_m(z_m).}
$$

Thus \(z\) expands without changing the represented function at that instant. The larger vector now permits children to move independently. For a linear synthesis, a subsequent gradient update is \(z_{m'}^+=P_{m\to m'}z_m-\eta b_{m'}\), where \(D_{m'}(b_{m'})\) approximates the correction field.

Expansion is not always appending zeros: an orthogonal basis extension can use zeros, while a cell subdivision duplicates coefficients. For a learned decoder, neither rule guarantees preservation. Measure the transfer error \(\|D_{m'}(P_{m\to m'}z_m)-D_m(z_m)\|\) in the task's function norm. The partition, basis, and decoder give \(z\) its meaning; extra stored values help only when those choices add useful function directions.

For a generated ensemble, an exact field-preserving transfer also leaves the current field law unchanged. It does not supply the randomness or correlations that newly resolved directions should have under the intended target law. The Outlook returns to that requirement for adaptive transport.

### Function autoencoders and physical structure

FunDiff also uses physical structure in the representation itself. For a sufficiently smooth two-dimensional streamfunction \(\psi\), define

$$
\mathbf v=(\partial_y\psi,-\partial_x\psi),
\qquad\nabla\cdot\mathbf v=0,
$$

The two mixed derivatives cancel, so this velocity is divergence-free by construction. A residual penalty takes another route: it penalizes violations at the points and with the weights used in the loss.

Figure 15 shows the FunDiff pipeline in the preprint. A Vision Transformer processes the sampled inputs, and a Perceiver encoder handles variable discretizations and maps them into a common latent representation. The decoder uses cross-attention between query coordinates and encoded features to evaluate the function. Physical constraints enter this function autoencoder through its architecture or training loss.

The second stage trains a Diffusion Transformer using rectified flow on the learned latent representation. Generation integrates a latent ODE from Gaussian noise and decodes its output into a function. Thus physical-prior enforcement is carried by the autoencoder and is decoupled from the generative model's training and inference.

{{< figure src="figures/fundiff-framework.png" link="figures/fundiff-framework.png" alt="FunDiff architecture: an encoder and coordinate-query decoder with physical priors, followed by latent diffusion training and inference for several field-reconstruction tasks." caption="**Figure 15.** FunDiff combines a function encoder and coordinate decoder, physical priors, and latent generation. Complete original diagram. Source: [Wang et al., Figure 1, preprint v2](https://arxiv.org/abs/2506.07902v2), [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)." >}}

We must also decide when a constraint should hold. For a linear constraint \(Au=0\), a source in its kernel and a velocity satisfying \(Av=0\) preserve it along a sufficiently regular path. Requiring physical validity throughout generation imposes more conditions than requiring it only of final samples.

### Numerical observations and function-space losses

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

The encoder, loss, and solver all approximate operations on functions. Their numerical errors belong in the accuracy budget alongside the learned model. The next section connects these errors to physical outputs and solution laws.

## 5. What controls accuracy and reliability?

The functional formulation makes accuracy requirements explicit. Three estimates explain what must be controlled: physical quantities, representable fields, and the accumulation of update errors.

### Physical norms and PDE stability

For fields with one square-integrable spatial derivative, a possible physical norm is

$$
\|u\|_H^2=\|u\|_{L^2}^2+\ell^2\|\nabla u\|_{L^2}^2,
\qquad \ell>0.
$$

Use the same length scale and channel scaling at every resolution. This gives \(\|\nabla u-\nabla v\|_{L^2}\le\ell^{-1}\|u-v\|_H\). Point evaluation may require stronger regularity; shocks may call for weak formulations.

A small residual implies a small solution error only when the PDE is stable in the chosen norms. For example, if \(\mathcal A:H\to H^*\) is strongly monotone,

$$
\langle\mathcal A(u)-\mathcal A(v),u-v\rangle
\ge\alpha\|u-v\|_H^2,\qquad \alpha>0,
$$

and \(\mathcal A(u^\star)=b\), duality gives

$$
\|u-u^\star\|_H
\le\alpha^{-1}\|\mathcal A(u)-b\|_{H^*}.
$$

This connects residual reduction to physical solution error. Strong monotonicity is a sufficient example; nonunique or ill-conditioned inverse problems require other stability statements or a target law that retains their uncertainty.

### Representation error in fields and laws

A fixed representation can leave an accuracy floor. For an orthogonal projection \(P_m\) onto a finite space \(H_m\), and \(U\sim\mu\) with finite second moment, that floor is the unresolved tail
\(\delta_m^2=\mathbb E\|(I-P_m)U\|_H^2\). Increasing query density does not remove it.

Field reconstruction also controls decoded-law error:

$$
W_{2,H}\bigl(\operatorname{Law}(D(E(U))),\mu\bigr)
\le
\bigl(\mathbb E\|D(E(U))-U\|_H^2\bigr)^{1/2}.
$$

Here \(W_{2,H}\) is the minimum RMS field distance over couplings of the two laws. The displayed coupling pairs each field with its reconstruction. This controls approximation of the law in the chosen norm; it does not establish the velocity accuracy needed by an evolving model.

### Update errors accumulate through stability

Suppose a reference trajectory satisfies \(\dot u_\tau=v_\tau(u_\tau)\), with integrable state-Lipschitz envelope \(L(\tau)\). Let a computed trajectory have defect \(r_\tau=\dot{\widetilde u}_\tau-v_\tau(\widetilde u_\tau)\), and jumps \(J_j\) at finitely many representation changes. Under the regularity needed for Grönwall's inequality,

$$
\begin{aligned}
\|\widetilde u_1-u_1\|_H
\le e^{\int_0^1L(\tau)d\tau}\Big[
&\|\widetilde u_0-u_0\|_H\\
&+\int_0^1\|r_\tau\|_H\,d\tau
+\sum_j\|J_j\|_H\Big].
\end{aligned}
$$

The defect can include velocity approximation and numerical integration errors. The jumps measure changes caused by transferring the field between representations. Exact copying in the FGD diagrams removes that transfer term; it does not remove the gradient or velocity defect.

For coupled random initial states, an RMS version gives a corresponding Wasserstein bound when the terms are square-integrable. This is why low training loss, accurate reconstruction, and accurate numerical transport are separate requirements. A useful adaptive rule must control their accumulated effect in the physical norm.


## 6. RKHS Flow: from observed velocities to field updates {#what-makes-an-rkhs-flow-functional}

A model may predict changes at a finite set of sensors while its intended output is a changing physical field. What should that update do between the sensors, and which field errors should training penalize? An RKHS gives a concrete answer: **choose a field geometry, lift the predicted readings into a function, and measure velocity errors in that same geometry.** This makes the represented field and its distribution explicit. We can then distinguish errors caused by missing observations from errors caused by learning or sampling the flow.

We develop that argument from a two-mode field example, then examine three choices that shape generation: the source distribution, the pairing of source and target samples, and their interpolation path. The [appendix](#rkhs-flow-proofs) supplies proofs and the numerical protocols. The examples explain these choices; they do not assume that a particular kernel improves every PDE forecast.

### What the network predicts

Suppose a field is represented through readings at a finite set of locations. During generation, a network predicts how those readings should change, conditioned on the current state, the noise level, and any available context, such as earlier fields. We can express its output as a velocity \(w_\theta\) in whitened coordinates: coordinates in which the chosen observation covariance becomes the identity. The RKHS lift then turns that finite vector into a spatial field update.

Let \(H\) be the chosen RKHS, \(\mathcal O_X:H\to\mathbb R^N\) the bounded observation operator, and \(G=\mathcal O_X\mathcal O_X^*\) its positive-definite Gram matrix. For one frame and channel, the prediction and its field interpretation are

$$
\boxed{
 v_\theta=G^{1/2}w_\theta,
 \qquad \widetilde v_\theta=\mathcal S_Xv_\theta,
 \qquad \mathcal S_X=\mathcal O_X^*G^{-1}.
}
$$

The network outputs observation-space velocities after coloring by \(G^{1/2}\). The fixed lift \(\mathcal S_X\) gives them a spatial meaning; the sampler need not materialize a dense field at every step. With representers \(r_i=\mathcal O_X^*e_i\), where \(e_i\) is a standard basis vector, the lifted velocity is \(\widetilde v_\theta(q)=\sum_i c_i r_i(q)\), with \(c=G^{-1}v_\theta\). These kernel coefficients follow from a linear solve and vary with the network's current state and conditioning.

For direct point observations, \(r_i(q)=K(x_i,q)\). For averaged or interpolated readings, the representers instead incorporate that observation rule. This distinction matters: the observation operator describes what is measured, while the kernel describes the geometry used to extend those measurements into a field. An arbitrary RKHS function need not have an exact expansion over these \(N\) representers; the finite expansion comes from selecting the observation span.

### What the lift makes optimal

The lift reproduces the readings and chooses their minimum-norm extension:

$$
\mathcal O_X\mathcal S_Xv=v,
\qquad
\mathcal S_Xv=\underset{f:\,\mathcal O_Xf=v}{\arg\min}\;\|f\|_H^2.
$$

This is a precise optimality statement. Among fields consistent with the same observation velocity, it selects the smallest correction in the chosen geometry. Components invisible to the observations are set to zero in that extension. Whether they vanish in the true field is a separate question. [Appendix A.1](#rkhs-lift-proof) proves the result and its induced-norm identity.

We can see the choice directly in a two-mode example. On \([0,1]\), write \(\phi_1(x)=\sin(2\pi x)\), \(\phi_3(x)=\sin(6\pi x)\), and \(v=a\phi_1+b\phi_3\). Choose

$$
K_\rho(x,q)=\phi_1(x)\phi_1(q)+\rho\phi_3(x)\phi_3(q),
\qquad \|v\|_{H_\rho}^2=a^2+\frac{b^2}{\rho},\quad \rho>0.
$$

A sensor reports \(v(1/4)=1\), so all we know is \(a-b=1\). The minimum-norm solution is

$$
\boxed{\widetilde v_\rho(x)=\frac{\phi_1(x)-\rho\phi_3(x)}{1+\rho}.}
$$

For \(\rho=0.1\), the coefficients are approximately \((0.909,-0.091)\); for \(\rho=1\), they are \((0.5,-0.5)\). Both fit the sensor perfectly. The smaller \(\rho\) makes the high-frequency mode more expensive and assigns less of the reading to it. Starting from \(f=0.3\phi_1\), the update \(f^+=f+0.5\widetilde v_\rho\) changes the sensor value from \(0.3\) to \(0.8\) under either kernel, while moving the rest of the field differently.

{{< figure src="figures/rkhs-lift-two-mode-toy.png" link="figures/rkhs-lift-two-mode-toy.png" alt="Four panels show two kernels lifting the same observed velocity into different functions, the resulting field updates, an unobserved function direction, and recovery of both coefficients after adding a second sensor." caption="**Figure 16.** An exact two-mode RKHS lift. A: both kernels fit the black observation but extend it differently. B: those velocities update an entire field. C: adding a function that vanishes at the sensor preserves its reading. D: a second independent observation determines both coefficients in this particular two-mode space. The dashed reference in A is chosen for illustration and is not available to the one-sensor lift. Original computation; [script](materials/toy-rkhs-lift/run.py) and [checks](materials/toy-rkhs-lift/results.json)." >}}

The function \(h=\phi_1+\phi_3\) vanishes at the sensor. Adding any multiple of it preserves the reading; the lift chooses the smallest-norm member of this family. If the reference velocity is \(\phi_1\), a second reading \(v(1/12)=1/2\) identifies both coefficients, and either kernel recovers the reference. This is exact because the example assumes only two modes and now has two independent measurements. [Appendix A.2](#rkhs-spectral-proof) works through the calculation. The toy explains the lift without any neural training or source sampling.

### How spectral weights shape the field and the loss

For a periodic field, one way to define the geometry is to assign positive weights to a finite collection of Fourier modes. For example, take

$$
c_k\propto\left(1+\frac{k_x^2+k_y^2}{16}\right)^{-3},
\qquad C=\operatorname{diag}(c_k),\qquad G=ACA^T,
$$

where \(A\) maps the unscaled Fourier coefficients to observations. The constants in this example set the scale and rate of spectral decay; other choices express different smoothness preferences. Higher frequencies receive smaller weights. Their effects depend on where those weights enter:

| Operation | Effect of the spectral geometry |
| --- | --- |
| Source \(Z\sim\mathcal N(0,G)\) | Allocates less underlying Fourier variance to high frequencies. |
| Field lift | Chooses coefficients minimizing \(\sum_k b_k^2/c_k\) subject to the predicted readings. |
| Velocity loss | Penalizes errors along low-variance observation directions more strongly. |

The lift discourages **unsupported high-frequency content**. The loss asks for **accurate low-variance components when they are present in the target**. The corresponding velocity-error penalty is

$$
\|\mathcal S_X(v_\theta-V)\|_H^2
=(v_\theta-V)^TG^{-1}(v_\theta-V)
=\|w_\theta-G^{-1/2}V\|_2^2.
$$

It penalizes the residual, rather than the prediction's amplitude. Removing a small-scale feature that belongs in the target can therefore be costly. On the actual observation geometry, the weighted directions are eigenvectors of \(G\), which need not be individual Fourier modes. Whitening and coloring express this same objective in matched coordinates; they are not an additional independent regularizer or a hard frequency cutoff.

For a fixed path and geometry, the unrestricted regression optimum is still

$$
v^*(U)=\mathbb E[V\mid U],
$$

with noise level and conditioning held fixed. Ordinary squared error has the same optimum. The RKHS metric changes how a finite model prioritizes approximation error. Changing the source covariance changes the path itself. [Appendix A.3](#rkhs-regression-proof) separates those two effects and the irreducible variability in the training targets.

### From observation distributions to field distributions {#rkhs-distribution-geometry}

The lift preserves distances exactly when observation space uses the matching metric:

$$
\|\mathcal S_Xu-\mathcal S_Xv\|_H^2=(u-v)^TG^{-1}(u-v).
$$

Its image is the observation representer span \(H_X=\operatorname{ran}\mathcal O_X^*\). On this span, the lift is an isometry: it changes the description of an object without changing its distance to another object. The same correspondence holds for distributions. If \(\mu,\nu\) are observation laws with finite second moments, then

$$
\boxed{W_{2,H}(\mathcal S_{X\#}\mu,\mathcal S_{X\#}\nu)
=W_{2,G^{-1}}(\mu,\nu).}
$$

Here \(\mathcal S_{X\#}\mu\) means the law obtained by lifting a random vector with law \(\mu\); \(W_2\) is the quadratic Wasserstein distance in the indicated metric. Thus matching observation laws in this metric exactly matches their reconstructed field laws. It need not recover the original field law.

To see what remains missing, let \(F\sim\rho\) be a field with finite second moment in \(H\), and write \(P_X=\mathcal S_X\mathcal O_X\). This is the orthogonal projection onto \(H_X\). Every law supported on that span faces the same best possible error:

$$
\boxed{\inf_{\nu:\,\nu(H_X)=1}W_{2,H}^2(\rho,\nu)
=\underbrace{\mathbb E\|(I-P_X)F\|_H^2}_{\eta_X^2}.}
$$

The projected law \(P_{X\#}\rho\) attains this bound. In the two-mode toy, variability along the unobserved direction contributes to \(\eta_X\); perfect sensor predictions cannot recover it. Querying more locations evaluates the same span. Reducing this error requires more informative observations, a richer representation, or a model for the conditional variability in the missing directions. [Appendix A.6](#rkhs-law-proof) proves both identities.

Learning and numerical integration add separate errors. Under the flow regularity assumptions in [Appendix A.7](#rkhs-generation-bound), on a unit generation-time interval,

$$
W_{2,H}(\widehat\rho,\rho)
\le \eta_X+e^L\sqrt{\mathcal E_{\rm vel}}+\varepsilon_{\rm num}.
$$

The terms measure representation error, excess velocity-regression risk amplified by flow sensitivity \(L\), and numerical sampling error. The raw training loss includes irreducible target variance and cannot replace \(\mathcal E_{\rm vel}\). This bound explains why improving the learned velocity cannot repair a missing field direction, and why a good fit still needs a stable sampler.

### When a suitable source simplifies transport

Consider a cosine interpolation between a clean field and Gaussian noise,

$$
U=aY+sZ,\qquad V=aZ-sY,
\qquad a=\cos\vartheta,\quad s=\sin\vartheta.
$$

Here \(Y\) is the clean target and \(Z\) the independent Gaussian source. The target \(V\) is the derivative with respect to noise angle \(\vartheta\). At \(\vartheta=0\), the state is the target; at \(\vartheta=\pi/2\), it is the source. Generation follows the learned flow toward decreasing angles. If we parameterize generation by a time \(\tau\), the drift includes the factor \(d\vartheta/d\tau\).

For a centered Gaussian mode with target variance \(\lambda\) and source variance \(g\), [Gaussian conditioning](https://gaussianprocess.org/gpml/chapters/RW2.pdf) gives

$$
\boxed{v^*(u,\vartheta)
=\frac{as(g-\lambda)}{a^2\lambda+s^2g}\,u.}
$$

When \(g=\lambda\), the marginal velocity is zero: every intermediate distribution already equals the target. Individual paired paths still move, but their velocity targets cancel in the conditional mean. Matching variances can therefore remove part of the transport's work. [Appendix A.4](#rkhs-gaussian-proof) derives the formula. A smoothness prior, however, need not match the target covariance. In a forecasting problem, that covariance can also depend on the observed history, so a fixed source may simplify some conditional transports more than others.

A stretched two-dimensional spiral makes the remaining work visible. We place 96 small Gaussian components along a curve, obtaining a known target covariance \(\Sigma\) and an analytic conditional velocity. Compare sources \(\mathcal N(0,\Sigma)\) and \(\mathcal N(0,\tfrac12\operatorname{tr}(\Sigma)I)\), with equal total variance and the same cosine schedule. The matched source already has the target's elongated spread. The flow still has to form the spiral's curved structure.

{{< figure src="figures/rkhs-flow-spiral-toy.png" link="figures/rkhs-flow-spiral-toy.png" alt="Two rows compare isotropic and covariance-matched Gaussian sources as they flow into a stretched spiral. Columns show source samples, the halfway state, generated endpoints, and independent target samples." caption="**Figure 17.** A two-dimensional illustration of source geometry. Each row follows 700 samples through the analytic mixture velocity using 120 RK4 steps. Axes and total source variance are matched. This is a finite-dimensional transport toy, with known target covariance and no neural training; it illustrates source design separately from the spatial RKHS lift. Original computation; [reproduction script](materials/toy-rkhs-flow/run.py) and [results](materials/toy-rkhs-flow/results.json)." >}}

This toy isolates source design; it contains no spatial field lift. To measure its effect in common coordinates, integrate the squared marginal drift along the prescribed probability path:

$$
\mathcal A(G)=\int_0^{\pi/2}
\mathbb E_{U\sim p_\vartheta}\|v^*(U,\vartheta)\|_2^2\,d\vartheta.
$$

For the spiral, the estimates are \(1.116\) for the isotropic source and \(0.438\) for the matched source. For a Gaussian target with the same covariance, they are \(0.806\) and exactly zero. Both comparisons use the same Euclidean norm and angle interval. Averaging squared paired training targets would measure a different quantity. [Appendix A.5](#rkhs-toy-protocol) gives the mixture velocity and numerical protocol.

{{< figure src="figures/rkhs-flow-toy-drift.png" link="figures/rkhs-flow-toy-drift.png" alt="Integrated squared angular drift for Gaussian and spiral targets. Covariance matching gives zero for the Gaussian target and reduces the estimate from 1.116 to 0.438 for the spiral target." caption="**Figure 18.** Covariance matching reduces this measure of motion in the constructed examples. Estimates use 4,096 independent source-target pairs and 129 angle values; error bars are plus or minus 1.96 Monte Carlo standard errors. Increasing quadrature from 65 to 129 angles changes each estimate by less than 0.00001. The quantity is schedule-dependent and does not measure training time, solver cost, or optimality over all transport paths. Original computation." >}}

### Choosing which source and target functions to pair {#rkhs-endpoint-coupling}

Source design changes where transport starts. Endpoint coupling changes which source sample is paired with which target during training. [Xu et al. (2026)](https://arxiv.org/abs/2609.38049) use a kernel on **whole functions** to compute an entropic optimal-transport plan before constructing the training paths. Their kFFM kernel supplies a pairing cost; our spatial kernel supplies the field lift. These are different roles that can be combined.

At a fixed observation geometry, a natural pairing cost is already available:

$$
d_X^2(u,v)=\|\mathcal S_Xu-\mathcal S_Xv\|_H^2
=(u-v)^TG^{-1}(u-v).
$$

Alternatively, a Gaussian kernel on the reconstructed fields gives the bounded cost

$$
\kappa_X(u,v)=\exp\!\left[-\frac{d_X^2(u,v)}{2\sigma^2}\right],
\qquad c_\kappa(u,v)=2[1-\kappa_X(u,v)].
$$

The quadratic cost measures squared field distance exactly on \(H_X\). The bounded cost compresses large distances, so minimizing it solves a different transport problem. Neither cost sees components outside the represented span. The paper's theorems establish properties of its coupling objective and its discretization; improvements in generated samples are empirical results. Its turbulent-flow comparison also shows why source choice must be controlled: kernel and raw-\(L^2\) coupling give overlapping results with the same white-noise source (Section 4.2, Table 4).

A coupling preserves both endpoint distributions. In forecasting, it must preserve them **conditional on the history**: global matching across unrelated histories can change the source law seen for each history. Sampling endpoint pairs from a plan preserves its marginals in law; replacing endpoints with their barycentric averages generally does not. These choices matter before any neural training begins.

### Why pairing and interpolation must be chosen together {#rkhs-coupling-schedule}

Closer endpoint pairs need not reduce motion under every interpolation. For the controls below, \(t\in[0,1]\) denotes normalized generation time, with physical time held fixed. Consider the simplest case: source and target are both standard Gaussian. Independent samples \(Y,Z\) give a stationary marginal law under the cosine path above. Pairing each sample with itself, \(Z=Y\), instead gives

$$
U_t=\left[\sin\!\left(\frac{\pi t}{2}\right)+\cos\!\left(\frac{\pi t}{2}\right)\right]Y.
$$

The distribution expands and contracts, even though this identity coupling has zero endpoint distance. With affine interpolation, \(U_t=(1-t)Z+tY\), identity pairs remain stationary. Independent pairs instead contract and expand. The coupling and the schedule jointly determine the intermediate distributions.

{{< figure src="figures/rkhs-coupling-schedule-gaussian.png" link="figures/rkhs-coupling-schedule-gaussian.png" alt="Standard-deviation curves and exact drift actions for identical Gaussian endpoint laws: independent pairing is stationary under cosine interpolation, while identity pairing is stationary under affine interpolation." caption="**Figure 19.** Identical endpoint laws can still produce moving intermediate laws. All four cases use generation time from zero to one; the cosine schedule is parameterized in the source-to-target direction. The plotted action uses the conditional-mean velocity, not individual pair velocities. The identity coupling is the zero-cost quadratic-OT limit, not a finite-entropy plan. Original analytic calculation; [script](materials/toy-rkhs-coupling/run.py), [results](materials/toy-rkhs-coupling/results.json), and [derivation](#rkhs-coupling-proof)." >}}

We also test a non-Gaussian spiral target while holding the endpoint laws fixed. Both endpoints are smooth mixtures with 24 Gaussian components. The source centers approximate a Gaussian cloud and have the same covariance as the spiral centers; this source is a fixed mixture, not the exact Gaussian used in Figure 17. We compare independent component labels with quadratic and bounded-kernel entropic transport between labels. Gaussian fluctuations within each paired component remain independent. The resulting joint mixtures have analytic marginal velocities.

{{< figure src="figures/rkhs-coupling-schedule-spiral.png" link="figures/rkhs-coupling-schedule-spiral.png" alt="Squared marginal drift for a fixed smooth spiral target, comparing independent, quadratic-OT, and bounded-kernel-OT component pairing under cosine and affine interpolation." caption="**Figure 20.** Coupling and schedule controls for fixed mixture endpoints. The metric and unit time interval are shared across all six cases. Error bars are plus or minus 1.96 Monte Carlo standard errors from 2,048 pairs; the integral uses 129 time points. OT acts on component labels, not on all possible couplings of the continuous endpoint laws. No neural model is trained. See the [protocol](https://github.com/chenzituo/chenzituo.github.io/blob/main/content/posts/functional-learning/materials/toy-rkhs-coupling/README.md) and [Appendix A.8](#rkhs-coupling-proof)." >}}

For this spiral, quadratic OT lowers the estimated action from \(2.921\) to \(0.335\) under affine interpolation, but raises it from \(1.312\) to \(3.904\) under the cosine schedule. Bounded-kernel OT gives similar estimates, \(0.337\) and \(3.906\), respectively. The endpoint laws have not changed. These controls measure motion along specified marginal paths; they do not measure velocity-learning difficulty or the number of solver steps required. Together with the Gaussian example, they show why a coupling should be assessed with its intended interpolation rather than by endpoint distance alone.

### What the functional interpretation provides

At fixed geometry, linearity gives \(\Delta f=\mathcal S_X\Delta u\). Observation updates therefore define field updates, a field-error norm, and a corresponding geometry of represented distributions. The reconstruction floor identifies what the observations leave unresolved. Source, coupling, and schedule then determine the transport problem within the chosen representation.

A field-valued update can describe different dynamics. Here it transports a probability distribution; the minimum-norm lift alone does not make it the negative functional gradient of a field objective. [Adaptive FGD](https://arxiv.org/html/2606.16926v1#S3) additionally requires a gradient relationship and control of approximation error under its stated assumptions. Training network weights with an optimizer does not establish that relationship for the field updates produced during sampling.

Refining the representation needs a further argument. It requires approximation estimates for the actual observations and compatible source laws across resolutions. In particular, for \(Z_X\sim\mathcal N(0,G_X)\), the expected squared norm of its lift is \(\mathbb E\|\mathcal S_XZ_X\|_H^2=N\). This finite-dimensional source is valid, but its RKHS second moment grows with observation dimension. An infinite-dimensional Gaussian field generally needs a weaker ambient space than its covariance kernel's RKHS. [Appendix A.9](#rkhs-refinement-scope) explains this distinction and the assumptions behind a discretization claim.

The practical motivation is now precise: RKHS geometry lets us say what field the model evolves, what errors its coordinates measure, and which uncertainties remain outside its representation. Better forecasts require those choices to suit the data and the conditional task. A matched experiment should vary source, pairing, or loss geometry separately, and evaluate new-observation errors and generated field statistics alongside training loss.

## 7. Outlook {#6-outlook}

Two directions follow from treating PDE solutions as functions: improve corrections to individual solutions, and generate their uncertainty more faithfully. The first two topics concern solvers; the last two extend the representation and source questions developed in the RKHS Flow example. Here \(H\) is a Hilbert space of fields whose norm measures relevant physical errors.

### Diagnosing stalled training {#diagnosing-blocked-functional-descent}

When a PINN stalls, a useful field correction may remain. For its field \(u_\theta\) and PDE loss \(\mathcal L\), let \(g=\nabla_H\mathcal L(u_\theta)\) be the functional gradient and \(J_\theta=\partial_\theta u_\theta\) the Jacobian mapping weight changes to field changes. In continuous training time, Euclidean weight descent induces

$$
\dot u_\theta=-J_\theta J_\theta^*g.
$$

Two causes deserve investigation: unreachable corrections and poorly scaled reachable ones. [PINN neural tangent kernel analysis](https://arxiv.org/abs/2007.14527) studies related training-rate imbalances.

For nonzero \(g\), a proposed diagnostic is

$$
\rho_\theta=\frac{\|(I-\Pi_\theta)g\|_H}{\|g\|_H}.
$$

Here \(\Pi_\theta g\) is the closest reachable correction, computed by least-squares fitting the Jacobian's field directions to \(g\). Large \(\rho_\theta\) indicates missing directions. With small \(\rho_\theta\), inspect the Gram matrix \(J_\theta^*J_\theta\) for poor scaling, alongside curvature and optimizer effects.

A Poisson test could compare ordinary descent, Gram-preconditioned descent with the same network, and adaptive functional descent. Match solution accuracy and record the diagnostic, conditioning, and total cost. Does correcting the geometry suffice, or are new directions needed?

### Learning reliable solution corrections {#learned-corrections-with-solver-guarantees}

We could learn both an initial field and subsequent corrections. Let \(a\) collect PDE coefficients, forcing, and boundary or initial conditions. A proposed solver is

$$
u^{(0)}=\mathcal S_\theta(a),
\qquad
\frac{du_s}{ds}=V_\theta(u_s;a).
$$

Here \(\mathcal S_\theta\) predicts the initial field and \(s\) is solver iteration time. The correction model \(V_\theta\) would train on intermediate fields and useful update directions, such as a functional gradient or a preconditioned PDE residual.

Reliability requires a justified progress criterion. An approximate gradient needs sufficient accuracy and suitable steps to decrease the objective. A small residual implies a small solution error only when the PDE provides a corresponding stability estimate.

Compare prediction with and without corrections at matched accuracy, including correction cost. Vary coefficients and meshes to test transfer.

### Adapting representations during generation {#adaptive-representations-that-preserve-evolving-laws}

Could capacity follow generative time \(\tau\) and the current sample? In a latent flow such as [FunDiff](https://arxiv.org/html/2506.07902v2), \(z\) is the finite code, \(D_m\) its decoder, and \(m\) its capacity. The model \(g_{\theta,m}\) predicts the code velocity:

$$
\frac{dz}{d\tau}=g_{\theta,m}(\tau,z),
\qquad
u=D_m(z).
$$

For fixed \(D_m\), its Jacobian \(D_m'(z)\) gives field velocity \(D_m'(z)g_{\theta,m}\) by the chain rule. More decoder queries refine evaluation; enlarging the code requires a compatible decoder and velocity model.

The refinement signal remains open. Different sampled paths can give different target velocities at the same state. A minibatch loss therefore mixes learning error with this variability. One candidate during generation is to compare decoded velocities from compatible coarse and fine flows at the same field and time. This consistency test needs control of reference, transfer, and integration errors before it supplies an error bound.

New coordinates need the intended conditional law. For orthonormal modes \(e_1,e_2\), a coarse code \(X\) might need extension to \(U=Xe_1+Ye_2\) with

$$
Y\mid X\sim\mathcal N(\alpha X,\sigma^2).
$$

Appending zero misses variability; independent noise misses the coupling to \(X\). Refinement must respect the conditional law at that time.

[Adaptive Flow Matching](https://proceedings.mlr.press/v267/fotiadis25a.html) and [Scale-Adaptive Generative Flows](https://arxiv.org/abs/2509.02971v2) adapt sources or schedules. A representation hierarchy adds coordinate transfers and laws for newly resolved modes.

Compare fixed fine, time-only, and state-dependent representations on Gaussian laws with known velocities. Match generated-law accuracy and record derivative statistics, correlations, and total cost.

### Conditioning the source and target laws {#sources-conditioned-on-physics-and-observations}

The source is the starting law over random functions. Its covariance allocates uncertainty among field modes. For a PDE inverse problem, boundaries, coefficients, and observations can change both the uncertain scales and their correlations.

We could condition the source on this information and learn a transport to a specified conditional solution law. Does a better-aligned source make transport easier?

Another possibility is physical guidance:

$$
v_\tau^{\mathrm{guided}}(u)
=v_\tau(u)-\lambda_\tau\nabla_H\mathcal L_{\mathrm{phys}}(u).
$$

Here \(v_\tau\) is the generative velocity, \(\mathcal L_{\mathrm{phys}}\) measures physical mismatch, and \(\lambda_\tau\) sets the correction strength. The correction changes the transport and generally its endpoint law. A lower physical loss therefore does not establish correct uncertainty.

A linear-Gaussian inverse problem with a known posterior provides a first test. Compare sources and velocities using physical residuals, posterior moments, uncertainty calibration, and sampling cost.

A functional formulation specifies which updates, errors, and laws should agree across discretizations. Its value for PDE learning becomes concrete when that agreement improves physical predictions and yields a reliable solver or a trustworthy ensemble at an affordable cost.

## Appendix A. RKHS Flow derivations {#rkhs-flow-proofs}

These arguments establish the lift, regression, and distributional properties used in Section 6. They are consequences of Hilbert-space projection, conditional expectation, and flow stability; the coupling controls are original illustrative calculations. They concern a fixed observation geometry unless refinement is explicitly discussed. They do not assume that a trained network attains the population regression optimum. The implementation and toys use finite-dimensional field spaces; no infinite-rank Gaussian with identity covariance on an RKHS is needed.

### A.1. Minimum-norm lift and induced metric {#rkhs-lift-proof}

Let \(H\) be a real RKHS and \(\mathcal O:H\to\mathbb R^N\) a bounded linear observation map. Assume \(G=\mathcal O\mathcal O^*\succ0\), and define \(\mathcal S=\mathcal O^*G^{-1}\). Then \(\mathcal O\mathcal S=I\). Moreover, for any observation vector \(v\),

$$
\begin{aligned}
\|\mathcal Sv\|_H^2
&=\langle\mathcal O^*G^{-1}v,\mathcal O^*G^{-1}v\rangle_H\\
&=(G^{-1}v)^TG(G^{-1}v)=v^TG^{-1}v.
\end{aligned}
$$

Every compatible field has the form \(f=\mathcal Sv+w\), with \(\mathcal Ow=0\). Since

$$
\langle\mathcal Sv,w\rangle_H
=\langle G^{-1}v,\mathcal Ow\rangle_{\mathbb R^N}=0,
$$

Pythagoras gives \(\|f\|_H^2=v^TG^{-1}v+\|w\|_H^2\). The unique minimizer is therefore \(\mathcal Sv\). This also shows that \(\mathcal S\mathcal O\) is the orthogonal projection onto the observation representer span. For multiple channels, apply the construction independently and sum the squared norms.

For any state increment, linearity yields \(\mathcal S(u+\Delta u)-\mathcal Su=\mathcal S\Delta u\). For a differentiable trajectory and fixed \(\mathcal S\), it yields \(d(\mathcal Su)/d\tau=\mathcal S\dot u\). If the observation geometry changes, this identity alone does not describe the transfer between spans.

### A.2. Fourier weights and the two-mode lift {#rkhs-spectral-proof}

Take linearly independent real basis functions \(\phi_k\), positive weights \(c_k\), and the finite kernel \(K(x,q)=\sum_k c_k\phi_k(x)\phi_k(q)\). For \(f=\sum_k b_k\phi_k\), the induced RKHS norm is \(\|f\|_H^2=b^TC^{-1}b\), where \(C=\operatorname{diag}(c_k)\). This follows by using the orthonormal RKHS basis \(\sqrt{c_k}\phi_k\); it is not a claim that the \(\phi_k\) are orthonormal under spatial quadrature.

Let \(A_{ik}=\mathcal O_i\phi_k\), so observations satisfy \(Ab=v\). If \(ACA^T\succ0\), the Lagrangian \(\tfrac12b^TC^{-1}b-\lambda^T(Ab-v)\) gives

$$
b=CA^T\lambda,\qquad
\lambda=(ACA^T)^{-1}v,\qquad
b=CA^T(ACA^T)^{-1}v.
$$

Thus the Fourier coefficients \(b\) and the representer coefficients \(\lambda\) are different coordinates for the same lift. Smaller \(c_k\) increases the norm cost of a coefficient; it does not exclude that mode.

For the two-mode toy, \(C=\operatorname{diag}(1,\rho)\), \(A=(1,-1)\), and \(v=1\). Hence \(G=1+\rho\) and \((a,b)=(1,-\rho)/(1+\rho)\). The invisible direction has coefficients \((1,1)\); its inner product with the lift is

$$
\frac{1}{1+\rho}+\frac{-\rho}{\rho(1+\rho)}=0.
$$

Adding a second sensor at \(x=1/12\) gives

$$
A_2=\begin{pmatrix}1&-1\\1/2&1\end{pmatrix},
\qquad v_2=\begin{pmatrix}1\\1/2\end{pmatrix}.
$$

Since \(\det A_2=3/2\), these readings uniquely identify \((a,b)=(1,0)\). The result is independent of \(\rho>0\) because the feasible set now contains just one coefficient vector. It assumes exact readings and this particular two-dimensional function space.

### A.3. Regression optimum and probability transport {#rkhs-regression-proof}

Condition on geometry, context, and noise level. Let \(U,V\) have finite second moments, with fixed positive-definite \(G\), and let \(m(U)=\mathbb E[V\mid U]\). For any predictor \(v\) of finite risk, expand the square and use \(\mathbb E[V-m(U)\mid U]=0\):

$$
\mathbb E\|v(U)-V\|_{G^{-1}}^2
=\mathbb E\|v(U)-m(U)\|_{G^{-1}}^2
+\mathbb E\|V-m(U)\|_{G^{-1}}^2.
$$

Positive definiteness makes \(m\) the unique optimum up to sets of zero probability. Replacing \(G\) by \(I\) leaves this optimum unchanged for the same joint law of \((U,V)\). The second term is irreducible conditional variance; the raw loss therefore does not equal velocity-approximation error.

For the colored prediction \(v_\theta=G^{1/2}w_\theta\), the quadratic is exactly \(\|w_\theta-G^{-1/2}V\|_2^2\). This identity concerns output coordinates, not the conditioning of the full neural-network optimization problem. The implemented frame masks and averaging further determine how training examples contribute.

For completeness, let \(U_\vartheta=\cos\vartheta\,Y+\sin\vartheta\,Z\), so \(dU_\vartheta/d\vartheta=V_\vartheta\). For a smooth compactly supported test function \(\psi\), finite moments justify

$$
\frac{d}{d\vartheta}\mathbb E\psi(U_\vartheta)
=\mathbb E[\nabla\psi(U_\vartheta)\cdot V_\vartheta]
=\mathbb E[\nabla\psi(U_\vartheta)\cdot m_\vartheta(U_\vartheta)].
$$

This is the weak continuity equation for the prescribed marginal laws. A unique ODE flow realizing those laws requires additional velocity regularity. Under a generation schedule \(\vartheta(\tau)\), the drift is \(\dot\vartheta(\tau)m_{\vartheta(\tau)}\); decreasing angle supplies the generation direction. A finite DDIM discretization is a separate numerical approximation.

### A.4. Gaussian covariance matching {#rkhs-gaussian-proof}

Let \(Y\sim\mathcal N(0,\Sigma)\), \(Z\sim\mathcal N(0,G)\) be independent, with \(\Sigma,G\succ0\). For \(U=aY+sZ\), \(V=aZ-sY\), and \(a^2+s^2=1\),

$$
Q=\operatorname{Cov}(U)=a^2\Sigma+s^2G,
\qquad \operatorname{Cov}(V,U)=as(G-\Sigma).
$$

Joint Gaussian conditioning gives

$$
\mathbb E[V\mid U=u]=as(G-\Sigma)Q^{-1}u.
$$

No commutation assumption is needed for this matrix formula. In one dimension it becomes the expression in Section 6. When \(G=\Sigma\), \(Q=\Sigma\) for every angle and the marginal drift vanishes. The conditional-variance term in Appendix A.3 can nevertheless remain positive: sample-pair paths move even when the marginal law stays constant.

### A.5. Exact spiral velocity and numerical scope {#rkhs-toy-protocol}

The spiral target is an equal mixture of \(J=96\) Gaussians, \(Y\mid j\sim\mathcal N(\mu_j,\sigma^2I)\), with \(\sigma=0.12\). The centered means follow the stretched curve specified in the [script](materials/toy-rkhs-flow/run.py). Its covariance is \(\Sigma=J^{-1}\sum_j\mu_j\mu_j^T+\sigma^2I\). Within a component, define

$$
Q=a^2\sigma^2I+s^2G,
\qquad B=as(G-\sigma^2I)Q^{-1}.
$$

Gaussian conditioning gives \(\mathbb E[V\mid U=u,j]=-s\mu_j+B(u-a\mu_j)\). The posterior component probabilities are

$$
\pi_j(u)=\frac{\exp[-\tfrac12(u-a\mu_j)^TQ^{-1}(u-a\mu_j)]}
{\sum_{\ell=1}^J\exp[-\tfrac12(u-a\mu_\ell)^TQ^{-1}(u-a\mu_\ell)]},
$$

so the exact marginal velocity is

$$
v^*(u,\vartheta)=\sum_{j=1}^J\pi_j(u)\big[-s\mu_j+B(u-a\mu_j)\big].
$$

Positive component variance keeps the target nonsingular. We compare \(G=\Sigma\) with \(G=\operatorname{tr}(\Sigma)I/2\). The diagnostic \(\mathcal A(G)\) uses a common Euclidean metric and angle interval, 4,096 independent source-target pairs, and trapezoidal quadrature at 129 angles. Source choices share the target draws and standard-normal source draws. Standard errors are computed across per-pair integrals, retaining dependence across angles. The diagnostic is schedule-dependent and is not minimized over paths or couplings.

The [recorded checks](materials/toy-rkhs-flow/results.json) include the zero-drift Gaussian identity, a finite-difference continuity-equation residual below \(2.7\times10^{-7}\), and changes below \(10^{-5}\) when refining from 65 to 129 angles. For 100 common matched-source particles, the RMS endpoint change from 120 to 240 RK4 steps is below \(2.7\times10^{-7}\). This tests numerical sensitivity, not neural learning or a benchmark-level distributional guarantee. The separate [lift script](materials/toy-rkhs-lift/run.py) checks exact re-observation, norm identities, null-space orthogonality, update linearity, and two-sensor recovery.

### A.6. Distribution geometry and the reconstruction floor {#rkhs-law-proof}

Let \(H\) be a real separable RKHS, \(\mathcal O:H\to\mathbb R^N\) bounded, and \(G=\mathcal O\mathcal O^*\succ0\). Write \(M=G^{-1}\), \(\mathcal S=\mathcal O^*G^{-1}\), \(H_X=\operatorname{ran}\mathcal O^*\), and \(P=\mathcal S\mathcal O\). Appendix A.1 shows that \(P\) is the orthogonal projection onto \(H_X\), and that \(\mathcal S:(\mathbb R^N,\|\cdot\|_M)\to H_X\) is an onto isometry with inverse \(\mathcal O|_{H_X}\).

For observation laws \(\mu,\nu\) with finite second moments, every coupling \(\gamma\in\Pi(\mu,\nu)\) pushes forward to a coupling of the lifted laws, with the same cost:

$$
\int\|\mathcal Su-\mathcal Sv\|_H^2\,d\gamma
=\int\|u-v\|_M^2\,d\gamma.
$$

Conversely, applying \(\mathcal O\) to both fields pulls every coupling of the lifted laws back to an observation coupling. Taking infima and square roots proves the Wasserstein isometry. It holds on the represented space; it does not identify arbitrary full-field laws from their observations.

Now let \(F\sim\rho\in\mathcal P_2(H)\), and let \(Q\) have any law supported on \(H_X\). In every coupling of \(F,Q\), orthogonality gives

$$
\|F-Q\|_H^2=\|(I-P)F\|_H^2+\|PF-Q\|_H^2.
$$

Thus \(W_{2,H}^2(\rho,\operatorname{Law}(Q))\ge\mathbb E\|(I-P)F\|_H^2\). The coupling \((F,PF)\) attains equality for \(\operatorname{Law}(Q)=P_\#\rho\). This proves the reconstruction floor and \(W_{2,H}(\rho,P_\#\rho)=\eta_X\). The statement presumes \(H\)-valued data of finite second moment; if data live in a weaker ambient space, the error must be formulated in that space instead.

### A.7. A conditional generation-error bound {#rkhs-generation-bound}

Fix geometry and context. Use normalized generation time \(t\in[0,1]\) (the local notation in A.7–A.8), with observation path laws \(p_t\) starting from the chosen source and ending at \(p_1=\mathcal O_\#\rho\). Let \(m_t\) be the conditional-mean velocity for this path. Assume its ODE has a flow realizing \(p_t\), and that the learned velocity \(v_\theta\) also has a well-defined flow and is uniformly \(L\)-Lipschitz in \(\|\cdot\|_M\), where \(M=G^{-1}\) and \(L\ge0\). Assume all terms below are finite. Define

$$
\mathcal E_{\rm vel}=\int_0^1
\mathbb E_{U_t\sim p_t}\|v_\theta(U_t,t)-m_t(U_t)\|_M^2\,dt.
$$

The risk is evaluated along the exact path. It is the population excess risk from Appendix A.3, not a finite training-set estimate. Its time variable is generation time: if \(\vartheta=\vartheta(t)\), both angular velocities are multiplied by \(\dot\vartheta(t)\) before forming this risk.

Couple exact and learned trajectories \(U_t,\widehat U_t\) with the same initial source draw. Splitting their drift difference at \(U_t\) yields

$$
\frac{d}{dt}\|\widehat U_t-U_t\|_M
\le L\|\widehat U_t-U_t\|_M
+\|v_\theta(U_t,t)-m_t(U_t)\|_M,
$$

in the usual almost-everywhere or upper-derivative sense. Gronwall's inequality, Minkowski's inequality, and Cauchy–Schwarz give

$$
\left(\mathbb E\|\widehat U_1-U_1\|_M^2\right)^{1/2}
\le \int_0^1e^{L(1-t)}
\left(\mathbb E\|v_\theta(U_t,t)-m_t(U_t)\|_M^2\right)^{1/2}dt
\le e^L\sqrt{\mathcal E_{\rm vel}}.
$$

This coupling bounds the Wasserstein distance between the endpoint observation laws. By Appendix A.6 it also bounds the distance between their lifted laws. Let \(\widehat\rho\) denote the numerically generated, lifted law, and let \(\varepsilon_{\rm num}\) bound its \(W_{2,H}\) distance from the exact learned-ODE endpoint law. The triangle inequality then gives

$$
\boxed{W_{2,H}(\widehat\rho,\rho)
\le \eta_X+e^L\sqrt{\mathcal E_{\rm vel}}+\varepsilon_{\rm num}.}
$$

This is a sufficient-condition bound, not an estimate of the errors of a trained network. It assumes the source is initialized correctly, the target path reaches the desired observation law, and the representation is fixed. A source mismatch or a nonzero terminal smoothing bias needs an additional term. Establishing a bound for a particular DDIM discretization also requires a numerical analysis of that scheme.

### A.8. Coupling and interpolation controls {#rkhs-coupling-proof}

**Matched Gaussian endpoints.** Let \(Y,Z\) be jointly centered scalar Gaussians with unit variances and correlation \(r\). Write \(U_t=a(t)Y+b(t)Z\), \(V_t=\dot aY+\dot bZ\), and

$$
q_t=a^2+b^2+2rab,\qquad
h_t=a\dot a+b\dot b+r(\dot ab+a\dot b).
$$

Gaussian conditioning gives \(m_t(u)=h_tu/q_t\) and \(\mathbb E|m_t(U_t)|^2=h_t^2/q_t\). The cosine schedule in Figures 19–20 uses \(a=\sin(\pi t/2)\), \(b=\cos(\pi t/2)\); the affine schedule uses \(a=t\), \(b=1-t\). Both run from source to target over the same unit interval. Independent pairing has \(r=0\); identity pairing has \(r=1\), for which the same formulas follow directly from \(Z=Y\). Integration gives

| Schedule | Independent pairing | Identity pairing |
| --- | --- | --- |
| Cosine | \(0\) | \(\pi^2/4-\pi/2\approx0.897\) |
| Affine | \(2-\pi/2\approx0.429\) | \(0\) |

These are marginal-drift actions \(\int_0^1\mathbb E|m_t(U_t)|^2dt\). Identity coupling is an unregularized quadratic-OT optimum for identical marginals. It is not the positive-entropy plan used by kFFM. The example disproves a universal inference from smaller endpoint cost to smaller marginal action under cosine interpolation. It makes no universal ordering claim about learned models. The unit-time cosine actions include the schedule derivative; they therefore differ from the angle-time actions in Figures 17–18 by a factor \(\pi/2\).

**Smooth spiral endpoints.** The [reproduction script](materials/toy-rkhs-coupling/run.py) fixes 24 source centers \(z_i\) and 24 spiral centers \(y_j\). Both sets are centered and have equal empirical covariance. Add independent isotropic Gaussian fluctuations of standard deviation \(\sigma_0=0.30\) to each endpoint. A joint law is specified by a label plan \(\pi_{ij}\) with every row and column summing to \(1/24\):

$$
(Z,Y)\mid(i,j)\sim
\mathcal N(z_i,\sigma_0^2I)\otimes\mathcal N(y_j,\sigma_0^2I).
$$

We compare the product label plan with entropic label plans minimizing quadratic cost \(d_{ij}^2=\|z_i-y_j\|_2^2\) or bounded cost \(2[1-\exp(-d_{ij}^2/(2s^2))]\), where \(s^2\) is the median squared center distance. Each cost matrix is divided by its median, and Sinkhorn uses \(\epsilon=0.05\). The optimization is over component labels with independent within-component noises. It is not the unrestricted continuous OT problem or an implementation of the paper's complete training procedure. All three plans nevertheless define exactly the same source and target mixture laws.

For a schedule \(a,b\), each path component has mean \(c_{ij}=bz_i+ay_j\) and covariance \(qI=\sigma_0^2(a^2+b^2)I\). Its conditional-mean velocity is

$$
m_{ij,t}(u)=\dot bz_i+\dot ay_j+
\frac{a\dot a+b\dot b}{a^2+b^2}(u-c_{ij}).
$$

Weighting these velocities by
\(p(i,j\mid u)\propto\pi_{ij}\exp[-\|u-c_{ij}\|_2^2/(2q)]\)
gives the exact marginal velocity of the joint-mixture path. Positive \(\sigma_0\) avoids the discrete-path degeneracy in which almost every intermediate point identifies its endpoint pair.

For each plan, 2,048 pairs are sampled using common random uniforms and Gaussian fluctuations. The integral uses 129 equally spaced times. Standard errors are computed across per-pair time integrals, retaining dependence across time. The [recorded checks](materials/toy-rkhs-coupling/results.json) verify plan marginals to below \(10^{-12}\), endpoint covariance matching to below \(10^{-12}\), and the finite-difference continuity equation to below \(10^{-7}\). Refining quadrature from 65 to 129 times changes every mean action by less than \(0.001\). The source is a Gaussian-like finite mixture, so these numbers should not be compared as matched trials with the exact Gaussian source of Figure 17.

### A.9. What a refinement guarantee would require {#rkhs-refinement-scope}

An exact lift at each resolution does not by itself give a convergence rate between resolutions. For an \(H\)-orthogonal projection \(P_X\), suppose a field class satisfies \(\sup_f\|(I-P_X)f\|_H\le\delta_X\). Then

$$
0\le\|f-g\|_H^2-\|P_Xf-P_Xg\|_H^2
=\|(I-P_X)(f-g)\|_H^2\le4\delta_X^2.
$$

For the Gaussian kernel \(\kappa(f,g)=\exp[-\|f-g\|_H^2/(2\sigma^2)]\), the corresponding cost difference is at most \(4\delta_X^2/\sigma^2\). This also bounds the absolute difference between the entropic OT objective values for the original laws and their projections, under the usual standard-Borel and integrability assumptions. One direction follows by projecting a coupling and using KL data processing. The reverse direction lifts a projected coupling by independently drawing original fields conditional on their projections; this preserves its KL relative to the product marginals and incurs at most the stated cost error.

The rate is only as good as the estimate for \(\delta_X\). In Xu et al.'s Theorem 3, \(P_N\) is Fourier projection and the fields lie in a bounded \(H^\alpha\) ball, with error measured in \(H^k\), \(\alpha>k\). The squared tail is \(O(N^{-2(\alpha-k)/d})\). An observation-induced projection on irregular sensors needs its own approximation estimate. No rate for a learned flow follows from a bound on an OT objective value alone. A Gaussian law is also not supported in a fixed bounded Sobolev ball; extending such a uniform estimate to it needs moment or tail control.

Finally, distinguish the reconstruction RKHS from the ambient space carrying a Gaussian field. At finite observation dimension,

$$
Z_X\sim\mathcal N(0,G_X)
\quad\Longrightarrow\quad
\mathbb E\|\mathcal S_XZ_X\|_H^2
=\operatorname{tr}(G_X^{-1}G_X)=N.
$$

Thus these sources do not have uniformly bounded second moments in the RKHS as \(N\) grows. For an infinite-rank covariance expansion \(Z=\sum_k\sqrt{c_k}\xi_k\phi_k\), the formal covariance-RKHS norm is \(\sum_k\xi_k^2\), which diverges almost surely for independent standard Gaussian \(\xi_k\). The field can still be a valid random element of a weaker ambient Hilbert space when its covariance is trace class there. Its Cameron–Martin space describes shifts, not typical draws. Our finite-dimensional examples avoid this infinite-rank issue; a continuum extension must specify the ambient norm, data and source regularity, and approximation of both laws before applying a distributional error bound.

## References

1. Csillag, D., et al. [Functional Gradient Descent with Adaptive Representations](https://arxiv.org/abs/2606.16926). arXiv:2606.16926, 2026.
2. Kerrigan, G., Migliorini, G., and Smyth, P. [Functional Flow Matching](https://proceedings.mlr.press/v238/kerrigan24a.html). AISTATS, PMLR 238:3934–3942, 2024.
3. Bunker, J., et al. [Autoencoders in Function Space](https://www.jmlr.org/papers/v26/25-0035.html). JMLR 26(165):1–54, 2025.
4. Wang, S., et al. [FunDiff: diffusion models over function spaces for physics-informed generative modeling](https://doi.org/10.1038/s41467-026-72292-0). Nature Communications 17, 5749, 2026. Figure 15 uses the 2025 preprint version.
5. Serrano, L., et al. [Operator Learning with Neural Fields: Tackling PDEs on General Geometries](https://arxiv.org/abs/2306.07266). arXiv:2306.07266, 2023.
6. Yin, Y., et al. [Continuous PDE Dynamics Forecasting with Implicit Neural Representations](https://arxiv.org/abs/2209.14855). ICLR, 2023.
7. Chen, Y., and Vanden-Eijnden, E. [Scale-Adaptive Generative Flows for Multiscale Scientific Data](https://arxiv.org/abs/2509.02971v2). arXiv:2509.02971v2, 2026 revision.
8. Fotiadis, S., et al. [Adaptive Flow Matching for Resolving Small-Scale Physics](https://proceedings.mlr.press/v267/fotiadis25a.html). ICML, 2025.
9. Kovachki, N., et al. [Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs](https://jmlr.org/papers/v24/21-1524.html). JMLR 24(89):1–97, 2023.
10. Bruna, J., Peherstorfer, B., and Vanden-Eijnden, E. [Neural Galerkin Schemes with Active Learning for High-Dimensional Evolution Equations](https://doi.org/10.1016/j.jcp.2023.112588). Journal of Computational Physics 496, 112588, 2024.
11. Lee, K., and Carlberg, K. T. [Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders](https://doi.org/10.1016/j.jcp.2019.108973). Journal of Computational Physics 404, 108973, 2020.
12. Raissi, M., Perdikaris, P., and Karniadakis, G. E. [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561). arXiv:1711.10561, 2017.
13. Lu, L., et al. [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://doi.org/10.1038/s42256-021-00302-5). Nature Machine Intelligence 3, 218–229, 2021.
14. Li, Z., et al. [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895). ICLR, 2021.
15. Furuya, T., Taniguchi, K., and Okuda, S. [Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4b6ccf3acd6ccbc1093e093df345ba2-Abstract-Conference.html). ICLR, 2025; preprint first posted in 2024.
16. Calvello, E., Kovachki, N. B., Levine, M. E., and Stuart, A. M. [Continuum Attention for Neural Operators](https://www.jmlr.org/papers/v26/24-0879.html). JMLR 26(300):1–52, 2025.
17. Wang, S., Yu, X., and Perdikaris, P. [When and why PINNs fail to train: A neural tangent kernel perspective](https://arxiv.org/abs/2007.14527). arXiv:2007.14527, 2020.
18. Xu, F., Markovich, T., Barancikova, B., and Sun, Y. [Improving Function Space Flow Matching with Kernel Optimal Transport](https://arxiv.org/abs/2609.38049). arXiv:2609.38049v1, 2026.

## Cite this note

{{< cite-note >}}
