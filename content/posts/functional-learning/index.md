---
title: "How and why we redefine latent representations for PDE solutions as functional learning?"
date: 2026-10-01T10:47:44-04:00
weight: -1
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

An ideal update is \(u_{k+1}=u_k-\eta\nabla_H\mathcal L(u_k)\). In practice, we have to represent both \(u_k\) and its gradient using finitely many degrees of freedom.

This is where adaptive FGD makes a specific choice. It refines the gradient representation until its approximation error is small relative to the update being taken. To see why this helps, consider the simpler Hilbert-space condition

$$
\|g_k-\nabla_H\mathcal L(u_k)\|_H
\leq\varepsilon\|g_k\|_H.
$$

Write \(L_{\rm sm}\) for the loss smoothness constant—the paper's \(K\), distinct from a kernel. It bounds the quadratic remainder of the loss in the chosen norm. In the Hilbert-space case, the descent calculation gives

$$
\begin{aligned}
\mathcal L(u_k-\eta g_k)
&\leq\mathcal L(u_k)\\
&\quad-\eta\left(1-\varepsilon-\frac{L_{\rm sm}\eta}{2}\right)
\|g_k\|_H^2.
\end{aligned}
$$

The coefficient in parentheses determines whether the approximate update still decreases the loss. As the gradient becomes small, a fixed absolute approximation error can overwhelm it; a relative criterion tightens the accuracy accordingly. This Hilbert-space estimate follows from \(\langle\nabla_H\mathcal L,g_k\rangle_H\ge(1-\varepsilon)\|g_k\|_H^2\) and the smoothness inequality.

In the paper's toy reconstruction, fixed representations plateau while the adaptive representation adds detail and reduces the loss (Figure 2).

{{< figure src="figures/fgd-adaptive.png" link="figures/fgd-adaptive.png" alt="Three reconstruction sequences and training-loss curves comparing a neural network, fixed-resolution functional gradients, and adaptive functional gradients." caption="**Figure 2.** Reconstructions and loss curves in the FGD toy example, comparing neural, fixed, and adaptive representations. Source: [Csillag et al., Figure 1](https://arxiv.org/abs/2606.16926v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

The paper also allows approximations in a larger Banach space \(B\). At a nonstationary iterate, Algorithm 1 holds the current function fixed, fits \(g_m\), and computes

$$
U_m\ge\|g_m-\nabla\mathcal L(u)\|_B,
\qquad S_m=\|g_m\|_B.
$$

It refines, transfers the current function, and refits until

$$
\boxed{(1+\epsilon)U_m<\epsilon S_m,\qquad 0<\epsilon<1,}
$$

then applies \(u^+=u-\eta g_m\). This compares gradient error with gradient size; multiplying both by \(\eta\) gives the same relative update error. Smoothness controls the step size, while this test controls representation accuracy. The Banach-space theorem additionally needs the paper's gradient-compatibility assumptions ([Algorithm 1 and Section 3](https://arxiv.org/html/2606.16926v1#S3)).

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


## 6. Outlook

The established examples show what the functional formulation lets us specify and compute. Four questions follow from its engineering and accuracy requirements.

### Diagnosing blocked functional descent

**Can parameter optimization stall while a useful functional correction remains?** A PINN can already lie in \(H\); \(J_\theta J_\theta^*\) can help or hinder descent. [PINN neural tangent kernel analysis](https://arxiv.org/abs/2007.14527) studies related training-rate imbalances.

Fix the loss and \(H\). With \(g=\nabla_H\mathcal L(u_\theta)\ne0\) and \(\Pi_\theta\) projecting onto \(\operatorname{range}J_\theta\), record

$$
\rho_\theta=\frac{\|(I-\Pi_\theta)g\|_H}{\|g\|_H},
\qquad
\frac{d\mathcal L}{ds}=-\|J_\theta^*g\|^2.
$$

Using Section 4's Gram projection, \(\rho_\theta\) measures the unavailable correction. When it is small, inspect the nonzero Gram eigenvalues and corresponding field modes for poorly scaled directions. Loss curvature and the optimizer also matter.

A proposed Poisson test would compare Euclidean descent, Gram-corrected descent with the same network, and adaptive functional descent at matched solution accuracy, recording diagnostics and total cost. Does metric correction suffice, or must the representation gain directions? Accurate FGD realization remains necessary; a speedup is not guaranteed.

### Learned corrections with solver guarantees

FGD suggests a complementary object to learn: an update of the current solution. We could use a learned solution map for initialization and then refine the field:

$$
u^{(0)}=\mathcal S_\theta(a),
\qquad
\frac{du_s}{ds}=V_\theta(u_s;a).
$$

This is a proposed solver design. Its update might approximate a functional gradient, a preconditioned residual correction, or another justified numerical step. A gradient approximation can inherit a descent guarantee only when the geometry, error tolerance, and step conditions needed by that guarantee hold. Small PDE residuals also need a problem-specific stability estimate before they imply small solution errors.

Does learning updates transfer better across problems and meshes than learning only endpoints? Compare endpoint training with training on states and update directions, including correction cost at matched solution accuracy.


### Adaptive representations that preserve evolving laws

FFM learns a time-dependent operator \(v_\theta(\tau,\cdot):H\to H\), connecting velocity learning to FOL. **Could its representation adapt to both \(\tau\) and the current field?**

Training residuals may suggest where more resolution is useful, but Section 3 shows they include conditional variability. A minibatch loss alone cannot certify marginal-velocity accuracy or prescribe refinement.

**The velocity's definition matters.** Section 4 projects a specified field velocity into coordinates. [FunDiff](https://arxiv.org/html/2506.07902v2) predicts a code velocity \(g_{\theta,m}\) directly; for a fixed decoder, the chain rule gives

$$
\dot z=g_{\theta,m}(\tau,z),
\qquad
\dot u=D_m'(z)g_{\theta,m}(\tau,z).
$$

The decoder derivative converts code motion into field motion; it does not supply a desired velocity. Comparing that motion with itself gives zero. A mismatch indicator needs a separately evaluated reference:

$$
e_{\mathrm{ref}}(\tau,z;m)
=\|D_m'(z)g_{\theta,m}(\tau,z)
-v_{\mathrm{ref}}(\tau,D_m(z))\|_H.
$$

A compatible finer flow at the same time and field could supply the reference, accounting for transfer and reference errors. This is a consistency indicator, not a pure representation-error certificate. Enlarging the code requires a compatible velocity model; more decoder queries only refine evaluation.

Transfers must also control field error and newly resolved randomness. Suppose the coarse field stores \(X\), while the intended fine field is

$$
U=Xe_1+Ye_2,\qquad
Y=\rho X+\sigma\epsilon,\quad
\epsilon\sim\mathcal N(0,1)
$$

with independent \(X,\epsilon\). Copying into \((X,0)\) misses fine-scale variability; adding independent noise misses the correlation. New coefficients need the conditional law at refinement time.

[Adaptive Flow Matching](https://proceedings.mlr.press/v267/fotiadis25a.html) and [Scale-Adaptive Generative Flows](https://arxiv.org/abs/2509.02971v2) adapt sources or schedules. Refining coordinates additionally requires preserving the field and its intended law.

Test Gaussian laws with known velocities: compare fixed fine, time-only, and state-dependent representations at matched endpoint accuracy. Measure derivative statistics, mode correlations, and total cost, including transfers and error estimation.


### Sources conditioned on physics and observations

A source covariance describes which functions are plausible before transport. Compatible spectra can keep a transport regular, but a marginal field covariance may be poorly suited to a conditional inverse problem. Boundaries, forcing, and measurements can change both the uncertain scales and their correlations.

This points toward context-dependent functional sources, constraint-preserving velocities, and transports informed by PDE stability. Evaluate the generated law alongside physical residuals and observables; accurate-looking samples do not determine uncertainty calibration.

An appealing shortcut is to add a physical correction to an already trained generative velocity:

$$
v_\tau^{\mathrm{guided}}(u)
=v_\tau(u)-\lambda_\tau\nabla_H\mathcal L(u).
$$

This changes the transport and generally changes its endpoint law. It is a new sampling model whose target must be justified or evaluated. A more deliberate connection would specify the desired law first, then design or learn a velocity for it. The open question is how physical information can help learn that transport without losing the uncertainty we intended to represent.

The title's reinterpretation is therefore consequential: a latent code is a way to compute with a function through a decoder. Carrying that meaning through the whole model lets the same solution be observed, differentiated, corrected, and sampled. Functional learning is especially valuable when a PDE application needs several of these capabilities together. The research opportunity is to make their mathematical compatibility translate into reliable computation.

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

## Cite this note

{{< cite-note >}}
