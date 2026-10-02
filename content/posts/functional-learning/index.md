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

**Functional learning makes the field, its physical operations, and its required accuracy part of the learning problem.** We begin with those requirements, explain how arrays and codes describe functions, and then develop three learning tasks: finding a function, learning a map between functions, and generating a law over functions. Examples from functional gradient descent (FGD), functional operator learning (FOL), and functional flow matching (FFM) make these tasks concrete. Their shared ingredients lead to model design, error analysis, PDE applications, and the open questions at the end.

## 1. What do we need from a learned PDE solution?

A solver returns a numerical description of a field. Our learning task may need to change how that field is sampled, apply physical operators to it, or use it under new conditions. [FunDiff](https://doi.org/10.1038/s41467-026-72292-0) and related function-representation models bring together three requirements: heterogeneous observations, physical operations, and a common field across tasks. Incomplete observations add a fourth requirement: an explicit account of uncertainty.

### Learning from simulations on different meshes

Suppose one simulation stores a field on a \(64\times64\) mesh and another uses \(128\times128\). Their arrays have different sizes, but both sample a function on the physical domain. Write

$$
y_i=\mathcal O_i u_i+\varepsilon_i.
$$

Here \(\mathcal O_i\) records the sampling rule, such as point values or cell averages, and \(\varepsilon_i\) is observation error. For point values, choose a function space \(U\) with enough regularity to define evaluation; arbitrary \(L^2\) equivalence classes do not supply point values.

More query locations give more samples of a represented field. Identifying finer physical structure also requires enough information in the data and enough capacity in the representation. Changing the domain geometry brings additional requirements beyond changing the sample count.

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

| Task | What must be added to the representation |
| --- | --- |
| Reconstruction and denoising | An observation model and prior information |
| Forward PDE prediction | A map from problem data to solution fields |
| Physical forecasting | A dynamics model and time integration |
| Inverse inference | A likelihood or conditioning model and uncertainty assessment |
| Derived quantities | A functional such as an integral, flux, or gradient |

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

The limitations in the table identify requirements to examine, rather than failures of every method in a family. One documented example is [Krishnapriyan et al.'s PINN study](https://arxiv.org/abs/2109.01050): its convection and reaction–diffusion examples expose optimization difficulties despite adequate network expressivity. Simply making the network larger does not address every source of error.

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

[CORAL](https://arxiv.org/abs/2306.07266) makes this interpretation explicit. In Figure 1, input observations are encoded into \(z_a\); a processor predicts an output code \(\widehat z_u\); and the output decoder evaluates the predicted function at query locations \(\mathcal X\). The locations enter the decoder separately from the code. Changing them changes the returned samples of the function.

{{< figure src="figures/coral-function-codes.png" link="figures/coral-function-codes.png" alt="CORAL diagram: sampled input values are encoded into a code, a processor maps it to an output code, and a coordinate-query decoder evaluates the output function." caption="**Figure 1.** CORAL maps an input function code to an output code, then evaluates the output function at query coordinates. Source: [Serrano et al., Figure 2](https://arxiv.org/abs/2306.07266v2)." >}}

A fixed decoder selects a family \(\mathcal M=D(Z)\subset H\). Training the decoder changes the family; updating a code moves within it. [Autoencoders in Function Space](https://www.jmlr.org/papers/v26/25-0035.html) uses this separation to define reconstruction at the function level.

### Function spaces, observations, and physical accuracy

An observation operator \(\mathcal O_h:H\to Y_h\) describes how a field becomes data on a particular mesh or sensor set. A physical operator \(\mathcal A_a\) describes a PDE or measurement model, and a functional \(Q\) extracts a quantity of interest. Their domains and continuity properties determine which field errors matter.

For example, differentiation is not a bounded operation on \(L^2\), as the temperature-field example showed. If we need derivative accuracy, we must control a stronger norm or establish a separate estimate. Point observations require enough regularity to define point values; cell averages and weak observations may be more suitable for rough fields.

The decoder's output space, the norm in its training objective, and the numerical approximation of that norm must therefore match the task. A smooth network gives computable derivatives, but their accuracy is a further requirement.

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

The PDE requirements lead to three different targets. An objective defines the function we want to find; a family of equations defines the solution map we want to approximate; data and conditioning information define the law we want to generate. The following examples develop the mathematics of each target.

### Functional optimization (FGD): finding one solution {#functional-gradient-descent}

Functional optimization treats the candidate function as the unknown. [Functional Gradient Descent](https://arxiv.org/abs/2606.16926) provides a concrete way to connect its updates to geometry and representation error. We first derive the gradient, compare PINN parameter updates with functional updates for the same PDE loss, and then use the paper's examples to see what changes when the representation adapts.

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

A neural PDE model also describes a function, but its optimizer usually updates network weights. How does that familiar gradient relate to the functional gradient above? A single PDE loss lets us compare the two directly.

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

By the chain rule, backpropagation computes

$$
\begin{aligned}
\frac{\partial}{\partial\theta_j}\mathcal L_N(u_\theta)
&=D\mathcal L_N(u_\theta)
\left[\frac{\partial u_\theta}{\partial\theta_j}\right]\\
&=\sum_i w_i r_i(u_\theta)
\left[-\Delta\frac{\partial u_\theta}{\partial\theta_j}(x_i)\right]
\\
&=\left\langle g_N(u_\theta),
\frac{\partial u_\theta}{\partial\theta_j}\right\rangle_H.
\end{aligned}
$$

Equivalently, with \(J_\theta^*\) the adjoint for the chosen field inner product and Euclidean parameter inner product,

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

The extra operator in the PINN column follows by decoding the weight update. In continuous optimization time \(s\),

$$
\dot\theta=-J_\theta^*g_N,
\qquad
\dot u_\theta=J_\theta\dot\theta
=-J_\theta J_\theta^*g_N.
$$

For a finite step, this field change holds to first order, with an \(o(\eta)\) remainder for a differentiable nonlinear network. Adam or quasi-Newton optimization further changes the parameter update; the chain-rule relation between the two gradients still holds.

Each weight affects a whole function \(\partial_{\theta_j}u_\theta\). Parameter descent measures the alignment of \(g_N\) with these available directions, then combines them to move the field. It can suppress directions outside their span and rescale directions within it. **\(J_\theta J_\theta^*\) is generally neither the identity nor an orthogonal projector.** The resulting kernel-mediated dynamics are also studied in [PINN neural tangent kernel analysis, Section 3.1](https://arxiv.org/abs/2007.14527); the Jacobian identity itself does not require an infinite-width limit.

Thus a zero parameter gradient \(J_\theta^*g_N=0\) need not imply a zero functional gradient \(g_N=0\). Conversely, this comparison supplies no blanket guarantee that functional descent is cheaper or more accurate: its desired direction must still be computed and represented. Both methods also inherit the information limits of the sampled loss; matching finitely many residual values does not by itself establish an accurate PDE solution everywhere.

The function-space metric and the representation are two distinct choices. Holding the loss fixed, changing \(H\) changes \(g_N\) and its associated adjoint \(J_\theta^*\), while their product remains the same Euclidean parameter gradient whenever the loss and network derivatives are well defined in both spaces. We revisit the metric choice in Section 4. The immediate question for FGD is how to approximate the desired correction field without restricting every update to the tangent directions of one fixed neural representation.

#### Adaptive approximation of the gradient

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

Functional gradients and inexact-gradient methods predate this paper. Its distinctive contribution is the combination of adaptive gradient representations, computable error tests, and convergence guarantees under the stated assumptions. The authors' priority claim concerns an implementable FGD method with these guarantees in a general setting ([Introduction and Sections 3.1–3.2](https://arxiv.org/html/2606.16926v1)). That claim does not establish that every ingredient is new, or that the same guarantees hold for an arbitrary learned representation.

#### Two applications: wave equations and inverse rendering

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

Both applications specify an objective on functions and then approximate its updates. They seek an optimized function. If the initial conditions or coefficients change, however, the desired function changes too. The next task learns a map that serves an entire family of problems.

### Functional operator learning (FOL): predicting across problems {#functional-operator-learning}

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

DeepONet and FNO are established examples of **functional operator learning (FOL)**. Here FOL names the learning task, rather than a new acronym assigned by the authors to their method. We use **Furuya, Taniguchi, and Okuda's [Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4b6ccf3acd6ccbc1093e093df345ba2-Abstract-Conference.html)** as the mathematical example. Its construction connects operator layers to a convergent solution procedure. This lets us ask what each layer computes and why approximation errors remain controlled.

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

A short time interval can make this possible. For intuition, suppose \(\|E(t)\|\le M_E\), and \(\mathcal N\) has Lipschitz constant \(L_{\mathcal N}\) on the relevant bounded range. In a supremum-in-time norm, the integral contribution has Lipschitz bound \(M_E L_{\mathcal N}T\). Choosing \(T\) small enough makes it contractive, provided the map also stays inside that range. The paper uses semigroup smoothing estimates to work in mixed Lebesgue norms; Appendix I records its hypotheses.

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

The initial-data term is carried through the blocks. Kernel operations communicate across locations, while \(\mathcal N_\theta\) acts pointwise. The same block can be reused at every iteration. This is the construction behind the paper's neural-operator approximation, rather than a claim that every trained operator architecture performs Picard iteration ([Section 4.1 and Remark 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)).

{{< figure src="figures/functional-operator-picard.svg" link="figures/functional-operator-picard.svg" width="720" alt="Two parallel constructions map initial data to an entire trajectory. Exact Picard iteration converges to the PDE solution; finite kernel and neural approximations produce an operator network. A bound separates iteration error from block approximation error." caption="**Figure 5.** A solution map built from a repeated functional update. The upper row is exact Picard iteration; the lower row approximates its blocks. The same construction serves a family of initial functions. Original LaTeX/TikZ exposition of the mechanism in [Furuya et al., Section 4.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf)." >}}

We can see the error mechanism without inspecting all the network weights. Suppose the approximate block has uniform defect at most \(\eta\) on the invariant ball, and its iterates remain there. With \(e_k=\|\widehat u^{(k)}-\mathcal S(a)\|_{\mathcal U}\), the triangle inequality gives

$$
e_{k+1}\le q e_k+\eta,
\qquad
\boxed{e_J\le q^J e_0+\frac{1-q^J}{1-q}\eta.}
$$

The first term is unfinished solution iteration. The second is the cost of approximating its blocks, amplified by stability. Greater depth reduces the first term; improving kernels, nonlinearities, or numerical quadrature reduces the second. **Depth alone cannot remove the approximation floor.** Appendix I derives this bound and distinguishes the continuum construction from its numerical evaluation.

#### What the FOL approximation theorem guarantees

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

#### A contemporary architectural connection

[**Continuum Attention for Neural Operators** (Calvello et al., JMLR 2025)](https://www.jmlr.org/papers/v26/24-0879.html) gives a complementary example. Attention can be defined on functions through an integral:

$$
\operatorname{Att}(u)(x)
=\frac{\int_\Omega e^{\langle Qu(x),Ku(y)\rangle}Vu(y)\,dy}
{\int_\Omega e^{\langle Qu(x),Ku(y)\rangle}\,dy}.
$$

Here \(Q,K,V\) are pointwise linear query, key, and value maps. Finite attention approximates this operation numerically; quadrature weights matter on nonuniform nodes. The paper proves universality for a specified modified transformer operator on compact input sets, including results in differentiable-function and Sobolev norms. Those are existence guarantees, with assumptions on the spaces and architecture (Theorems 22–23), rather than an error rate for arbitrary meshes. Its spatial attention weights also differ from FFM's probability law over whole functions.

Figure 6 shows how the continuum-attention paper realizes an operator architecture. The input function is combined with coordinates, lifted to a feature function, processed by attention blocks, and projected to an output function. In the paper's notation the output is \(z(x)\); this denotes a function, while our \(z\) denotes a latent vector.

{{< figure src="figures/continuum-attention-architecture.png" link="figures/continuum-attention-architecture.png" alt="Transformer neural operator architecture: an input function and coordinates are lifted, processed by repeated attention encoder layers, and projected to an output function." caption="**Figure 6.** An implemented operator architecture with attention acting on feature functions. Source: [Calvello et al., Figure 1](https://www.jmlr.org/papers/v26/24-0879.html), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

#### Example: mapping a Darcy coefficient field to its solution

For a uniformly positive coefficient \(a(x)\), consider

$$
-\nabla\cdot(a(x)\nabla u(x))=1
\quad\text{in }(0,1)^2,
\qquad u|_{\partial\Omega}=0.
$$

The operator takes an input coefficient function to an output solution function. Figure 7 shows two inputs and their corresponding predicted solutions, produced with the same model. Its columns show the input visualization, reference solution, prediction, and pointwise absolute error on a logarithmic scale. The two rows are the test samples with median and maximum relative \(L^2\) error.

{{< figure src="figures/continuum-attention-darcy.png" link="figures/continuum-attention-darcy.png" alt="Darcy operator-learning examples with input fields, reference solutions, predictions, and log-scale pointwise error for the median and maximum relative-error samples." caption="**Figure 7.** Darcy predictions from the Fourier attention neural operator variant in the continuum-attention paper. These are experiments from a separate 2025 operator-learning source, rather than numerical results of the parabolic approximation theorem. Source: [Calvello et al., Figure 12](https://www.jmlr.org/papers/v26/24-0879.html), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

The example uses the paper's Fourier attention variant, which modifies the attention parameterization. It illustrates an implemented solution map; the Picard theorem above concerns a different, semilinear parabolic family. The theoretical construction and the numerical example answer complementary questions about operator learning ([Sections 4 and 6.2.2](https://www.jmlr.org/papers/volume26/24-0879/24-0879.pdf)).

Random inputs \(a\sim\rho\) induce a solution law \(\mathcal S_\#\rho\). When observations leave several fields possible, we may instead want to learn a conditional law directly. This brings us to functional transport learning.

### Functional transport learning (FFM): generating solution distributions {#functional-flow-matching}

Suppose we have samples of functions: solution fields from a simulator, for example. We want to generate new functions from the same distribution. FFM starts with a reference distribution of random functions and learns a velocity that transports it toward the data distribution.

Figure 8 illustrates one conditional path. Each blue curve is a complete function. As generative time advances, the curve moves from noise toward a target function; the arrows show the required functional velocity. We will construct such paths first, then see how their velocities give a training objective.

{{< figure src="figures/ffm-function-flow.png" link="figures/ffm-function-flow.png" alt="Four stages of a noisy function evolving toward a sine curve, with arrows showing its function-space velocity." caption="**Figure 8.** A conditional function-space path from a noisy function toward a target curve. Arrows indicate its functional velocity. Source: [Kerrigan et al., Figure 1](https://arxiv.org/abs/2305.17209v2)." >}}

#### Constructing a path between noise and a function

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

This averages the velocities of the conditional paths arriving at \(u\). The regression identity in Appendix B explains why fitting these sampled velocities also fits the marginal velocity, up to a parameter-independent variance term.

After training, generate a function by drawing \(u_0\sim\mu_0\) and integrating

$$
\frac{du_\tau}{d\tau}=v_\theta(\tau,u_\tau).
$$

If \(\Phi_\tau\) denotes the resulting flow map, then the evolved distribution is \(\mu_\tau=(\Phi_\tau)_\#\mu_0\): draw from the initial law and apply the map. Training uses directly sampled paths; generation uses an ODE solver. The transport interpretation requires the flow and measure assumptions developed in [Kerrigan et al., Sections 3–4](https://proceedings.mlr.press/v238/kerrigan24a.html), discussed further in Appendix C.

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

The FFM paper tests generation of \(64\times64\) Navier–Stokes fields using an FNO velocity model. Figure 9 compares independent data and generated samples, together with errors in density and spectral statistics. This is a distribution-learning experiment: the generated panels are new draws, rather than predictions paired with particular ground-truth fields.

{{< figure src="figures/ffm-navier-stokes-samples.png" link="figures/ffm-navier-stokes-samples.png" alt="Independent Navier–Stokes field samples from the dataset, FFM, DDPM, DDO, and GANO, with a table comparing density and spectrum errors." caption="**Figure 9.** Samples and distributional diagnostics in the FFM Navier–Stokes experiment. Matching a collection of field statistics differs from solving an individual initial-value problem. Source: [Kerrigan et al., Figure 2 and Table 2a](https://proceedings.mlr.press/v238/kerrigan24a.html)." >}}

The generative ODE advances \(\tau\), not the fluid's physical time. The experiment supports an empirical comparison of field distributions; it does not establish that each generated field satisfies a specified PDE trajectory.

#### Example: conditioning a function law on observations

The same transport idea can use side information \(c\). Draw training functions from the conditional data law and fit

$$
\mathcal J_{\rm cond}(\theta)
=\mathbb E\|v_\theta(\tau,u_\tau;c)-w_\tau\|_H^2.
$$

The population target becomes \(\mathbb E[w_\tau\mid\tau,u_\tau,c]\). Figure 10 illustrates this on the paper's AEMET temperature curves. Several generated curves share the conditioning information while varying elsewhere. This is a time-series example, separate from the Navier–Stokes experiment.

{{< figure src="figures/ffm-conditional-functions.png" link="figures/ffm-conditional-functions.png" width="720" alt="Conditional FFM temperature curves: dark generated curves, pale data curves, and black observations, comparing conditional training with additional conditional sampling." caption="**Figure 10.** Conditional function generation on AEMET. The left column uses conditional training; the right additionally modifies sampling to enforce the observations. Source: [Kerrigan et al., Figure 5](https://proceedings.mlr.press/v238/kerrigan24a.html)." >}}

Conditioning the learned velocity and enforcing observations during sampling are different mechanisms. The right column demonstrates the latter's effect; exact agreement at observed points does not by itself establish a correct posterior law ([Section 5 and Appendix A.4](https://proceedings.mlr.press/v238/kerrigan24a/kerrigan24a.pdf)). The paper's FNO implementation also uses uniform grids, so its continuum formulation should be distinguished from that numerical restriction.

| | FGD: functional gradient descent | FOL: the Picard construction | FFM: functional flow matching |
| --- | --- | --- | --- |
| Object | One function | A map between functions | A law over functions |
| Goal | Decrease a functional | Approximate a solution map uniformly over inputs | Transport a probability law |
| Update mechanism | Negative Riesz gradient | Repeated approximate fixed-point blocks | Learned generative velocity |
| Essential structure | Geometry and gradient accuracy | PDE well-posedness, contraction, kernel and nonlinear approximation | Measures, covariance, conditional paths |
| Iteration meaning | Optimization time | Refinement of a whole candidate trajectory | Generative time |

FGD obtains its velocity from an objective and an inner product. The operator construction obtains its blocks from a PDE integral equation. FFM obtains its velocity by regression against a probability path; that velocity need not be a gradient field. These supply three distinct learning objects, with shared questions about representation, approximation, and stability.

## 4. Designing a functional PDE model

With the three learning targets in place, we can choose the representation and computations that realize them. This section concerns the common design ingredients: physical operations, coordinate geometry, available updates, source laws, and numerical evaluation.

### The shared building elements

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

An operator can predict a solution directly or construct it through iterations such as Figure 5. When we use a continuous evolution, its parameter \(r\) has a meaning supplied by the task. In FGD it is optimization time \(s\), with \(V=-\nabla_H\mathcal L\). In FFM it is generative time \(\tau\), with a learned velocity chosen to transport laws. An evolution PDE supplies a third possibility: physical time \(t\), with a velocity specified by the governing equation. Figure 11 separates these continuous motions from the coordinates used to implement them.

{{< figure src="figures/functional-learning-elements.svg" link="figures/functional-learning-elements.svg" width="720" alt="A functional learning problem specifies a field, physical operations, geometry, and a target. Its motion can be optimization, physical evolution, or generative transport. A decoder and numerical solver realize the chosen motion in finite coordinates." caption="**Figure 11.** Different tasks supply different motions of a function. The decoder and numerical solver must realize the chosen motion and preserve the quantities the task needs. Original explanatory schematic, drawn in LaTeX/TikZ." >}}

This is what makes the functional viewpoint useful: the same field can be observed on several meshes, evaluated by a physical operator, optimized, or sampled from a law. Each operation has requirements that can be stated before we settle on its coordinates. A continuous decoder is one ingredient; the rest of the learning problem must respect the function it describes.

### Function-space and representation geometry {#geometry-induced-by-the-representation}

There are two choices to distinguish: the inner product used to define a functional gradient, and the coordinates used to represent its motion. Section 3 compared neural parameter descent with functional descent for a fixed loss. We now examine the metric choice, then the geometry a decoder induces on its coordinates.

#### Choosing the function-space metric

The inner product is a separate design choice. To isolate its effect, return to the same Poisson boundary-value problem:

$$
\begin{cases}
-\Delta u^*(x)=f(x), & x\in\Omega,\\
u^*(x)=0, & x\in\partial\Omega.
\end{cases}
$$

For this comparison, use the variational Dirichlet energy. This is a different objective from the sampled PINN residual loss in Section 3; here we hold this energy fixed and change only the function-space metric:

$$
\mathcal E(u)=\frac12\int_\Omega|\nabla u|^2\,dx
-\int_\Omega fu\,dx,
\qquad u\in H_0^1(\Omega).
$$

The space \(H_0^1(\Omega)\) encodes the zero boundary condition and square-integrable first derivatives.

For \(f\in L^2(\Omega)\) on a suitable domain, differentiating this energy in direction \(h\in H_0^1(\Omega)\) gives

$$
D\mathcal E(u)[h]
=\int_\Omega\nabla u\cdot\nabla h\,dx
-\int_\Omega fh\,dx.
$$

At the minimizer, this derivative is zero for every admissible \(h\). That is exactly the weak form of the Poisson equation above. The energy therefore gives us an optimization objective for finding its solution; now we can compare how two gradient choices reduce that same objective.

Write \(A=-\Delta\) for the weak Dirichlet operator and \(r_k=Au_k-f\) for the current PDE residual. Integration by parts identifies the formal \(L^2\) gradient as this residual. In the energy inner product, the gradient \(g\) instead satisfies

$$
\int_\Omega\nabla g\cdot\nabla h\,dx
=D\mathcal E(u)[h].
$$

The last equation says \(Ag=Au-f\), hence \(g=A^{-1}(Au-f)=u-u^*\). This gives a direct comparison:

| For the same energy \(\mathcal E\) | Residual update in \(L^2\) | Energy-metric update in \(H_0^1\) |
| --- | --- | --- |
| Inner product | \(\int_\Omega gh\,dx\) | \(\int_\Omega\nabla g\cdot\nabla h\,dx\) |
| Gradient at \(u_k\) | \(r_k=Au_k-f\) | \(A^{-1}r_k=u_k-u^*\) |
| Descent step | \(u_{k+1}=u_k-\eta r_k\) | \(u_{k+1}=u_k-\eta A^{-1}r_k\) |
| How the field changes | Subtract the residual at each location | Solve \(Ag_k=r_k\), then subtract that whole correction field |
| Work needed for the direction | Evaluate the PDE residual | Apply an inverse elliptic operator to the residual |

**Both columns are functional gradient descent.** The residual update already couples neighboring values through the Laplacian. The energy metric adds a global elliptic solve that converts the residual into a correction. The \(L^2\) expression requires \(Au\in L^2\); the energy and its weak gradient are defined on \(H_0^1\). Boundary conditions must also be enforced in a numerical update.

The difference is especially visible for a rapidly oscillating error. On \(\Omega=(0,\pi)\), choose \(f(x)=\sin x\), so \(u^*(x)=\sin x\), and let

$$
u_k(x)=\sin x+\delta\sin(nx),\qquad n\in\mathbb N,\ n\geq2.
$$

The residual is \(r_k(x)=n^2\delta\sin(nx)\), whereas the energy gradient is \(A^{-1}r_k(x)=\delta\sin(nx)\). After one step, the two error amplitudes are

$$
\begin{aligned}
\text{Residual update:}\quad
u_{k+1}-u^*&=(1-\eta n^2)\delta\sin(nx),\\
\text{Energy-metric update:}\quad
u_{k+1}-u^*&=(1-\eta)\delta\sin(nx).
\end{aligned}
$$

For the residual update, higher frequencies require a smaller explicit step: this mode contracts only when \(0<\eta<2/n^2\). The energy-metric update contracts every mode by the same factor for \(0<\eta<2\). Its favorable scaling comes at a cost: computing \(A^{-1}r_k\) requires an elliptic solve. In this linear example, an exact energy-gradient step with \(\eta=1\) already solves the original problem; the inverse operator is doing that work.

The inner product determines **which correction field counts as the gradient**. This is a separate choice from optimizing network weights or taking functional updates. Once the function-space metric is chosen, a representation introduces another geometry, through its available directions and their scaling.

#### Geometry induced by the decoder

The PINN Jacobian calculation extends to any differentiable decoder. Let \(u=D(z)\), with differentiable \(D:\mathbb R^m\to H\). A small code change \(\delta z\) produces the first-order field change \(D'(z)\delta z\). The decoder derivative therefore determines both which directions are available and how large they are in the function norm.

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

### Representing fields and their update directions

If the code follows \(\dot z=b_\tau(z)\), then

$$
\dot u=D'(z)b_\tau(z),\qquad
\mu_\tau=D_\#\nu_\tau,
$$

where \(\nu_\tau\) is its code law. With a locally invertible chart on the represented family, this identifies a field velocity. If several codes represent the same field, their decoded velocities must agree for a single-valued deterministic field dynamics to be defined. Otherwise the code carries extra state beyond the function.

A decoder can reconstruct the endpoint fields well while giving poor tangent directions between them. Reconstruction error and velocity approximation consequently measure different aspects of the representation. The tangent directions are part of what must be assessed when the representation is used for dynamics.

#### Directions available to the dynamics

Let \(\mathcal M_m=D_m(Z_m)\) be a represented family, with a well-defined tangent space at \(u\). A measure of missing dynamics is

$$
\eta_m(\tau,u)
=\|(I-\Pi_{T_u\mathcal M_m})v_\tau(u)\|_H.
$$

The quantity \(\eta_m\) measures the velocity lost by projecting onto the available tangent directions. It can be large even when the current field itself is reconstructed well.

This is related to [dynamical low-rank approximation](https://doi.org/10.1137/050639703), which projects derivatives onto the tangent space of a moving approximation manifold. For a transport model, the projected velocity must also be assessed through its effect on the generated distribution.

FGD has a computable error test in its analyzed settings. For a learned transport, the exact population velocity is usually unavailable. A richer model, hierarchical detail modes, or an analytic reference problem can estimate missing directions, but disagreement alone is not a certified error bound.

If the decoder itself changes continuously with generative time, the chain rule also requires

$$
\frac{du_\tau}{d\tau}
=\partial_\tau D_\tau(z_\tau)
+D_\tau'(z_\tau)\dot z_\tau.
$$

The first term is motion caused by the changing decoder itself. It must be included in the intended field velocity.

#### Function autoencoders and physical structure

FunDiff also uses physical structure in the representation itself. For a sufficiently smooth two-dimensional streamfunction \(\psi\), define

$$
\mathbf v=(\partial_y\psi,-\partial_x\psi),
\qquad\nabla\cdot\mathbf v=0,
$$

The two mixed derivatives cancel, so this velocity is divergence-free by construction. A residual penalty takes another route: it penalizes violations at the points and with the weights used in the loss.

Figure 12 shows the FunDiff pipeline in the preprint. A Vision Transformer processes the sampled inputs, and a Perceiver encoder handles variable discretizations and maps them into a common latent representation. The decoder uses cross-attention between query coordinates and encoded features to evaluate the function. Physical constraints enter this function autoencoder through its architecture or training loss.

The second stage trains a Diffusion Transformer using rectified flow on the learned latent representation. Generation integrates a latent ODE from Gaussian noise and decodes its output into a function. Thus physical-prior enforcement is carried by the autoencoder and is decoupled from the generative model's training and inference.

{{< figure src="figures/fundiff-framework.png" link="figures/fundiff-framework.png" alt="FunDiff architecture: an encoder and coordinate-query decoder with physical priors, followed by latent diffusion training and inference for several field-reconstruction tasks." caption="**Figure 12.** FunDiff combines a function encoder and coordinate decoder, physical priors, and latent generation. Complete original diagram. Source: [Wang et al., Figure 1, preprint v2](https://arxiv.org/abs/2506.07902v2), [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)." >}}

We must also decide when a constraint should hold. For a linear constraint \(Au=0\), a source in its kernel and a velocity satisfying \(Av=0\) preserve the constraint along a sufficiently regular path. For a nonlinear constraint \(\mathcal C(u)=0\), the velocity must be tangent: \(D\mathcal C(u)[v]=0\). Requiring physical validity at every generative time restricts the source and path more strongly than requiring it only of final samples.

### Reference laws, transport paths, and numerical computation

| What adapts | Purpose | What must be checked |
| --- | --- | --- |
| Source covariance | Align stochastic scales with the field geometry | The continuum law and transport regularity |
| Interpolation schedule | Allocate generative time across scales | Endpoint law and integration error |
| Representation | Add or move directions available to the dynamics | Velocity defect, transfer error, and new-mode law |
| Solver step size | Integrate a given represented velocity accurately | Local and accumulated numerical error |

[Scale-Adaptive Generative Flows](https://arxiv.org/abs/2509.02971v2) studies the source-spectrum and interpolation-schedule choices. Their effect is visible in Figure 13. For the Gaussian-field experiment shown, a spectrum-matched source tracks the target spectrum with five RK4 steps. The white-source curves retain excess energy at finer frequencies at the displayed integration budgets, across the three resolutions.

{{< figure src="figures/scale-adaptive-spectra.png" link="figures/scale-adaptive-spectra.png" alt="Energy spectra at 32 by 32, 64 by 64, and 128 by 128 resolutions, comparing target Gaussian fields with generated fields using spectrum-matched or white noise and different integration budgets." caption="**Figure 13.** Gaussian-field energy spectra at 32², 64², and 128² resolution, comparing spectrum-matched and white sources at the displayed RK4 budgets. Source: [Chen and Vanden-Eijnden, Figure 2](https://arxiv.org/abs/2509.02971v2)." >}}

[Adaptive Flow Matching for Resolving Small-Scale Physics](https://proceedings.mlr.press/v267/fotiadis25a.html) uses an encoded base distribution and adaptive noise scaling. Representation refinement changes which directions are available to the velocity at a given time; its algorithmic design is developed in the Outlook. Source covariance, interpolation, representation, and solver steps can each affect accuracy and cost, through the different mechanisms in the table.

#### Numerical observations and function-space losses

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

We now ask when finite representations and learned dynamics approximate the intended functional objects. Physical stability connects errors to PDE solutions, while approximation and transport estimates connect them to laws. The construction below develops the accompanying theory draft. Projection, conditional regression, and stability supply standard ingredients; the source-conditioning comparison remains a proof draft whose independent review and priority are open.

### Physical norms and PDE stability

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

### Representation error in fields and distributions

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

#### Changing a representation and its law

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

### Constructing regular approximating transports

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

### Learning error and numerical error

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

Finally, let \(E_{\rm num}\) bound the Wasserstein difference between the numerical sampler and the exact learned ODE. Figure 14 organizes the resulting error budget:

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

{{< figure src="figures/functional-theory-error-chain.svg" link="figures/functional-theory-error-chain.svg" width="720" alt="A LaTeX diagram tracing a target function law through projection, prototype approximation, Gaussian smoothing, a learned flow, and numerical sampling. Each transition is labeled by its contribution to the physical Wasserstein error." caption="**Figure 14.** From a target law to a computed sample law: each approximation introduces an error measured in the same physical geometry. Original LaTeX/TikZ diagram. The learning term assumes a stable learned velocity and population excess risk." >}}

The budget makes a consistency claim testable. Refinement must remove the representation floor; prototype and smoothing errors must vanish; learning must achieve \(e^{\widehat\Lambda}\sqrt{\mathcal E}\to0\); and integration error must vanish. A low empirical training loss alone establishes none of these limits. Statistical estimation, model approximation, and optimization determine the excess risk; solver accuracy determines the final term.

The result also transfers to physical outputs: for an \(L_Q\)-Lipschitz observable, \(W_2(Q_\#\widehat\mu,Q_\#\mu)\le L_QW_{2,H}(\widehat\mu,\mu)\). With the Sobolev norm above, it controls gradient-law error in \(L^2\). PDE satisfaction and conservation still need their own operator assumptions.

Changing representations introduces another numerical contribution. Appendix D separates initial error, accumulated velocity defect, and jumps in the represented function under a state-Lipschitz reference velocity. This is a different coupling estimate from the learned-field bound above. Any refinement rule must account for both the size of a transfer error and the remaining flow's sensitivity to it.

### Source geometry and resolution dependence

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

### Information lost through finite observations

The construction above starts with full functions or their exact coefficients. Real training data provide \(Y=\mathcal O_hU+\xi\). Even an unlimited deterministic reconstructor has an information floor:

$$
\inf_{\mathcal R}\mathbb E\|U-\mathcal R(Y)\|_H^2
=\mathbb E\|U-\mathbb E[U\mid Y]\|_H^2.
$$

If two admissible fields have identical observations, querying a decoder more densely cannot identify which one was observed. A conditional generative model can instead represent their conditional law. That is a different target from exact recovery of each individual field.

For fixed orthogonal projection \(P_m\), the projected marginal of an exact fine flow has the formal effective velocity

$$
\bar v_m(\tau,w)
=\mathbb E[P_mv_\tau(U_\tau)\mid P_mU_\tau=w],
$$

under suitable integrability, where \(w\) is a resolved field state. This need not equal \(P_mv_\tau(w)\). Unresolved modes can influence resolved motion. Consequently, removing modes, learning their average effect, and later restoring them is a closure problem as well as a representation problem.

There is an analogous issue for learned velocities. The resolved regression target is
\(\mathbb E[P_m w_\tau\mid P_mX_\tau]\), which averages over unresolved information. It need not equal a continuum velocity evaluated on a truncated field. This is a closure issue: the resolved dynamics average over information that the coarse field does not retain. [Shikhman's Theorem 13](https://arxiv.org/html/2608.04531v1) exhibits a Lipschitz continuum velocity whose finite conditional targets lack a uniform Lipschitz envelope; the example still has convergent flows. Uniform Grönwall bounds are therefore sufficient tools whose failure does not by itself prove failure of convergence.

The theory has now located the missing work precisely. We need observation consistency, approximation of the correct conditional velocity, control of learned dynamics, and accurate numerical realization. Functional learning makes these requirements refer to one common object. It gives a route from representation to PDE operations and solution laws; realizing that route requires checking each link.

## 6. Using the framework for PDE problems

The construction and accuracy conditions now have concrete uses. The following examples apply the three learning targets to prediction, physical evolution, inference, and correction. The proposed extensions are collected in the Outlook that follows.

### Forward prediction and physical evolution

For forward prediction, a neural operator returns a solution function from the problem data. CORAL's code processor and the operator examples in Section 3 realize this structure in different ways. A physical solver or functional optimization can then assess or refine a predicted field using the governing equation; the residual-to-solution estimate in Section 5 explains when a residual is informative.

Physical forecasting has a different time variable: it follows the evolution of the field itself. [DINo](https://arxiv.org/abs/2209.14855) learns latent ODE dynamics, while Neural Galerkin uses the governing PDE to determine the represented velocity.

#### When the PDE supplies the velocity

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

### Inverse inference and conditional field generation

Inverse PDE and rendering problems often admit several fields consistent with the data. Functional optimization can find a candidate; functional transport can aim to describe their uncertainty. To connect them, we need to decide what probabilities the candidates should have.

#### A descent ensemble and a specified target law

Initialize a random function and apply functional gradient flow to each realization. Under sufficient regularity, its law moves according to the same weak continuity equation as any other deterministic transport. Moreover,

$$
\frac{d}{ds}\mathbb E_{u\sim\mu_s}\mathcal L(u)
=-\mathbb E_{u\sim\mu_s}\|\nabla_H\mathcal L(u)\|_H^2.
$$

The ensemble's expected loss decreases. Its eventual distribution depends on the initial law and the attraction basins of the objective. If several solutions fit the observations, random initialization followed by optimization does not prescribe their relative probabilities.

#### A reference measure and an observation likelihood

Given a reference law \(\mu_0\), we can define

$$
\frac{d\pi_\beta}{d\mu_0}(u)
=Z_\beta^{-1}\exp[-\beta\mathcal L(u)].
$$

Assume the objective is measurable and \(0 < Z_\beta < \infty\). The reference law specifies the starting notion of plausible functions; the objective favors functions that fit the physical task. The scale \(\beta\) controls the strength of this preference. When the objective is a properly specified negative log likelihood, this has a Bayesian interpretation; an arbitrary residual penalty is a modeling choice. Densities are taken relative to a function-space reference measure, as in [Stuart's Bayesian formulation](https://doi.org/10.1017/S0962492910000061).

A transport model would seek \(T_\#\mu_0\approx\pi_\beta\). This supplies a distributional target for a PDE or rendering problem. Learning the transport requires target samples, as in ordinary sample-based flow matching, or another justified procedure for learning from the objective.

For inverse PDE inference, the unknown function might be a coefficient field whose forward solution must match observations. For inverse rendering, it might describe density and color whose rendered images must match photographs. The operator and the likelihood change; the construction still concerns a law over possible functions.

Function-space Bayesian inversion and [functional normalizing flows](https://arxiv.org/abs/2411.13277) provide foundations for learning such transports. The sampler must be assessed through its target distribution as well as the physical objective.

[GeoFunFlow](https://arxiv.org/abs/2509.24117) combines a geometry-aware function representation with conditioning for posterior field generation. This supplies the inference mechanism needed in addition to a decoder. The conditional FFM curves in Section 3 illustrate why many field realizations can share observations; posterior assessment must also check the intended uncertainty.

### Probabilistic correction of approximate solutions

| Intended result | Role of the physical model | What success means |
| --- | --- | --- |
| One solution | Define a residual or variational objective | Small solution error under suitable stability estimates |
| A distribution of solutions | Define a target law or reweight a reference law | Correct distribution as well as physical consistency |
| Improved approximate solutions | Define corrections between coarse and fine field laws | Reduced error and a preserved target distribution |

[Residual-augmented flow matching operators](https://arxiv.org/abs/2512.12749v3) uses the third route. A low-fidelity solver first predicts a field. The generative model learns a distribution of residual functions conditioned on that prediction and the problem input; adding a residual gives a corrected field.

In Figure 15, the upper branch supplies the low-fidelity context. The lower branch transports a Gaussian reference to residual functions. This changes what the generative model must produce: the uncertainty and structure of the correction, rather than the entire field.

{{< figure src="figures/residual-function-transport.png" link="figures/residual-function-transport.png" alt="A low-fidelity PDE solution conditions a functional flow that transports a Gaussian reference toward a distribution of residual functions, which correct the low-fidelity solution." caption="**Figure 15.** A Gaussian reference is transported to residual functions, conditioned on the problem input and a low-fidelity prediction. Source: [Bhola and Duraisamy, Figure 1](https://arxiv.org/abs/2512.12749v3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." >}}

Distributional accuracy and physical accuracy remain separate objectives. [Physics vs Distributions](https://arxiv.org/abs/2506.08604v3) studies their tension explicitly. A model can reduce a PDE residual while changing the distribution it was meant to reproduce.

## 7. Outlook

The established examples show what the functional formulation lets us specify and compute. The remaining opportunity is to turn those mathematical requirements into effective learned algorithms. Four questions follow from the framework and its limits.

### Learned corrections with solver guarantees

FGD suggests a complementary object to learn: an update of the current solution. We could use a learned solution map for initialization and then refine the field:

$$
u^{(0)}=\mathcal S_\theta(a),
\qquad
\frac{du_s}{ds}=V_\theta(u_s;a).
$$

This is a proposed solver design. Its update might approximate a functional gradient, a preconditioned residual correction, or another justified numerical step. A gradient approximation can inherit a descent guarantee only when the geometry, error tolerance, and step conditions needed by that guarantee hold. Small PDE residuals also need a problem-specific stability estimate before they imply small solution errors.

The interesting question is whether learning the update transfers better across a family of problems than learning only their endpoints. Can the same correction act on different mesh samplings or improve an imperfect initial prediction? The comparison should measure physical quantities and total cost at matched solution accuracy, including the correction steps.

#### Training a representation for its required updates

Suppose an autoencoder reconstructs every training snapshot well. That does not establish that its tangent directions can express the PDE velocity or a useful correction. A representation trained for a solver might therefore include both kinds of error:

$$
\mathbb E\left[
\|D(E(u))-u\|_H^2
+\gamma\inf_b
\|D'(E(u))b-V(u;a)\|_H^2
\right].
$$

This illustrative objective asks the representation to capture states and their required changes. The weight \(\gamma\) must account for units and scaling; the velocity must be available or estimated. Constraints such as boundary conditions may need to be built into the represented family as well.

The question becomes concrete: **can a representation with similar reconstruction error support more accurate or cheaper functional updates?** This connects representation learning to solver design, and motivates adaptive capacity when the required directions change.

A controlled comparison can start with the linear dictionary from Section 4, where the metric and projected velocity are known exactly, then move to nonlinear decoders. Compare Euclidean and induced-metric updates through field error, derivative error, conditioning, and computational cost.

A controlled study can compare a frozen decoder with one trained on both states and update directions. Measure field and derivative errors, residuals, descent or stability properties, and total correction cost at matched physical accuracy. The question is whether representing the required updates improves a solve beyond representing its snapshots.

### Adaptive representations that preserve evolving laws

The missing-velocity quantity and transfer examples above suggest a representation that grows with the dynamics. The design question is how to choose refinement times while controlling the accumulated error and preserving the current law.

FGD offers two possible connections to FFM. We could use functional optimization to **learn the velocity**, or borrow error-controlled refinement to **compute its transport**. These are different algorithms, with different unknowns and guarantees.

#### Using FGD to learn an FFM velocity

Hold the probability path fixed. Its population regression objective can be viewed as a functional of the whole velocity map \(v:(\tau,u)\mapsto H\). The appropriate Hilbert space is

$$
\mathcal V=L^2(d\tau\,\mu_\tau(du);H),
\qquad
\|v\|_{\mathcal V}^2
=\int_0^1\int_H\|v(\tau,u)\|_H^2\,\mu_\tau(du)\,d\tau.
$$

For square-integrable targets, Appendix B's conditional-regression identity gives

$$
\mathcal J(v)-\mathcal J(v^*)=\|v-v^*\|_{\mathcal V}^2,
\qquad
\nabla_{\mathcal V}\mathcal J(v)=2(v-v^*).
$$

The unrestricted population problem is a strongly convex quadratic in this space, identifying velocities up to equality almost everywhere along the training path. A proposed FGD scheme would update the velocity function:

$$
v_{k+1}=v_k-\alpha g_k,
\qquad
\|g_k-\nabla_{\mathcal V}\mathcal J(v_k)\|_{\mathcal V}
\le\epsilon\|g_k\|_{\mathcal V}.
$$

Here \(k\) indexes training updates; \(\tau\) remains generative time. Appendix A's descent calculation applies with smoothness constant \(2\). Under a uniform \(\epsilon<1\) and a suitable fixed step, it yields geometric reduction of population excess risk. Section 5 then connects that risk to generated-law error when the learned velocities also satisfy the required stability assumptions.

This is a mathematical opportunity, rather than an implemented FFM method. The population gradient contains the unknown conditional mean \(v^*\). A practical scheme needs a computable bound on gradient approximation and statistical error, an enrichable representation of maps on function-valued inputs, and control of the resulting velocity's regularity. A minibatch loss or agreement between two models does not supply that certificate. The quadratic functional geometry also does not make a nonlinear neural parameterization convex.

#### Using refinement to compute the transport

During sampling, we update a field by \(du_\tau/d\tau=v_\tau(u_\tau)\). This velocity need not decrease a scalar objective. The transferable idea is to make the representation accurate enough for the required update, with an error budget appropriate to the resulting law.

Let \(\widetilde v_m\) be the represented velocity at capacity \(m\). If a computable estimator bounds its defect against the intended reference velocity, one candidate rule is

$$
U_m(\tau,u)\ge
\|\widetilde v_m(\tau,u)-v_\tau(u)\|_H,
\qquad
U_m(\tau,u)\le
\epsilon_{\rm rel}\|\widetilde v_m(\tau,u)\|_H
+\epsilon_{\rm abs}.
$$

The absolute tolerance handles regions of nearly zero velocity. Representation error, learned-velocity error, and integration error must be separated; refining the representation cannot remove the other two.

Local velocity accuracy is only part of the calculation. With a reference Lipschitz envelope \(L\), define the remaining amplification

$$
A(\tau)=\exp\!\left(\int_\tau^1L(q)\,dq\right).
$$

Appendix D weights each dynamical defect and each representation-transfer jump by this factor. A useful refinement policy would allocate accuracy according to its contribution to endpoint error. It also needs a consistent law for newly added modes, as the two-mode example in Section 5 showed. Keeping a common fine-scale random seed is one possible design; the coarse dynamics must still account for unresolved modes.

#### Existing connections and a decisive first test

Adaptation and coarse-to-fine generation already appear in related work:

| Work | What it changes or establishes | Connection to the proposed transfer |
| --- | --- | --- |
| [Adaptive Flow Matching](https://proceedings.mlr.press/v267/fotiadis25a.html) | Encoded base distributions and validation-error-based noise scaling | Adjusts reference uncertainty during training |
| [Scale-Adaptive Generative Flows](https://arxiv.org/abs/2509.02971v2) | Source spectra and interpolation schedules | Controls drift regularity and numerical conditioning |
| [Scale-autoregressive modeling](https://arxiv.org/html/2604.11403v1#S3) | Samples finer field values conditional on coarser scales | Supplies an existing hierarchy of conditional generative models |
| [FFM discretization consistency](https://arxiv.org/html/2608.04531v1) | Convergence of finite conditional velocity targets, including nonnested reconstruction sequences | Explains why changing observations changes the target being learned |

In a targeted literature review as of October 1, 2026, I found no direct application of Csillag et al.'s adaptive FGD algorithm to FFM. That finding supports exploring a connection; it is not a proof of priority. The specific question is whether **computable update-error control can guide representation refinement while preserving the intended function law**.

Start with Gaussian or finite-prototype laws whose reference velocities are known. Compare a fixed fine representation, a prescribed coarse-to-fine hierarchy, and error-controlled refinement at matched endpoint accuracy. Measure physical Wasserstein error, derivative statistics, mode correlations, and total cost including error estimation, transfers, and model evaluations. Hold the source and interpolation path fixed to isolate representation adaptation. Test both independent and coupled modes before learning the velocity from data. A benefit would be lower cost at the same physical and distributional accuracy; fast samples alone would not establish it.

### Sources conditioned on physics and observations

A source covariance describes which functions are plausible before transport. Compatible spectra can keep a transport regular, but a marginal field covariance may be poorly suited to a conditional inverse problem. Boundaries, forcing, and measurements can change both the uncertain scales and their correlations.

This points toward context-dependent functional sources, constraint-preserving velocities, and transports informed by PDE stability. Compare white, scalar-normalized, and spectrally colored sources with the same architecture before attributing an additional benefit to the architecture. Evaluate the generated law alongside physical residuals and observables; accurate-looking samples do not determine uncertainty calibration.

#### Physical guidance and the intended distribution

An appealing shortcut is to add a physical correction to an already trained generative velocity:

$$
v_\tau^{\mathrm{guided}}(u)
=v_\tau(u)-\lambda_\tau\nabla_H\mathcal L(u).
$$

This changes the transport and generally changes its endpoint law. It is a new sampling model whose target must be justified or evaluated. A more deliberate connection would specify the desired law first, then design or learn a velocity for it. The open question is how physical information can help learn that transport without losing the uncertainty we intended to represent.

### Data requirements and cost at a prescribed accuracy

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

Couple the initial states and assume finite second moments and integrable defect norms. At deterministic refinement times \(\tau_j\), let \(J_j\) be the jump in the represented field. Define \(A(\tau)=\exp(\int_\tau^1L(q)dq)\). Applying the integrating-factor form of Grönwall between these times, and then Minkowski to the coupling, gives

$$
\begin{aligned}
W_{2,H}(\operatorname{Law}(u_1),\operatorname{Law}(\widetilde u_1))
\le{}&
A(0)\|u_0-\widetilde u_0\|_{L^2(\mathbb P;H)}\\
&+\int_0^1 A(\tau)\|r_\tau\|_{L^2(\mathbb P;H)}\,d\tau\\
&+\sum_j A(\tau_j)\|J_j\|_{L^2(\mathbb P;H)}.
\end{aligned}
$$

Since a Lipschitz envelope is nonnegative, \(A(\tau)\le A(0)\). A simpler uniform-amplification bound is therefore

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

The reference marginals equal the training-path marginals, so Appendix B identifies the integral with \(\mathcal E\). The coupling bounds endpoint Wasserstein distance. Adding approximation, smoothing, and numerical error by the triangle inequality yields Figure 14's budget.

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

Furuya et al.'s FOL approximation result concerns a fixed semilinear parabolic PDE and a varying initial function. In our notation, its assumptions include:

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
4. Wang, S., et al. [FunDiff: diffusion models over function spaces for physics-informed generative modeling](https://doi.org/10.1038/s41467-026-72292-0). Nature Communications 17, 5749, 2026. Figure 12 uses the 2025 preprint version.
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
29. Lino, M., and Thuerey, N. [One Scale at a Time: Scale-Autoregressive Modeling for Fluid Flow Distributions](https://arxiv.org/abs/2604.11403). arXiv:2604.11403, 2026.
30. Wang, S., Yu, X., and Perdikaris, P. [When and why PINNs fail to train: A neural tangent kernel perspective](https://arxiv.org/abs/2007.14527). arXiv:2007.14527, 2020.

## Cite this note

{{< cite-note >}}
