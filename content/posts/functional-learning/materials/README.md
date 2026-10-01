# Functional learning literature for neural PDE models

A focused reading collection for the Functional Learning blog entry, assembled on 2026-09-30. These papers cover optimization over functions, latent representations of continuous fields, operators between function spaces, and generative models of physical functions. The suggested uses below are editorial recommendations, not claims made by the papers.

Start with Functional Gradient Descent, Autoencoders in Function Space, and FunDiff; then read CORAL, DINo, and AROMA for concrete latent PDE models. Functional Flow Matching provides the direct function-space flow formulation; GeoFunFlow connects the topic to inverse problems. Neural Operator, FNO, and SIREN provide foundations.

All 11 PDFs are saved in [papers](papers/). Citation entries are in [references.bib](references.bib), and file provenance is in [sources.json](sources.json). Coverage is limited to abstracts, publication metadata, and PDF first-page identity checks; full methods, proofs, and experiments still need a deeper read. First-page text extraction is clean for all files; matches are high confidence.

## Function space optimization

### Functional Gradient Descent with Adaptive Representations

Csillag, Daniel; Schuller, Rodrigo; Dall'Antonia, Pedro; Guibas, Leonidas; Velho, Luiz; Novello, Tiago. **arXiv preprint; first submission 2026.**

Adapts the representation of functional gradients during optimization and includes numerical PDE experiments. Suggested use: explain why choosing a function space and controlling approximation error changes the optimization problem. This is a function-space optimization reference, not a latent neural PDE architecture.

[Source](https://arxiv.org/abs/2606.16926) · [Local PDF](papers/functional-gradient-descent.pdf) · Identifier: `2606.16926`.

## Latent neural fields for PDEs

### Operator Learning with Neural Fields: Tackling PDEs on General Geometries

Serrano, Louis; Le Boudec, Lise; Kassaï Koupaï, Armand; Wang, Thomas X; Yin, Yuan; Vittaut, Jean-Noël; Gallinari, Patrick. **arXiv preprint; first submission 2023.**

Uses coordinate-based neural fields for operator learning on general geometries, including PDE prediction and inverse design. Suggested use: discuss how a latent code can represent a continuous field across different spatial samplings.

[Source](https://arxiv.org/abs/2306.07266) · [Local PDF](papers/coral.pdf) · Identifier: `2306.07266`.

### Continuous PDE Dynamics Forecasting with Implicit Neural Representations

Yin, Yuan; Kirchmeyer, Matthieu; Franceschi, Jean-Yves; Rakotomamonjy, Alain; Gallinari, Patrick. **ICLR 2023.**

Embeds spatial observations through implicit neural representations and evolves their latent states with a learned ODE. Suggested use: separate continuous spatial representation from continuous-time latent dynamics.

[Source](https://arxiv.org/abs/2209.14855) · [Local PDF](papers/dino.pdf) · Identifier: `2209.14855`.

### AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields

Serrano, Louis; Wang, Thomas X; Le Naour, Etienne; Vittaut, Jean-Noël; Gallinari, Patrick. **arXiv preprint; first submission 2024.**

Uses local neural fields and spatially structured latent representations for PDE dynamics, with a diffusion-based temporal formulation. Suggested use: compare a global latent vector with representations that retain spatial organization.

[Source](https://arxiv.org/abs/2406.02176) · [Local PDF](papers/aroma.pdf) · Identifier: `2406.02176`.

## Generative models of physical functions

### Functional Flow Matching

Kerrigan, Gavin; Migliorini, Giosue; Smyth, Padhraic. **AISTATS 2024, PMLR 238, 3934–3942; preprint first posted in 2023.**

Defines flow matching through probability measures and learned vector fields on a function space. The introduction describes experiments on time series and a two-dimensional Navier–Stokes dataset. Suggested use: introduce direct function-space generative dynamics before discussing latent function autoencoders and GeoFunFlow. Its generative flow time should be distinguished from the physical time of a PDE solution.

[Source](https://proceedings.mlr.press/v238/kerrigan24a.html) · [Local PDF](papers/functional-flow-matching.pdf) · [Author code](https://github.com/GavinKerrigan/functional_flow_matching) · Identifier: `2305.17209`.

The saved Mendeley copy is arXiv v2 dated 5 December 2023; the bibliography follows the 2024 proceedings. Title and authors match, with clean first-page extraction; a deeper methods and results read remains pending.

Existing vault mention: [Meta-Operator Flow Map Proposal](</Users/zituochen/Documents/Obsidian Vault/Rebuttal/Direct Measurement to Field/proposal/Meta-Operator Flow Map Proposal.md>).

### FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling

Wang, Sifan; Dou, Zehao; Shan, Siming; Liu, Tong-Rui; Lu, Lu. **Nature Communications 17, 5749 (2026); preprint first posted in 2025.**

Combines a function autoencoder with latent diffusion and physical priors. Suggested use: motivate continuous, coordinate-queryable decoding of latent representations. The published paper explicitly describes generation of PDE-governed solution distributions rather than a neural PDE solver.

[Source](https://doi.org/10.1038/s41467-026-72292-0) · [Local PDF](papers/fundiff.pdf) · Identifier: `2506.07902`.

The saved PDF is the 2025 arXiv v2 preprint; the citation points to the 2026 published article. [Author code](https://github.com/sifanexisted/fundiff).

Existing vault mentions: [Sifan Wang](</Users/zituochen/Documents/Obsidian Vault/Crawler/NeuralOperators/Sifan Wang.md>), [ADAPTATION_NOTES](</Users/zituochen/Documents/Obsidian Vault/Rebuttal/Latent Generative Solver/ICML/supplement/baselines_generative/ADAPTATION_NOTES.md>).

### GeoFunFlow: Geometric Function Flow Matching for Inverse Operator Learning over Complex Geometries

Wang, Sifan; Wu, Zhikai; van Dijk, David; Lu, Lu. **arXiv preprint; first submission 2025.**

Combines a geometric function autoencoder with rectified-flow training for PDE inverse problems on complex geometries. Suggested use: connect function representations to sparse observations, unstructured meshes, and posterior sampling.

[Source](https://arxiv.org/abs/2509.24117) · [Local PDF](papers/geofunflow.pdf) · Identifier: `2509.24117`.

## Foundations and representation theory

### Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs

Kovachki, Nikola; Li, Zongyi; Liu, Burigede; Azizzadenesheli, Kamyar; Bhattacharya, Kaushik; Stuart, Andrew; Anandkumar, Anima. **JMLR 24(89), 1–97 (2023); local copy is arXiv v5.**

Develops neural operators as maps between function spaces, including discretization-invariant parameterizations and PDE solution operators. Suggested use: define operator learning and distinguish shared parameters across discretizations from a guarantee of identical numerical error.

[Source](https://jmlr.org/papers/v24/21-1524.html) · [Local PDF](papers/neural-operator.pdf) · Identifier: `2108.08481`.

Version note: the Mendeley filename says 2021 and its PDF header says 2022; the official journal record gives 2023. The bibliography follows the journal record.

### Fourier Neural Operator for Parametric Partial Differential Equations

Li, Zongyi; Kovachki, Nikola; Azizzadenesheli, Kamyar; Liu, Burigede; Bhattacharya, Kaushik; Stuart, Andrew; Anandkumar, Anima. **ICLR 2021.**

Parameterizes operator kernels in Fourier space and evaluates them on parametric PDE families. Suggested use: provide a concrete operator-learning baseline alongside latent neural-field methods.

[Source](https://arxiv.org/abs/2010.08895) · [Local PDF](papers/fourier-neural-operator.pdf) · Identifier: `2010.08895`.

Existing vault mentions: [DOWNLOAD](</Users/zituochen/Documents/Obsidian Vault/Rebuttal/Latent Generative Solver/ICML/supplement/preprocess/DOWNLOAD.md>), [Zongyi Li](</Users/zituochen/Documents/Obsidian Vault/Crawler/NeuralOperators/Zongyi Li.md>), [Literature Index](</Users/zituochen/Documents/Obsidian Vault/Funding/FM DreamBESS/papers/Neural Symbolic/Literature Index.md>).

### Implicit Neural Representations with Periodic Activation Functions

Sitzmann, Vincent; Martel, Julien N. P.; Bergman, Alexander W.; Lindell, David B.; Wetzstein, Gordon. **arXiv preprint; first submission 2020.**

Introduces periodic-activation neural representations of continuous signals and derivatives, with PDE boundary-value examples. Suggested use: explain why the decoder and its derivatives matter for representing physical solution fields.

[Source](https://arxiv.org/abs/2006.09661) · [Local PDF](papers/siren.pdf) · Identifier: `2006.09661`.

### Autoencoders in Function Space

Bunker, Justin; Girolami, Mark; Lambley, Hefin; Stuart, Andrew M.; Sullivan, T. J.. **JMLR 26(165), 1–54 (2025).**

Defines and studies function-space autoencoders and variational autoencoders, including when their objectives are well-defined. Suggested use: give the mathematical basis for encoding functions into finite-dimensional latent variables and decoding across meshes.

[Source](https://www.jmlr.org/papers/v26/25-0035.html) · [Local PDF](papers/autoencoders-in-function-space.pdf) · Identifier: `jmlr25-0035`.

## Reading questions for the blog

- What is being learned: a solution function, a map between functions, a latent evolution law, or a distribution over functions?
- Which parts depend on the observation mesh, and which can be evaluated at new coordinates?
- What function-space norm or loss defines accuracy, including derivative accuracy?
- Are physical constraints built into the architecture or encouraged through a loss?
- Which benefits come from representation, and which come from optimization or additional training data?

## Local source notes

The Mendeley filename search found existing copies of Neural Operator, FNO, SIREN, and Functional Flow Matching, which were copied here. The other seven PDFs were obtained from arXiv or JMLR; no matching Mendeley filenames were found in the targeted search. This does not exclude differently named copies. Exact source paths, download URLs, checksums, and identifier-based vault matches are recorded in sources.json. An empty vault-match list means the identifier search found no match, not that the paper is absent from the vault.

Additional FunDiff context: [Sifan Wang literature note](</Users/zituochen/Documents/Obsidian Vault/Crawler/NeuralOperators/Sifan Wang.md>) and [Latent Generative Solver experiment plan](</Users/zituochen/Documents/Obsidian Vault/Rebuttal/Latent Generative Solver/ICML/experiment_plan.md>).
