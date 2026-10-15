# Focused source checks

## 2026-10-15 — Figure 14: operator depth versus physical time

This entry records a targeted correction, not a full re-audit of the papers.

| Source and inspected coverage | Supported claim | Scope |
| --- | --- | --- |
| Furuya et al., *Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations*, Section 4.1, PDF pp. 8–9; [conference PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf) | The proof composes approximate Picard blocks to approximate a solution operator. Each block updates a whole trajectory; more blocks add network depth. | One Picard block can contain multiple neural layers. The PDE-specific contraction construction is not a convergence theorem for arbitrary trained operators. |
| Calvello et al., *Continuum Attention for Neural Operators*, Section 4.1, Eq. (10), and Section 4.2, Fig. 1, Eqs. (14)–(19), PDF pp. 14–16; [JMLR PDF](https://www.jmlr.org/papers/volume26/24-0879/24-0879.pdf) | Encoder layers update feature functions via attention, normalization and feedforward operations; input/output maps surround the layer composition. | Intermediate features need not be physical solutions. The layer index is discrete and is not physical time. |

Figure 14 and the adjacent text now distinguish optimization time, network-layer depth, and generative time. The coordinate panel separates discrete layer composition from continuous velocity integration. The following Neural Galerkin example still explicitly describes physical-time evolution. No benchmark or new convergence result is claimed.
