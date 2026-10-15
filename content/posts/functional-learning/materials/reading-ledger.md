# Focused source checks

## 2026-10-15 — Figure 14: operator depth versus physical time

This entry records a targeted correction, not a full re-audit of the papers.

| Source and inspected coverage | Supported claim | Scope |
| --- | --- | --- |
| Furuya et al., *Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations*, Section 4.1, PDF pp. 8–9; [conference PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4b6ccf3acd6ccbc1093e093df345ba2-Paper-Conference.pdf) | The proof composes approximate Picard blocks to approximate a solution operator. Each block updates a whole trajectory; more blocks add network depth. | One Picard block can contain multiple neural layers. The PDE-specific contraction construction is not a convergence theorem for arbitrary trained operators. |
| Calvello et al., *Continuum Attention for Neural Operators*, Section 4.1, Eq. (10), and Section 4.2, Fig. 1, Eqs. (14)–(19), PDF pp. 14–16; [JMLR PDF](https://www.jmlr.org/papers/volume26/24-0879/24-0879.pdf) | Encoder layers update feature functions via attention, normalization and feedforward operations; input/output maps surround the layer composition. | Intermediate features need not be physical solutions. The layer index is discrete and is not physical time. |

Figure 14 and the adjacent text now distinguish optimization time, network-layer depth, and generative time. The coordinate panel separates discrete layer composition from continuous velocity integration. The following Neural Galerkin example still explicitly describes physical-time evolution. No benchmark or new convergence result is claimed.

## 2026-10-15 — RKHS Flow chapter organization from the architecture note

Reading mode: focused comparison of the full 11-page author-supplied `FGD_architecture_and_field_denoising.pdf` with Section 6. All page text was extracted and read; all pages were visually reviewed, with full-size inspection of the four reused figure pages (3, 4, 7, 8). No extraction gap was found. This is a document review, not an independent code, checkpoint, or benchmark audit.

Source identity: *FGD: a field model with denoising dynamics* / *Architecture & worked trajectories*, dated 15 October 2026 in the document. The supplied local PDF has SHA-256 `655a33b05c462e5a836b3cc0b5d612baeb745c714bd4b32c4452c680f407d39c`. No public source URL or explicit external license was supplied. Selected figures from this author-provided research note are used for the requested blog explanation; the complete PDF is not redistributed. The extraction script requires the local input PDF.

| Claim or distinction | Inspected evidence | Status and limit |
| --- | --- | --- |
| Shared construction separates field geometry, learned velocity, and conditional generation | pp. 1–3 | Verified as the note's organizing argument; adopted as the chapter structure. |
| DynaBench uses six width-128 Point Transformer blocks, nine 900-site frames, whitening plus raw-value input, and per-frame noise embedding | p. 2 | Reported implementation details; underlying source not independently inspected in this revision. |
| Ocean uses a 12-layer width-384 DiT3D and a separate recurrent guidance updater | p. 2 | Reported architecture. Different weights and conditioning mechanism from DynaBench. |
| Cloud readings are local RBF observations of a finite Fourier field; the lift is re-observation-consistent | p. 3 | Source reports maximum error 6.89e-14. New-query evaluations are not off-sensor ground truth. |
| DynaBench fixes eight clean history frames and updates the ninth through 50 deterministic DDIM steps | p. 4 | Reported saved sampler trajectory. The clean/source endpoint identities are also algebraically checked from alpha² + sigma² = 1. |
| Physical rollout uses each ensemble member's own generated history | p. 5 | Reported mechanism. No aggregate accuracy or uncertainty-calibration claim inferred. |
| Ocean alternates denoising, covariance-based observation correction, recurrent correction, and fresh renoising | pp. 6–9 | Source and guidance covariances are distinct; correction is a soft solve. Within-iteration differences are not independent ablations. |
| Fifteen physical days are generated jointly; the displayed endpoint is a window before boundary repair and overlap aggregation | p. 10 | Single-window scope, not full benchmark output. |
| Replay provenance and tolerances | p. 11 | Read as reported historical checks. They were not rerun, and their numeric replay tolerances were not promoted to new validation results. |

### Editorial decisions

- Main sequence: representation → velocity learning and DynaBench sampling → conditioning via Ocean measurements → exact representation guarantees → source/coupling/interpolation choices.
- Preserve the existing two-mode and transport toys, their scripts, mathematical claims, and appendix derivations. Put them after the mechanism they analyze.
- Distinguish layer depth, denoising index, and physical time. The velocity target is an angular derivative, not a derivative with respect to the integer noise index.
- Use B for the covariance square root, S_X for the field lift, M for the Ocean measurement operator, and C_guide for guidance covariance. The source PDF's S and H therefore need translation when comparing equations.
- Keep the chapter name RKHS Flow: the supplied note's FGD label would collide with functional gradient descent earlier in the blog.
- Figures 16–19 are source excerpts; the existing original toys move to Figures 20–24. Updated all article and appendix figure references.

### Visual study

- Page 3 repeats the same field through three readout operations → makes representation tangible → use it immediately after defining the lift.
- Page 4 holds physical history fixed while the candidate forecast changes → separates conditioning from generated state → show it beside the DDIM update.
- Pages 7–8 align stages and use shared color scales; page 8 fixes the physical date → exposes what each loop operation changes → retain panel labels, all selected panels, and color bars, with explicit reading directions in the caption.

Remaining research decision: evaluate representation/source/loss choices with matched backbones and budgets, and test forecast skill or reconstruction accuracy on full support. The note's worked examples motivate this protocol; they do not resolve those comparisons.
