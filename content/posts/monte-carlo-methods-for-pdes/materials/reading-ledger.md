# Walk on Anything — reading and development ledger

Updated 2026-10-12. Mode: targeted comparison and theory development. No production code audit, benchmark reproduction, or foundation-model implementation.

## Source identity and coverage

Exact URLs, versions, PDF hashes and verified local filenames are in `sources.json`. The Mendeley lookup verified the first page of three local PDFs; the neural-shape filename says “Anonymous” but its contents identify the named v2 authors. Remote PDFs were downloaded independently to ensure requested version identity. No new full-text copies are committed here.

| Source | Inspected locations | Role / status |
|---|---|---|
| Huang et al., arXiv:2410.18944v3 | Abstract; §3–4 pp.3–5; §7 p.8 | Proposal learning; verified method, no benchmark audit |
| WoDS, DOI 10.1145/3811340 | Introduction pp.2–3; §§3–5 pp.5–10; §8 operator-representation limitations p.16; p.9 rendered | Interface construction and Eq.(11) verified; experiment suite unread |
| Welschinger et al., arXiv:2512.21311v2 | Abstract; §3 pp.3–5 (targeted pipeline passages); limitations p.8 | Surface extension / grid update; not a random-walk method |
| Gupta et al., arXiv:2606.14756v1 | Abstract; §3.1–3.3 pp.3–4 | Allocation/composition connection; proofs and benchmark tables unread |
| Chen, Zhang, Deng, arXiv:2609.03069v1 | §3 pp.3–5; selected §4 passages pp.7–8; Appendix C.1–C.2 p.16 | Two local evaluations, input correction, and coverage/selection bound verified; no independent performance validation |
| Balla and Lawrence, ICLR Blogposts 2026 | Main narrative read during Reads entry; carried forward here | Representation/order motivation; not a physical-field theorem |

Extraction note: WoDS contains large whitespace and embedded LaTeXiT strings. A second extraction removed those metadata strings; p.9 was visually checked for the matrix equation and absorption condition. Other math passages were read in extracted text; no exhaustive rendering audit is claimed.

## Claim provenance

| Draft claim | Evidence / derivation | Status |
|---|---|---|
| Harmonic first-exit representation | Classical background, also WoDS §4.1–4.2 | Restricted to stated assumptions |
| Guided proposal is distinct from a learned solution | Huang §4; importance-sampling identity | Verified / algebraic explanation |
| Global discrete solve requires eventual absorption | WoDS §5.1, Eq.(11)–(12), p.9 | Verified |
| “Deterministic solve” does not remove operator-estimation error | Conditional versus unconditional randomness | Our interpretation; do not call full pipeline variance-free |
| Two local fields can match values and violate flux conservation | `interface_example.py`, analytic piecewise conductivity | Independently derived and checked |
| Local representability and composition selection are different | LatentDDM §3.2, Appendix C.1 | Conditional bound, not a universal transfer theorem |
| A local field token could encode a response family | Synthesis across these sources | Research proposal |
| A static encoder could supply reusable K/V | Explicit dependency structure in draft §6 | Proposed architecture; not established for current LatentDDM |
| Cache does not eliminate dense attention interactions | Matrix dimensions: M queries × N keys | Complexity calculation, no timing measurements |

## What to develop next

1. Specify the physical token contract: fields versus response operators, exterior conditions, geometry, coordinate transforms, units, coefficient ranges and decoder operations.
2. Start with an analytic elliptic response task, then a numerical reference with varied boundary conditions. Compare token families at matched storage and required physical accuracy.
3. Audit the actual LatentDDM encoder before proposing modifications: which representations depend on local predictions, dynamic conditions and global attention? This post does not certify existing cache reuse.
4. Build an uncached reference and a static-context cached variant. Measure equivalence, stale-cache controls, cold/warm latency, memory and invalidation after local edits.
5. Keep variable geometry, transient histories and nonlinear PDEs as distinct transfer tests. A foundation-model claim needs task diversity and systematic held-out physics/geometry evaluation beyond a two-case demonstration.

The supplied Divide-and-Denoise ID is intentional: the user confirmed its role as a broader composition connection. Fair allocation is not a replacement for physical compatibility.

## Scope of this update

The existing post URL is retained. Initial development was a local draft; the subsequent publication request on 2026-10-12 changes `index.md` to `draft: false`. The article remains a developing research note. All six figures are original, with generators/results saved here. Existing unrelated working-tree changes are untouched.

## Visual development

Six original figures plus an interactive disk example develop the argument. The 4,096-walk disk run used seed 20261012, epsilon 1e-5 and no cap hits; estimate 0.09692, exact 0.10000, SE 0.01101. These are pedagogical checks, not source-paper reproductions. Architecture and cache-invalidation figures are proposals/illustrations. Source map: `figure-sources.json`.

Validation: Hugo draft/production builds passed (existing PaperMod deprecation warnings only); 62 local HTML targets/fragments checked; all six figures visually inspected. Browser review exercised query 0.8 with 1,024 walks (estimate 0.6252, exact 0.6400), then verified deterministic Reset to the initial query/seed. These demo checks establish control behavior, not statistical coverage or solver convergence. Desktop article math and sandboxed iframe rendered. Mobile rendering remains a follow-up check.

Publication check (2026-10-12): the exact staged snapshot builds successfully with Hugo 0.167.0 in production mode, including the article and demo; 62 local targets/fragments pass. Only existing PaperMod deprecation warnings remain.
