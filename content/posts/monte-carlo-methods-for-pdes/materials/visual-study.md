# Visual and structural revision — 2026-10-12

The initial post used six self-created figures and presented a paper survey before an insufficiently connected KV-cache outlook. The revision organizes the same research around repeated physical solves: response representation → compatible composition → reuse of persistent neural context.

| Inspected source | Observed design choice | Use in this revision |
|---|---|---|
| Guiding, Fig. 1, PDF p.1 | Physical query, learned proposal, and resulting comparison share one path | Reproduce the source figure; explain the network's actual learned object and distinguish it from a reusable exit law |
| WoDS, Fig. 5, p.5 | Same domain is shown whole, partitioned, then as a boundary close-up | Reproduce; explain why artificial interfaces carry unknown values |
| WoDS, Fig. 8, p.8 | Query/boundary locations and colors distinguish row and column interpretations | Reproduce in place of our disk-kernel plot; use concrete operator glyphs in the original cache proposal |
| Neural shapes, Fig. 2, p.4 | Representation changes are visible as intermediate geometries and local coordinate frames | Reproduce; make geometry → response tokens → K/V visible in the new proposal |
| LatentDDM, Fig. 1, p.4 | Aligned computational paths expose tokenization choices and repeated operator calls | Replace our generic composition schematic; identify prediction dependence as the cache-design constraint |
| WoDS, Fig. 3, p.3 | Same viewpoint across local scene edits exposes global path changes | Replace our generic invalidation grid; distinguish numerical local reuse from neural dependency invalidation |

## What remains original, and why

- The two-material conduction plot establishes our own compatibility counterexample; no source figure makes this exact argument.
- `cache-architecture-v2.svg` explains an architecture we propose, not one claimed in the cited papers. It uses concrete geometric patches, response arrays, K/V arrays and two illustrative fields; blue denotes persistent context and magenta the dynamic query path. It is independently drawn.
- The disk demo keeps the basic first-exit mechanism inspectable by changing the query and sample count.

All extracted figures were inspected as full source pages before cropping, then inspected individually after rendering. Crops omit surrounding prose/captions, preserve figure content, and are linked to full resolution. Rights notes distinguish CC BY sources from limited attributed figure excerpts; arXiv's distribution license is not represented as a reuse grant.

## Clarity check

The narrative now states what is invariant before mentioning caching; derives a second boundary-condition response (h = 0.2 for reversed endpoint values); distinguishes physical operators from neural K/V; explains why current LatentDDM prediction tokens change; and compares reuse against numerical factorization reuse. The cost model includes preparation, warm solves and invalidation. Physical correctness and cached/recomputed model equivalence are separate tests. No implementation or timing result is claimed.

## Sequential-modeling refinement

The author's next-work objective is sequential composition, not cache optimization in isolation. `sequential-composition.svg` repeats a marked regional geometry across a history, including a revisit, then separates the causal composer, local operator, updated field and newly appended event. Its history is immutable while the current field is revisable. The earlier static-only architecture is retained as a development asset but no longer displayed. This is original proposed machinery, so a source-paper figure cannot replace it without misattribution.
