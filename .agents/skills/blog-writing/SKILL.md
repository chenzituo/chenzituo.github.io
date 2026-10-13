---
name: blog-writing
description: Develop research-led, visually rich articles and attributed reading notes for this Hugo blog, with reproducible figures, source coverage, and rendering checks. Use for substantive blog development in this repository.
---

# Research-led visual blog writing

Work relative to the blog repository root. This skill is local to the blog; it does not configure other writing projects.

## Local conventions

- Hugo with PaperMod; navigation in `hugo.yaml`. Preserve established URLs. Existing placeholders are usually the starting point for a new topic.
- Articles are leaf bundles: `content/posts/<slug>/index.md`, `figures/`, and `materials/`. Math uses `math: true`, `\(...\)` inline and `$$...$$` displays. `UseHugoToc: true` works with numbered headings.
- `content/reads/` holds summaries of others' posts: source title, authors, original URL/date, a concise attributed summary, then the author's own takeaway. Reads has its own listing/RSS; `mainSections: [posts]` keeps it separate from Posts/Archive while search includes it.
- Functional Learning is an example of sustained technical exposition with reproducible toys. Walk on Anything demonstrates a developing visual argument and an isolated interactive demo. Use these as patterns, not text templates or mandatory lengths.

## Build the argument and the evidence together

Organize a synthesis around one reader question and a running example. Let each section resolve the question needed by the next; assign papers roles in that argument instead of giving each paper a disconnected mini-summary. Before adding an outlook, write the missing bridge explicitly: what object, operation, or dependency carries over, and what does not?

Start with the reader's concrete problem, then develop the mechanism and its implications. Name what is learned, what is solved, what is represented, and what is assumed. Distinguish a paper's demonstrated scope from the author's research extrapolation.

For supplied papers, verify identity and requested versions. Record inspected pages/sections, claims, extraction gaps, and unresolved comparisons in `materials/reading-ledger.md`; record exact URLs and useful provenance in `sources.json`. Use the available research-paper-reading skill for substantial paper analysis. Keep quotes short, respect source-specific summarization limits, and write original derivations/examples where useful. Do not use abstract-level reading to imply a full theorem or benchmark audit.

The main text should explain why each equation matters. Define symbols before use. Separate distinct times (physical, diffusion, optimization), norms, and conditional distributions. A theorem's assumptions, a model's expressivity and its measured performance are different evidence levels.

## Visual exposition

The author's preference is visually ample technical posts. Distribute figures where they answer the next question in the argument, rather than collecting decoration at the end. Useful roles include mechanism, worked example, controlled comparison, architecture, and failure mode. Choose the number from the argument, not a quota.

Use [visual-and-interactive.md](references/visual-and-interactive.md) when creating figures or demos. For explanations of published methods, prefer carefully selected figures from the articles over newly drawn substitutes. Read the figure and caption in context; retain attribution and record the reuse basis. Reserve original diagrams and reproducible plots for the author's new synthesis, derivation, or experiment. Study the source figures to improve the design of those original visuals. Keep plot generators and numerical results in materials. Store a `figure-sources.json` mapping each figure to its generator, source/derivation, and evidence status.

## Finish the requested stage

“Start developing” normally means a reviewable local draft, not an automatic public release. Honor explicit publication requests and earlier authorization in the session; do not invent a new permission gate. State whether the result is local, committed, pushed, or live.

Read [build-and-publish.md](references/build-and-publish.md) for site verification and publication. Preserve unrelated dirty files. Stop preview servers started for checks before finishing unless the user requests a running preview.

## Improve the skill from actual work

Add compact practices that improve future blog work, especially repeated numerical or rendering failures. Put reusable procedures here and paper-specific hypotheses in the post. Avoid a growing list of universal prohibitions or duplicating the agent's general instructions.
