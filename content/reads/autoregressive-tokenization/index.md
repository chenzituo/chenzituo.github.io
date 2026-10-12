---
title: "Reading: Square Peg, Round Hole"
date: 2026-10-12
description: "Notes on tokenization, prediction order, and learning from non-sequential data."
summary: "A useful distinction for generative modeling: choosing the order of existing tokens versus learning a representation that makes successive predictions easier."
tags: [autoregressive-models, representation-learning, tokenization]
showtoc: false
ShowPostNavLinks: false
sourceTitle: "Square Peg, Round Hole: Plugging Non-Sequential Data into Sequential Language Models"
sourceURL: "https://iclr-blogposts.github.io/2026/blog/2026/autoregressive-tokenization/"
sourceAuthors: [Julia Balla, Hannah Lawrence]
sourcePublication: "ICLR Blogposts 2026"
sourceDate: 2026-04-27
---

**Original post:** [Square Peg, Round Hole: Plugging Non-Sequential Data into Sequential Language Models](https://iclr-blogposts.github.io/2026/blog/2026/autoregressive-tokenization/) by **Julia Balla and Hannah Lawrence**, ICLR Blogposts 2026, April 27, 2026.

## Summary

Images, graphs, and point clouds do not supply an obvious next token. An autoregressive model can still factor their distribution in any chosen order, but the resulting prediction problems may differ greatly in difficulty for a finite model.

The post separates two ways to address this:

- **Change the order:** retain the tokens while choosing, learning, or training across different generation orders.
- **Change the representation:** design tokens for sequential prediction, for example through coarse-to-fine structure or an autoregressive objective during tokenizer training.

Its central warning is that good reconstruction and compact encoding do not guarantee easy generation. The authors illustrate order sensitivity with a Markov-chain experiment and survey approaches that align representations with the generator. “Modelability” depends on the representation, model class, and computational constraints. [Read the original discussion and examples](https://iclr-blogposts.github.io/2026/blog/2026/autoregressive-tokenization/).

## Why I’m keeping it

My takeaway for [functional learning](/posts/functional-learning/) is a question to ask before choosing a larger model: **what prediction problem does the representation create?**

For a field, physical-time forecasting, spatial traversal, and refinement across scales are different choices. A coarse-to-fine field representation is one possible experiment, but preserving a field accurately does not tell us whether its next component is easy to predict.

The connection to RKHS flow is a research analogy: our lift specifies reconstruction and error geometry; it does not, by itself, make a conditional velocity easier to learn. I would keep the representation and the generative path as separate experimental factors. The post does not establish an advantage for our RKHS method or DynaBench forecasts.

## Question to revisit

At matched reconstruction error and training compute, does a multiscale field representation improve held-out forecasting over the current observation-space representation? Measure rollout error and generation cost together, so compression alone cannot stand in for a forecasting gain.
