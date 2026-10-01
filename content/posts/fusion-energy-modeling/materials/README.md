# Fusion energy modeling blog plan

Logged on 2026-09-30. Working title: **Modeling fusion energy: plasma physics, neural networks, and AI**.

The central question is which parts of fusion development better models can accelerate. Suggested thesis: AI's contribution should be judged by the physical decisions it improves and how reliably those improvements transfer from simulations to experiments. This is an editorial direction, not a conclusion established by the sources below.

## Proposed outline

1. Why fusion needs multiple physical models: confinement, stability, transport, and engineering constraints.
2. What MHD captures and where its fluid approximations break down.
3. Where neural networks enter: fast surrogates, diagnostic inference, and feedback control.
4. How different fusion companies use modeling, with evidence from the cases below.
5. Validation, uncertainty, transfer between devices, and extrapolation beyond training conditions.

## Physical modeling and AI

MHD describes conducting-fluid motion coupled to magnetic fields. Helion reports using it for field-reversed-configuration formation, translation, merging, and circuit coupling, alongside hybrid-kinetic models that capture additional particle physics. This gives a concrete example of why multiple physical descriptions are needed. [Helion modeling overview](https://www.helionenergy.com/blog/from-code-to-compression-how-simulation-accelerates-fusion-engineering).

Neural-network transport models such as QLKNN accelerate predictions of heat and particle fluxes. TORAX provides a differentiable tokamak transport simulator for integrated modeling, optimization, and control development. Keep transport simulation distinct from a full three-dimensional MHD or kinetic model. [QLKNN](https://arxiv.org/abs/1911.05617), [TORAX](https://arxiv.org/abs/2406.06718).

Two experimental anchors are deep reinforcement learning for magnetic shape control on TCV and reinforcement learning for avoiding tearing instability on DIII-D. Describe the demonstrated tasks and devices precisely; these results alone do not establish commercial power-plant readiness. [TCV control, Nature 2022](https://www.nature.com/articles/s41586-021-04301-9), [DIII-D instability avoidance, Nature 2024](https://www.nature.com/articles/s41586-024-07024-9).

## Company cases and sources

| Company | Focus | Evidence and scope |
| --- | --- | --- |
| Commonwealth Fusion Systems | Tokamak operating scenarios and control | [DeepMind collaboration](https://deepmind.google/blog/bringing-ai-to-the-next-generation-of-fusion-energy/): TORAX simulation, AI optimization, and reinforcement-learning research. Separate announced aims from experimental demonstrations. |
| TAE Technologies | Experimental optimization | [Google collaboration](https://tae.com/tri-alpha-energy-and-google-combine-human-and-machine-interaction-to-further-plasma-science/): human-guided machine learning for plasma experiments. Useful for explaining that AI is broader than neural networks. |
| Proxima Fusion | Stellarator design | [AI and expert knowledge](https://proximafusion.com/press-news/the-fusion-of-expert-knowledge-and-machine-learning-ai-in-action-at-proxima): surrogate modeling across simulation fidelities and design optimization. |
| Helion | Pulsed plasma dynamics | [Simulation overview](https://www.helionenergy.com/blog/from-code-to-compression-how-simulation-accelerates-fusion-engineering): MHD and hybrid-kinetic models. This source establishes simulation use, not a neural-network deployment. |
| Zap Energy | Sheared-flow-stabilized Z-pinch | [Whole Device Modeling of the FuZE Sheared-Flow-Stabilized Z Pinch](https://arxiv.org/abs/2401.10366): resistive MHD and comparisons of synthetic diagnostics with experiments. |

## Questions for the next research pass

- Which state variables, scales, and instabilities require MHD, extended fluid models, or kinetic descriptions?
- For each AI example, what is learned, what supplies the training data, and what remains physics-based?
- How are uncertainty, conservation, and out-of-distribution behavior evaluated?
- What experimental evidence supports each company's modeling claims?
- How should the article connect to the existing functional learning and reinforcement learning notes?

This is an initial source-backed outline, not a full literature review. Sources were checked for the initial topic discussion; detailed methods, benchmark comparisons, and current company milestones need verification when drafting. Papers are linked here; PDFs have not been collected for this entry.
