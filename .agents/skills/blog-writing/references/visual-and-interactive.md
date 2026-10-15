# Figures and interactive examples

A visual should explain an object, relation or comparison that the adjacent prose needs. Give the reader its interpretation and evidence scope in the caption.

## Choose and read source figures first

For a paper-led explanation, inspect the actual figures before planning new artwork. Choose a panel that makes the adjacent argument easier to understand; prefer a method diagram over a decorative teaser unless the physical scene itself matters. Do not replace a useful source diagram with a generic row of boxes. Avoid a gallery with no reading guidance.

Record source title/authors, exact version, figure number, PDF page, URL, crop coordinates and any modifications in the post's figure manifest. Verify the source identity before extraction. Prefer source assets or high-resolution PDF rendering to screenshots. Crop surrounding prose while preserving the full selected panel, notation, axes, legends and color scales; identify partial panels. Link dense images to full-resolution assets. Write a new caption that tells readers what to look at and why it matters, then credits the source and names the crop/adaptation. Source results remain the authors' results.

Check the actual reuse terms; open access and arXiv hosting do not themselves establish a permissive image license. Include the license link when applicable; do not invent permission. Keep any non-permissively licensed excerpt limited to the figure needed for focused scholarly discussion and record that basis distinctly from an open license. Do not redistribute full papers as site assets by default.

## Learn visual design from the articles

Save a short, paper-specific visual study in the post's materials: **observed design choice → explanatory benefit → change made here**. Carry reusable lessons into this reference, rather than adopting a single paper's style everywhere.

Lessons from the Walk on Anything revision:

- WoDS Figs. 5/8 draw actual boundaries, query points, and operator entries. For an abstract concept, show the mathematical/physical object it acts on. Keep a stable color meaning and repeat the same object across transformations.
- Neural Shape Representations Fig. 2 shows the intermediate patches and coordinate frames. Make representation changes visible; label operations along the path instead of hiding them in large text boxes.
- LatentDDM Fig. 1 aligns comparable inputs/outputs and makes repeated local-operator calls visible. Use aligned comparison paths when the difference in computation is the point.
- WoDS Fig. 3 changes geometry within the same scene. Hold the view fixed when explaining the effect of an edit; separate local recomputation from its global physical consequences.

From the field-denoising architecture note: hold the physical date fixed when rows show sampler progress, and align columns with the actual operations within one iteration. State whether a comparison is within one run or between independently evaluated methods; a correction-stage image is not an ablation.

For original figures, use these explanatory choices, not copied artwork or unexplained borrowed notation. White space, restrained semantic colors, short direct labels and concrete field/mesh/operator glyphs often work better than decorative cards. Inspect the result at article width. A physically themed illustration must not look like measured data unless it is.

## Original figures and interactive examples

- Scientific figures: use NumPy/Matplotlib or another standard plotting tool. Save the seed, sampling budget, numerical tolerance, uncertainty definition and result file. Do not present toy simulations as paper replications or forecast benchmarks.
- Mechanism/architecture diagrams: use editable SVG or code-generated plots with readable labels and arrows. Compare methods only at the level their assumptions permit.
- Build figures at intended article width; inspect actual rendering. Avoid long display-math lines: aligned rows and prose definitions read better than a horizontal scroll for core equations. In this Hugo Markdown pipeline use `\lt` rather than a bare `<` next to letters inside math, which can be interpreted as HTML; check rendered math, not just the build. For SVGs, preserve typography with supported font fallbacks or outlined glyphs and retain the editable generator.
- Interactive examples suit a small changeable mechanism. Store post-local demo source as `demos/*.html.txt`; the shortcode emits it as an HTML resource. Hugo 0.167 blocks raw HTML content pages by default, so keep the existing security configuration. Use this asset and the `local-demo` shortcode. Its iframe is sandboxed with scripts only. Keep controls labelled, keyboard operable, outputs accessible, and layout usable at mobile widths. Prefer no external dependencies.
- Provide a static explanation/figure alongside an interaction. Report truncation, numerical error and sample uncertainty honestly. A cap should not silently turn an unfinished simulation into an estimate.
- Exercise the actual controls: initial state, a changed parameter, rerun and reset. Verify the intended behavior, not just the presence of a button. Reset should reproduce a fixed-seed initial result when determinism is part of the design.
- Inspect figures individually and in their surrounding article. Check fonts, clipping, contrast, loaded assets and mathematical labels. Optional contact sheets help review a figure set but do not replace inspecting dense panels.

Examples: `content/posts/monte-carlo-methods-for-pdes/materials/visuals.py` and its `demos/walk-disk.html.txt`; `content/posts/functional-learning/materials/toy-rkhs-coupling/` for a controlled numerical comparison with uncertainty and caveats.
