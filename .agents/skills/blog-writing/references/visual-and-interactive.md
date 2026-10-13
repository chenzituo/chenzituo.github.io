# Figures and interactive examples

A visual should explain an object, relation or comparison that the adjacent prose needs. Give the reader its interpretation and evidence scope in the caption.

- Scientific figures: use NumPy/Matplotlib or another standard plotting tool. Save the seed, sampling budget, numerical tolerance, uncertainty definition and result file. Do not present toy simulations as paper replications or forecast benchmarks.
- Mechanism/architecture diagrams: use editable SVG or code-generated plots with readable labels and arrows. Compare methods only at the level their assumptions permit.
- Build figures at intended article width; inspect actual rendering. Avoid long display-math lines: aligned rows and prose definitions read better than a horizontal scroll for core equations.
- Interactive examples suit a small changeable mechanism. Store post-local demo source as `demos/*.html.txt`; the shortcode emits it as an HTML resource. Hugo 0.167 blocks raw HTML content pages by default, so keep the existing security configuration. Use this asset and the `local-demo` shortcode. Its iframe is sandboxed with scripts only. Keep controls labelled, keyboard operable, outputs accessible, and layout usable at mobile widths. Prefer no external dependencies.
- Provide a static explanation/figure alongside an interaction. Report truncation, numerical error and sample uncertainty honestly. A cap should not silently turn an unfinished simulation into an estimate.
- Exercise the actual controls: initial state, a changed parameter, rerun and reset. Verify the intended behavior, not just the presence of a button. Reset should reproduce a fixed-seed initial result when determinism is part of the design.
- Inspect figures individually and in their surrounding article. Check fonts, clipping, contrast, loaded assets and mathematical labels. Optional contact sheets help review a figure set but do not replace inspecting dense panels.

Examples: `content/posts/monte-carlo-methods-for-pdes/materials/visuals.py` and its `demos/walk-disk.html.txt`; `content/posts/functional-learning/materials/toy-rkhs-coupling/` for a controlled numerical comparison with uncertainty and caveats.
