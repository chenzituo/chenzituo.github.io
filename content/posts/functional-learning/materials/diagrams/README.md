# Original explanatory diagrams

[functional-learning-elements.tex](functional-learning-elements.tex) is the editable LaTeX/TikZ source for Figure 6. It summarizes the note's building elements and separates optimization, physical evolution, and generative transport. The website uses a vector SVG exported from the compiled PDF, with a white background and embedded glyph outlines for consistent rendering.

From the blog repository root:

```sh
mkdir -p /tmp/functional-learning-diagram
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/functional-learning-diagram content/posts/functional-learning/materials/diagrams/functional-learning-elements.tex
pdftocairo -svg /tmp/functional-learning-diagram/functional-learning-elements.pdf content/posts/functional-learning/figures/functional-learning-elements.svg
pdftoppm -scale-to 1600 -png -singlefile /tmp/functional-learning-diagram/functional-learning-elements.pdf /tmp/functional-learning-diagram/preview
```

Inspect the rendered PNG after changes and verify the SVG in the blog preview. The PDF, auxiliary files, log, and PNG are build intermediates outside the repository. Paper figures remain separate, unchanged extracts with source attribution.

[functional-theory-error-chain.tex](functional-theory-error-chain.tex) is the source for Figure 10. It traces representation, prototype, smoothing, velocity-learning, and numerical errors in one physical Wasserstein metric. It illustrates the derived bound; it is not a measured result.

From the blog repository root:

```sh
mkdir -p /tmp/functional-learning-theory-diagram
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/functional-learning-theory-diagram content/posts/functional-learning/materials/diagrams/functional-theory-error-chain.tex
pdftocairo -svg /tmp/functional-learning-theory-diagram/functional-theory-error-chain.pdf content/posts/functional-learning/figures/functional-theory-error-chain.svg
pdftoppm -scale-to 1600 -png -singlefile /tmp/functional-learning-theory-diagram/functional-theory-error-chain.pdf /tmp/functional-learning-theory-diagram/preview
```
