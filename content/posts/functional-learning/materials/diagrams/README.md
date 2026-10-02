# Original explanatory diagrams

[functional-learning-elements.tex](functional-learning-elements.tex) is the editable LaTeX/TikZ source for Figure 14. It summarizes the note's building elements and separates optimization, physical evolution, and generative transport. It explicitly shows a finite code whose dimension can grow, with a field-preserving transfer as the refinement design condition. The website uses a vector SVG exported from the compiled PDF, with a white background and embedded glyph outlines for consistent rendering.

From the blog repository root:

```sh
mkdir -p /tmp/functional-learning-diagram
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/functional-learning-diagram content/posts/functional-learning/materials/diagrams/functional-learning-elements.tex
pdftocairo -svg /tmp/functional-learning-diagram/functional-learning-elements.pdf content/posts/functional-learning/figures/functional-learning-elements.svg
pdftoppm -scale-to 1600 -png -singlefile /tmp/functional-learning-diagram/functional-learning-elements.pdf /tmp/functional-learning-diagram/preview
```

Inspect the rendered PNG after changes and verify the SVG in the blog preview. The PDF, auxiliary files, log, and PNG are build intermediates outside the repository. Paper figures remain separate, unchanged extracts with source attribution.

[functional-theory-error-chain.tex](functional-theory-error-chain.tex) is the source for Figure 17 in the [previous long version](https://github.com/chenzituo/chenzituo.github.io/blob/c874949743fd709c45dede636673df473eec6179/content/posts/functional-learning/index.md). A local copy is preserved at materials/archive/functional-learning-long-version.md. The shorter public post omits this extended derivation. The diagram traces representation, prototype, smoothing, velocity-learning, and numerical errors in one physical Wasserstein metric. It illustrates the derived bound; it is not a measured result.

From the blog repository root:

```sh
mkdir -p /tmp/functional-learning-theory-diagram
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/functional-learning-theory-diagram content/posts/functional-learning/materials/diagrams/functional-theory-error-chain.tex
pdftocairo -svg /tmp/functional-learning-theory-diagram/functional-theory-error-chain.pdf content/posts/functional-learning/figures/functional-theory-error-chain.svg
pdftoppm -scale-to 1600 -png -singlefile /tmp/functional-learning-theory-diagram/functional-theory-error-chain.pdf /tmp/functional-learning-theory-diagram/preview
```


[functional-operator-picard.tex](functional-operator-picard.tex) is the source for Figure 8. It compares exact Picard iteration with finite neural-operator blocks, then displays the perturbed-contraction error bound. It is an original mathematical exposition of Furuya et al.'s construction, not a reproduced paper figure or an experimental result.

From the blog repository root:

```sh
mkdir -p /tmp/functional-operator-diagram
pdflatex -interaction=batchmode -halt-on-error -output-directory=/tmp/functional-operator-diagram content/posts/functional-learning/materials/diagrams/functional-operator-picard.tex
pdftocairo -svg /tmp/functional-operator-diagram/functional-operator-picard.pdf content/posts/functional-learning/figures/functional-operator-picard.svg
pdftoppm -scale-to 1600 -png -singlefile /tmp/functional-operator-diagram/functional-operator-picard.pdf /tmp/functional-operator-diagram/preview
```

## FGD representation diagrams

The three original schematics explain what each finite code stores and how the released FGD implementation refines it. They are based on static source inspection at revision 57f0bab8df7bd3fe5e4d1ff8c23953989c62befc, not reproduced experiment results.

- [fgd-tree-representation.tex](fgd-tree-representation.tex), Figure 3: tree-leaf values, copying on subdivision, and a separately computed RKHS gradient. The displayed numerical update is illustrative.
- [fgd-fourier-representation.tex](fgd-fourier-representation.tex), Figure 4: a two-dimensional slice of the three-dimensional frequency-bin representation, followed by inverse Fourier synthesis.
- [fgd-voxel-representation.tex](fgd-voxel-representation.tex), Figure 6: the default piecewise-constant density and fixed-degree RGB spherical-harmonic coefficients. One voxel is shown, although refinement is global. Four color slots illustrate degree L=1.

The copying operations preserve the represented functions. Rendering's additional visibility masking is a separate operation and can change density. Source: [the authors' implementation at the inspected revision](https://github.com/dccsillag/experiments-adaptive-fgd/tree/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc).

From the blog repository root:

```sh
for diagram_name in fgd-tree-representation fgd-fourier-representation fgd-voxel-representation; do
  diagram_build="/tmp/functional-learning-representation-diagrams/$diagram_name"
  mkdir -p "$diagram_build"
  pdflatex -interaction=batchmode -halt-on-error -output-directory="$diagram_build" "content/posts/functional-learning/materials/diagrams/$diagram_name.tex"
  pdftocairo -svg "$diagram_build/$diagram_name.pdf" "content/posts/functional-learning/figures/$diagram_name.svg"
  pdftoppm -scale-to 1600 -png -singlefile "$diagram_build/$diagram_name.pdf" "$diagram_build/preview"
done
```

Inspect each preview after changing the sources; also check the diagrams at the article's reading-column width.
