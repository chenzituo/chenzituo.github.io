# Walk on Anything — development materials

The live article is `../index.md`. It now develops one running question: what can be reused across physical solves when boundary values change?

- `reading-ledger.md` and `sources.json`: source identities, inspected coverage and evidence boundaries.
- `figure-sources.json`: provenance, displayed/retired status, crops, authors and reuse basis. The article displays six source figure excerpts and two original figures.
- `extract_source_figures.py`: verify PDF hashes and render the six excerpts using Poppler/Pillow. Run with a directory containing the named source PDFs; full PDFs are not redistributed.
- `visual-study.md`: source design choices and how they changed this post.
- `interface_example.py` and `interface-example-results.json`: original analytic conduction example.
- `cache_architecture.py`: revised original static-context/dynamic-query proposal, emitted as SVG.
- `visuals.py` and `walk-example-results.json`: original disk experiment and earlier visual studies; their images are retained for development history, not all displayed.
- `../demos/walk-disk.html.txt`: interactive disk example, unchanged by this revision.

The numerical illustrations used Python 3.9, NumPy 2.0.2 and Matplotlib 3.9.4. Scripts resolve output paths relative to their locations. Source extraction additionally uses Pillow and `pdftoppm`. Proposed model and cache performance experiments have not been run.
