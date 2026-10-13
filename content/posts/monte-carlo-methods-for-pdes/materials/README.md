# Walk on Anything — development materials

- `reading-ledger.md`: source coverage, evidence boundaries and next research steps.
- `sources.json`: verified paper identities, source URLs, PDF hashes and local lookup provenance.
- `interface_example.py`: original two-material conduction illustration (NumPy/Matplotlib).
- `interface-example-results.json`: analytic interface value and flux checks.

The article is at `../index.md`; its original figure is `../figures/interface-response.png`. The outlook jointly considers representation, composition model and KV-cacheable infrastructure. Proposed model and cache experiments have not been run.

Regenerate the figure from any directory with `python /path/to/materials/interface_example.py`. Outputs are resolved relative to the script. The run used Python 3.9, NumPy 2.0.2 and Matplotlib 3.9.4 in the existing blog environment. The original PDF files are referenced by URL and local provenance, rather than redistributed as blog assets.

- `visuals.py` and `walk-example-results.json`: reproducible disk experiment and five additional figures.
- `figure-sources.json`: provenance and evidence scope for every visual.
- `../demos/walk-disk.html.txt`: self-contained interactive disk example.
