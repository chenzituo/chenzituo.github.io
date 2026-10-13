# Build, verify and publish

The repository may contain substantial unrelated research materials. Inspect `git status` and scoped diffs before editing or staging. Do not publish the full working tree by accident.

For an isolated preview, export `git archive HEAD` into a temporary directory, overlay only the intended changed/new files, and supply the actual PaperMod submodule (the archive leaves an empty submodule directory). Build with the installed Hugo version matching `.github/workflows/hugo.yaml`. For a draft use `--buildDrafts`; also confirm ordinary production output excludes it. A draft changed from a published placeholder will remove that placeholder on a future deploy, so distinguish the local draft from the live site.

Check the generated article with `scripts/check_page.py <output-root> <relative-html-path>`. It checks local href/src targets and HTML fragments. It does not verify external sources, MathJax, JavaScript, visual layout or scientific correctness.

Use the available computer-use browser tool for rendered review. Check equations, figures and interactions in context. Inspect narrow layouts when introducing a wide table, diagram or demo. Keep long equations aligned. A successful build is not evidence of correct math rendering.

For an authorized publication:

- Stage the exact intended files, inspect the staged diff, and build that intended snapshot.
- Commit/push as requested. Observe the GitHub Actions deployment for the new commit and verify the actual live URLs/content. Distinguish a successful push from a successful deployment.
- Do not include downloaded PDFs or private research receipts in public assets without a reason and authorization. Leaf-bundle Markdown material files are not automatically published as standalone HTML pages; use a verified repository link or an actual public artifact when appropriate.
- End temporary servers and watch processes; preserve a requested live preview only when its continuation is intentional.

Report what changed, what was checked, and any unresolved limitation. Never claim tests, performance measurements or publication that did not occur.
