# Zituo’s Notes

A Markdown blog built with Hugo and PaperMod, following the reading-focused format of https://lilianweng.github.io/. Includes posts, archives, search, tags, RSS, dark mode, code highlighting, a table of contents, and optional equations.

## Preview

Hugo 0.167.0 is installed on this Mac. From this folder run:

```sh
hugo server -D
```

Open http://localhost:1313/. Changes to Markdown update the preview automatically. `-D` includes drafts locally; the publishing workflow excludes them.

On a fresh machine, install Hugo and initialize the pinned theme:

```sh
brew install hugo
git submodule update --init --recursive
```

## Write a post

```sh
hugo new content posts/my-first-post/index.md
```

Edit the generated Markdown file in Obsidian or any text editor. Set its title, date, tags, and summary. Change `draft: true` to `draft: false` when it is ready. Future-dated posts are excluded until their date arrives.

`content/about.md` holds the About page. Edit the title, author, description, and home introduction in `hugo.yaml`.

### Finish a planned note

Replace the Coming soon text in `content/posts/<slug>/index.md` with your finished note, then run:

```sh
python3 scripts/finish_note.py physical-trajectory-modeling --summary "A short summary of the finished note."
```

Use the folder name of the note you finished. The command stamps `date` and `publishDate` with the current New York time, removes the `planned` tag, enables the table of contents and reading time, and places completed notes ahead of planned notes (newest completed first). The archive groups and displays the completion date after you commit and push. Later edits leave that date unchanged; running the command again on a completed note is rejected to preserve it. The command requires Python 3.9+ and no extra packages.

Keep post images beside their `index.md` and reference them as `![Description](figure.png)`. Use standard Markdown links and images; Obsidian wikilinks and embeds are not automatically converted.

For equations, set `math: true` and use `\(x\)` inline or `$$...$$` on separate lines for display equations. Math pages load MathJax 3.2.2 from jsDelivr; ordinary pages do not load it.

## Publish on GitHub Pages

1. Create a GitHub repository for the blog. Use `<username>.github.io` for a personal root site, or another name for a project site.
2. Set the public `baseURL` in `hugo.yaml` to your final URL.
3. Commit this folder, including `.gitmodules` and the theme submodule, and push the `main` branch to that repository.
4. In the repository’s Settings → Pages, select **GitHub Actions** as the source.
5. Run **Build and deploy blog** in the Actions tab, or push another change.

The included workflow builds with Hugo 0.167.0 and uses the URL supplied by GitHub Pages, including any project subdirectory.

This blog is configured for https://chenzituo.github.io/, with source at https://github.com/chenzituo/chenzituo.github.io. After editing and previewing a post, publish updates from this folder:

```sh
git add content assets static hugo.yaml
git commit -m "Update blog"
git push origin main
```

GitHub Actions builds and deploys each push to `main`. Check the repository’s **Actions** tab for deployment progress. The public site stays online when your Mac and local Hugo server are off.

For another static host, run `hugo --minify --baseURL 'https://your-domain.example/'` and publish `public/`. Do not commit `public/`.

## Theme

PaperMod is pinned as a Git submodule; its MIT license is in `themes/PaperMod/LICENSE`. Site-specific overrides live in `assets/css/extended/` and `layouts/_partials/`. The theme currently emits Hugo deprecation warnings about its language properties; builds succeed.

References: [PaperMod setup](https://github.com/adityatelange/hugo-PaperMod/wiki/Installation), [Hugo Pages deployment](https://gohugo.io/host-and-deploy/host-on-github-pages/).
