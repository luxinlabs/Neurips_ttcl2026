# Compiling `paper.tex`

The official `neurips_2026.sty` is **not** included in this repository — it was deliberately not
fetched via an AI text-summarizing tool, since that risks silently corrupting exact LaTeX macro
syntax. Download it yourself from the verified official source before compiling:

- `media.neurips.cc/Conferences/NeurIPS2026/Styles` (official author kit), or
- the NeurIPS 2026 Overleaf template, which bundles the same file.

Place `neurips_2026.sty` in this directory (`paper/`), next to `paper.tex`, then compile with
the standard four-pass LaTeX/BibTeX cycle:

```
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Use `\usepackage[preprint]{neurips_2026}` (already set in `paper.tex`) for anonymous
submission, or switch to `[final]` for camera-ready.

## What's already verified

`paper.tex` and `references.bib` were compile-tested against a minimal syntax-check shim
(not the real style file) before this README was written: 0 undefined citations, 0 undefined
cross-references, clean `bibtex` run, valid PDF output. The page count under that shim (9
pages, plain 1-inch margins) is **not** representative of the real NeurIPS single-column
layout and should not be used to judge the 4–9 page limit — re-check page count only after
compiling with the real `neurips_2026.sty`.

## Known placeholders still in the text

- Author affiliation (`[Affiliation --- to be added]`)
- AI Usage Disclosure section (pending the venue-specific disclosure step)
