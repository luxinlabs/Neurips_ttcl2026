# Compiling `paper.tex`

The official `neurips_2026.sty` is **not** included in this repository — it was deliberately not
fetched via an AI text-summarizing tool, since that risks silently corrupting exact LaTeX macro
syntax. Download it yourself from the verified official source before compiling:

- `media.neurips.cc` (official author kit — exact zip path not confirmed by direct download
  in this repo's development; guessed paths returned 404, so don't assume a specific URL —
  navigate from `https://neurips.cc` to the current Style Files page instead), or
- the NeurIPS 2026 Overleaf template ("Formatting Instructions For NeurIPS 2026"), which
  bundles the same file.

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

`paper.tex` and `references.bib` have been compile-tested twice against syntax-check shims
(not the real style file):

1. An initial pass with generic 1-inch margins (9 pages — not representative, superseded).
2. A follow-up pass with a shim matching **NeurIPS's actual text-rectangle dimensions**
   (5.5in-wide column, 1.5in margins, 10pt/11pt leading — pulled from the authoritative
   "Formatting Instructions For NeurIPS 2026" PDF, downloaded directly rather than fetched
   through a summarizer). Result: **10 total pages**, section breakpoints Related Work p.2 /
   Method p.4 / Results p.6 / Discussion+Conclusion p.7.

Both passes: 0 undefined citations, 0 undefined cross-references, clean `bibtex` run, valid
PDF output. `\citep`/`\citet` usage was also audited — 13 instances that would have rendered
as duplicated author names (e.g. "Kirkpatrick et al. (Kirkpatrick et al., 2017)") were found
and fixed.

**Unresolved risk, needs the real style file to settle**: NeurIPS's page-limit rule exempts
"acknowledgments, references, checklist, and optional technical appendices" from the 9-page
content cap — it does **not** explicitly exempt the paper's Limitations, Ethics Statement,
Author Contributions, Conflict of Interest, Funding, or Use of AI Tools sections, all of which
sit between Conclusion and References. Under the dimensionally-accurate shim, core content
(Intro through Conclusion) already reaches page 7-8; adding those six mandatory sections could
plausibly push counted content past 9 pages depending on how the TTCL workshop interprets the
exemption. Verify against the workshop's own OpenReview submission instructions, and consider
consolidating some of those sections (e.g. folding CoI + Funding into one short paragraph) if
the real compile comes in over budget.

## References

All 18 entries were independently verified against primary sources (arXiv abstract pages,
Crossref, or the publisher page) in this repo's development — not just checked once but
re-audited from scratch as a final pass. One real error was caught and fixed in that re-audit:
the Shumailov et al. (Nature 2024) author order had Yarin Gal in the wrong position (4th
instead of last). No fabricated or unreachable references remain as of the last audit.

## Known placeholders still in the text

- Author affiliation (`[Affiliation --- to be added]`)
