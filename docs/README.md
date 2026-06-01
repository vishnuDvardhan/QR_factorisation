# docs/

Explanatory LaTeX writeups of the v7 DAG-priority HVX Householder QR scheduler.

## Report (article)
- `v7_dag_hvx_sched.tex` / `.pdf` — 9-page report: visuals via TikZ, worked
  numerical examples, algorithm2e pseudocode tied to the real code, measured
  ISS results.

## Animated walkthrough (Beamer slides)
- `v7_walkthrough_slides.tex` / `.pdf` — a **presentation-style, click-to-animate**
  full dry run of Householder QR on a concrete 4x3 matrix
  (`A = [[4,-2,1],[0,6,-1],[3,1,2],[0,2,4]]`). 15 frames expand to ~52 overlay
  pages: open in a PDF viewer's presentation/full-screen mode and press
  space/arrow to reveal each sub-step (norm -> alpha -> v -> tau -> critical
  column update -> worker columns -> diagonal), watching R fill in. The dry-run
  numbers are machine-verified against `numpy.linalg.qr` (|R| match to 9e-16).
  Also animates the HVX tree reduction, the DAG critical path, the
  barrier-vs-async_prio schedule timeline, and the 4-context allocator.

## Building

The conda TeX Live on this machine has broken Perl wrappers (`fmtutil`/`mktexlsr`
fail, `pdflatex.fmt` cannot be generated). Use **tectonic** instead, which is
self-contained:

```bash
cd docs
tectonic v7_dag_hvx_sched.tex      # produces v7_dag_hvx_sched.pdf
```

(UTF-8 warnings from `algorithm2e.sty` and one underfull-hbox are cosmetic.)
