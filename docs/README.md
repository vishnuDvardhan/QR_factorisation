# docs/

Explanatory LaTeX writeup of the v7 DAG-priority HVX Householder QR scheduler.

- `v7_dag_hvx_sched.tex` — source (visuals via TikZ, worked numerical examples,
  algorithm2e pseudocode tied to the real code, measured ISS results).
- `v7_dag_hvx_sched.pdf` — compiled output (9 pages).

## Building

The conda TeX Live on this machine has broken Perl wrappers (`fmtutil`/`mktexlsr`
fail, `pdflatex.fmt` cannot be generated). Use **tectonic** instead, which is
self-contained:

```bash
cd docs
tectonic v7_dag_hvx_sched.tex      # produces v7_dag_hvx_sched.pdf
```

(UTF-8 warnings from `algorithm2e.sty` and one underfull-hbox are cosmetic.)
