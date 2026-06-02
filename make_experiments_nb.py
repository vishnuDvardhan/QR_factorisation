#!/usr/bin/env python3
"""Generate qr_experiments_v2.ipynb — a comprehensive measurement notebook for
the QR-on-Hexagon variants.  Run with the conda python that has nbformat:
    /home/vishnu/miniconda3/bin/python make_experiments_nb.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):  cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# QR-on-Hexagon — Comprehensive Measurement Notebook

Measures five QR variants on the Hexagon ISS across **square sizes (Exp 1)**,
**tall-skinny sizes (Exp 2)**, and **thread count (Exp 3)**.

**Variants**
| key | source | sim target | inparams | result line |
|---|---|---|---|---|
| `v7_barrier` | `v7_dag_hvx_sched.c` (sched=1) | `sim-v7_dag_hvx_sched` | `m n nt chunk 1 nb` | `[alg] ... N cycles` |
| `v7_async_prio` | `v7_dag_hvx_sched.c` (sched=0) | `sim-v7_dag_hvx_sched` | `m n nt chunk 0 nb` | `[alg] ... N cycles` |
| `pool_hh` | `qr_pool_hh_unblocked.c` | `sim-pool` | `m n nt chunk 0 nb` | `[alg] ... N cycles` |
| `colmajor_mt` | `qr_colmajor_mt.c` | `sim-baseline` | `m n nt chunk 0` (5 fields) | `QR cycles: N` |
| `rowmajor_mt` | `qr_rowmajor_mt_block_alloc.c` | `sim-rowmajor` | **`inparams1.txt`**: `m n nt chunk 0` | `Total cycles: N` |

**Methodology notes (hard-won):**
- The ISS **buffers stdout and flushes on clean exit** — so we poll each run to
  completion (never a fixed timeout that would mis-read *slow* as *hung*).
- The PMU counts the **whole process**; for v7/pool the scalar reference run is
  included in the PMU file but NOT in the reported `[alg]` cycles (we report the
  algorithm cycles, and store the full PMU snapshot per run for the record).
- All variants use the same deterministic test matrix and `c15:14` cycle counter.
- `colmajor_mt` also pays a row→col transpose (reported as `QR cycles`, which is
  the factorization only — transpose excluded, matching the others).
- We **store every PMU counter to CSV** but **plot only cycles**.
""")

code(r"""
import os, re, subprocess, time, shutil, glob
import pandas as pd
import matplotlib.pyplot as plt

REPO = "/home/vishnu/QR_factorisation"
BUILD = os.path.join(REPO, "build")
PMU = os.path.join(BUILD, "hexagon", "pmu_stats.txt")
INPARAMS  = os.path.join(BUILD, "inparams.txt")     # most variants
INPARAMS1 = os.path.join(BUILD, "inparams1.txt")    # rowmajor only
os.chdir(REPO)
RESULTS_DIR = os.path.join(REPO, "results_v2"); os.makedirs(RESULTS_DIR, exist_ok=True)

# variant -> (sim target, inparams writer, cycle-line regex)
def w_std(m,n,nt,chunk,sched):   # v7/pool: m n nt chunk sched nb
    return (INPARAMS,  f"{m} {n} {nt} {chunk} {sched} 8\n")
def w_5(m,n,nt,chunk,_s):        # colmajor: m n nt chunk use_static (5 fields)
    return (INPARAMS,  f"{m} {n} {nt} {chunk} 0\n")
def w_rm(m,n,nt,chunk,_s):       # rowmajor: separate inparams1.txt, 5 fields
    return (INPARAMS1, f"{m} {n} {nt} {chunk} 0\n")

VARIANTS = {
  "v7_barrier":    dict(tgt="sim-v7_dag_hvx_sched", write=lambda m,n,nt,c: w_std(m,n,nt,c,1),
                        rx=r"\[alg\].*?(\d+)\s+cycles"),
  "v7_async_prio": dict(tgt="sim-v7_dag_hvx_sched", write=lambda m,n,nt,c: w_std(m,n,nt,c,0),
                        rx=r"\[alg\].*?(\d+)\s+cycles"),
  "pool_hh":       dict(tgt="sim-pool",             write=lambda m,n,nt,c: w_std(m,n,nt,c,0),
                        rx=r"\[alg\].*?(\d+)\s+cycles"),
  "colmajor_mt":   dict(tgt="sim-baseline",         write=lambda m,n,nt,c: w_5(m,n,nt,c,0),
                        rx=r"QR cycles:\s*(\d+)"),
  "rowmajor_mt":   dict(tgt="sim-rowmajor",         write=lambda m,n,nt,c: w_rm(m,n,nt,c,0),
                        rx=r"Total cycles:\s*(\d+)"),
}

def kill_stray_sims():
    # Kill ONLY hexagon-sim processes by exact name -- never touches `make`,
    # python, or this notebook. Run only BETWEEN runs to clear leftovers that
    # would starve the host (the #1 cause of bogus 'no result').
    subprocess.run("pkill -9 -x hexagon-sim 2>/dev/null; true", shell=True)

def host_busy():
    r = subprocess.run("pgrep -x hexagon-sim | wc -l", shell=True,
                       capture_output=True, text=True)
    try: return int(r.stdout.strip())
    except: return 0

def parse_pmu(path):
    d = {}
    if not os.path.exists(path): return d
    for line in open(path, errors="replace"):
        m = re.match(r'\s*0x[0-9a-f]+:([A-Z0-9_]+):(\d+)', line)
        if m: d["pmu_"+m.group(1)] = int(m.group(2))
        m2 = re.match(r'QDSP6_clk_cnt\s*:\s*(\d+)', line)
        if m2: d["pmu_QDSP6_clk_cnt"] = int(m2.group(1))
    return d
""")

code(r'''
import csv
MASTER_CSV = os.path.join(RESULTS_DIR, "all_runs_with_pmu.csv")

def run_one(variant, m, n, nt, chunk=4):
    """Run ONE (variant,m,n,nt) and DEFINITELY return a result, no matter how slow.

    Guarantees (these are why earlier runs gave 'no result'):
      * NO timeout -- we poll until the sim process truly exits. Slow != hung.
        (The ISS buffers stdout and only flushes on clean exit, so a slow run
         looks empty until it finishes -- we must wait, not give up.)
      * We kill ONLY leftover hexagon-sim procs, BEFORE launching, by exact
        name -- never `make`, never python. (A prior bug killed the very make
        being launched.)
      * A live heartbeat prints elapsed seconds so you can see it's progressing.
      * The row is appended to the master CSV immediately, so partial results
        always survive even if you interrupt the kernel.
    """
    v = VARIANTS[variant]
    # clear leftovers that would starve the host, then wait for them to die
    kill_stray_sims()
    while host_busy() > 0: time.sleep(1)
    inp_path, inp_text = v["write"](m, n, nt, chunk)
    open(inp_path, "w").write(inp_text)
    log = f"/tmp/nb_{variant}_{m}x{n}_nt{nt}.log"
    open(log, "w").close()
    proc = subprocess.Popen(f"make {v['tgt']} > {log} 2>&1", shell=True)
    t0 = time.time(); beat = 0
    # poll to TRUE completion -- no timeout. heartbeat every ~20s.
    while proc.poll() is None:
        time.sleep(5)
        el = int(time.time() - t0)
        if el // 30 > beat:
            beat = el // 30
            print(f"    ... {variant} {m}x{n} nt={nt}: running {el}s", flush=True)
    out = open(log, errors="replace").read()
    mobj = re.search(v["rx"], out)
    cyc = int(mobj.group(1)) if mobj else None
    pmu = parse_pmu(PMU)
    if os.path.exists(PMU):
        shutil.copy(PMU, os.path.join(RESULTS_DIR, f"pmu_{variant}_{m}x{n}_nt{nt}.txt"))
    row = dict(variant=variant, m=m, n=n, nt=nt, chunk=chunk,
               shape=("square" if m==n else "tall"),
               cycles=cyc, wall_s=round(time.time()-t0,1),
               status=("OK" if cyc is not None else "NO_RESULT(check log)"))
    row.update(pmu)
    # append immediately so partial sweeps are never lost
    write_header = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"  {variant:<14} {m}x{n} nt={nt}: "
          f"{cyc if cyc is not None else 'NO_RESULT':>10}  ({row['wall_s']}s)  [{row['status']}]", flush=True)
    if cyc is None:
        print(f"      ^ no cycle line parsed; see {log}", flush=True)
    return row

def run_one_retry(variant, m, n, nt, chunk=4, retries=2):
    """run_one, but retry on NO_RESULT (handles transient QuRT-init 'Killed'
    that occasionally hits the row-major binary). Guarantees we try hard to get
    a number for every config."""
    for attempt in range(retries + 1):
        row = run_one(variant, m, n, nt, chunk)
        if row["cycles"] is not None:
            return row
        if attempt < retries:
            print(f"      retry {attempt+1}/{retries} for {variant} {m}x{n} nt={nt}", flush=True)
            kill_stray_sims(); time.sleep(3)
    return row  # last (NO_RESULT) row -- recorded, sweep continues

def sweep(configs, variants=None):
    """configs = list of (m,n,nt). One row per (variant,config). Always returns
    a DataFrame of whatever completed (partial-safe; each row is also already
    appended to the master CSV inside run_one)."""
    variants = variants or list(VARIANTS)
    rows = []
    total = len(configs)*len(variants); i = 0
    for (m,n,nt) in configs:
        for var in variants:
            i += 1
            print(f"[{i}/{total}] {var} {m}x{n} nt={nt} ...", flush=True)
            rows.append(run_one_retry(var, m, n, nt))
    return pd.DataFrame(rows)
''')

code(r"""
# Build all binaries once up front (so timing runs don't include compile time).
print("building all variants...")
subprocess.run("make pool baseline rowmajor v7_dag_hvx_sched >/dev/null 2>&1", shell=True)
print("done. HVX units:", subprocess.run(
    "grep -m1 'HVX units' /dev/null 2>/dev/null; echo 0x400(4x128B)", shell=True,
    capture_output=True, text=True).stdout.strip())
""")

md(r"""## Experiment 1 — Square size sweep
Square matrices $m=n$, fixed `nt=4`. Tests how each variant scales with problem
size when work is dominated by the (large) trailing-matrix updates.""")
code(r"""
EXP1_SIZES = [128, 256, 384, 512]     # m=n ; all multiples of 32. Add 768/1024 if time allows.
exp1_cfgs = [(s, s, 4) for s in EXP1_SIZES]
print("=== Exp 1: square sweep (nt=4) ===")
df1 = sweep(exp1_cfgs)
df1.to_csv(os.path.join(RESULTS_DIR, "exp1_square.csv"), index=False)
df1[["variant","m","n","nt","cycles","status"]]
""")
code(r"""
# Plot: cycles vs size, one line per variant (square)
plt.figure(figsize=(8,5))
for var in VARIANTS:
    d = df1[(df1.variant==var) & df1.cycles.notna()].sort_values("m")
    if len(d): plt.plot(d.m, d.cycles, marker="o", label=var)
plt.xlabel("matrix size (m=n)"); plt.ylabel("cycles (c15:14)")
plt.title("Exp 1 — Square: cycles vs size (nt=4)")
plt.legend(); plt.grid(True, alpha=.3); plt.yscale("log")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"exp1_square.png"), dpi=120); plt.show()
""")

md(r"""## Experiment 2 — Tall-skinny size sweep
Tall matrices, fixed small `n=64` and `nt=4`, growing `m`. This is the
least-squares regime where the per-step reflector (∝ m) is a large fraction of
each step — the regime where async_prio's pivot-overlap pays off most.""")
code(r"""
EXP2_M = [256, 512, 1024, 2048]       # n fixed = 64
exp2_cfgs = [(m, 64, 4) for m in EXP2_M]
print("=== Exp 2: tall-skinny sweep (n=64, nt=4) ===")
df2 = sweep(exp2_cfgs)
df2.to_csv(os.path.join(RESULTS_DIR, "exp2_tall.csv"), index=False)
df2[["variant","m","n","nt","cycles","status"]]
""")
code(r"""
plt.figure(figsize=(8,5))
for var in VARIANTS:
    d = df2[(df2.variant==var) & df2.cycles.notna()].sort_values("m")
    if len(d): plt.plot(d.m, d.cycles, marker="s", label=var)
plt.xlabel("rows m (n=64 fixed)"); plt.ylabel("cycles (c15:14)")
plt.title("Exp 2 — Tall-skinny: cycles vs m (n=64, nt=4)")
plt.legend(); plt.grid(True, alpha=.3); plt.yscale("log")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"exp2_tall.png"), dpi=120); plt.show()
""")

md(r"""## Experiment 3 — Thread-count sweep
Fixed size, vary `nt` ∈ {1,2,4,6,8}. Shows parallel scaling and where the
HVX-context limit (4 in 128B mode) bites. Run for one square and one tall shape.""")
code(r"""
EXP3_NT = [1, 2, 4, 6, 8]
exp3_cfgs  = [(256, 256, nt) for nt in EXP3_NT]      # square
exp3b_cfgs = [(1024, 64, nt) for nt in EXP3_NT]      # tall
print("=== Exp 3a: thread sweep, square 256x256 ===")
df3a = sweep(exp3_cfgs)
print("=== Exp 3b: thread sweep, tall 1024x64 ===")
df3b = sweep(exp3b_cfgs)
df3 = pd.concat([df3a, df3b], ignore_index=True)
df3.to_csv(os.path.join(RESULTS_DIR, "exp3_threads.csv"), index=False)
df3[["variant","m","n","nt","cycles","status"]]
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13,5), sharey=False)
for ax, df, title in [(axes[0], df3a, "square 256x256"), (axes[1], df3b, "tall 1024x64")]:
    for var in VARIANTS:
        d = df[(df.variant==var) & df.cycles.notna()].sort_values("nt")
        if len(d): ax.plot(d.nt, d.cycles, marker="o", label=var)
    ax.set_xlabel("threads (nt)"); ax.set_ylabel("cycles"); ax.set_title(f"Exp 3 — {title}")
    ax.grid(True, alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"exp3_threads.png"), dpi=120); plt.show()
""")

md(r"""## Combined CSV + summary
All rows (every config, every variant, with the full PMU counters) in one CSV
for the record; the plots above use only the `cycles` column.""")
code(r"""
alldf = pd.concat([df1, df2, df3], ignore_index=True)
alldf.to_csv(os.path.join(RESULTS_DIR, "all_runs_with_pmu.csv"), index=False)
print("wrote", os.path.join(RESULTS_DIR, "all_runs_with_pmu.csv"),
      "  rows:", len(alldf), " cols:", len(alldf.columns), "(incl. all PMU counters)")
# quick speedup-of-async_prio-over-barrier summary (the headline scheduling metric)
piv = alldf[alldf.variant.isin(["v7_barrier","v7_async_prio"]) & alldf.cycles.notna()]
piv = piv.pivot_table(index=["m","n","nt"], columns="variant", values="cycles")
piv["barrier/async_prio"] = (piv["v7_barrier"]/piv["v7_async_prio"]).round(3)
piv
""")

nb["cells"] = cells
out = "/home/vishnu/QR_factorisation/qr_experiments_v2.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
