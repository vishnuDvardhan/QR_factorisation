#!/usr/bin/env python3
"""Parse qr_compare_all log and produce a comparison plot + CSV.

Usage:  python3 plot_compare.py results/compare_128.log
"""
import re, sys, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RX_LINE = re.compile(
    r'\[(\d)\]\s+(.+?)\s+\.\.\.\s+(\d+)\s+cycles'
    r'(?:\s+\|\s+max \|R\|-err vs CPU:\s+([\d.eE+\-]+)\s+(OK|FAIL))?'
)


def parse(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = RX_LINE.search(line)
            if m:
                idx, name, cyc, err, status = m.groups()
                rows.append({
                    'idx': int(idx),
                    'name': name.strip(),
                    'cycles': int(cyc),
                    'err': float(err) if err else 0.0,
                    'status': status or 'ref',
                })
    return rows


def plot(rows, out_png):
    cpu_cyc = next((r['cycles'] for r in rows if r['idx'] == 1), None)
    rows_plot = [r for r in rows if r['idx'] != 1]

    names = [r['name'] for r in rows_plot]
    cycs  = [r['cycles'] for r in rows_plot]
    # Color by relative performance vs CPU ref (faster=green, slower=red)
    if cpu_cyc:
        colors = ['#2ca02c' if c < cpu_cyc else '#d62728' for c in cycs]
    else:
        colors = ['#1f77b4'] * len(cycs)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), cycs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('cycles')
    ax.set_title('QR algorithm comparison — Hexagon HVX + QuRT')
    ax.grid(axis='x', alpha=0.3)
    if cpu_cyc:
        ax.axvline(cpu_cyc, color='gray', linestyle='--', alpha=0.5,
                   label=f'CPU scalar ref ({cpu_cyc:,})')
        ax.legend(loc='lower right')
    # annotate cycles
    for b, c in zip(bars, cycs):
        ax.text(c * 1.01, b.get_y() + b.get_height()/2, f'{c:,}',
                va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f'saved {out_png}')


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else 'results/compare_128.log'
    log = Path(log)
    rows = parse(log)
    if not rows:
        print('no algorithm lines parsed; sim may not be complete yet')
        return
    print(f'\n=== {log} ===')
    print(f'{"#":>2s}  {"algorithm":<34s}  {"cycles":>12s}  vs ref  status')
    print('-' * 80)
    cpu_cyc = next((r['cycles'] for r in rows if r['idx'] == 1), None)
    for r in rows:
        vs = (cpu_cyc / r['cycles']) if cpu_cyc and r['idx'] != 1 else 1.0
        print(f'{r["idx"]:>2d}  {r["name"]:<34s}  {r["cycles"]:>12,d}  {vs:>5.2f}x  {r["status"]}')

    csv_path = log.with_suffix('.csv')
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'name', 'cycles', 'err', 'status'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved {csv_path}')

    png_path = log.with_suffix('.png')
    plot(rows, png_path)


if __name__ == '__main__':
    main()
