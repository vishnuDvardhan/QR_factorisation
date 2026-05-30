#!/usr/bin/env python3
"""Plot the per-layout nb sweep from results/sweep.csv."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path('/home/vishnu/QR_factorisation/results')
df = pd.read_csv(RESULTS / 'sweep.csv')
df = df[df['unblocked_cyc'].notna() & df['blocked_cyc'].notna()].copy()
df['speedup'] = df['unblocked_cyc'] / df['blocked_cyc']

print('\n=== sweep results ===')
print(df[['layout','m','n','nt','nb','unblocked_cyc','blocked_cyc',
         'speedup','max_err','match']].to_string(index=False))

if df.empty:
    print('no valid rows yet'); raise SystemExit

# Plot 1: cycles vs nb, per layout
layouts = sorted(df['layout'].unique())
fig, axes = plt.subplots(1, len(layouts), figsize=(6 * len(layouts), 4), squeeze=False)
for ax, layout in zip(axes[0], layouts):
    sub = df[df['layout'] == layout].sort_values('nb')
    if sub.empty: continue
    ax.plot(sub['nb'], sub['unblocked_cyc'], marker='o', label='Unblocked')
    ax.plot(sub['nb'], sub['blocked_cyc'],   marker='s', label='WY-Blocked')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sub['nb']); ax.set_xticklabels(sub['nb'])
    ax.set_xlabel('block width nb'); ax.set_ylabel('cycles')
    ax.set_title(f'{layout}  (m={int(sub.iloc[0].m)}, n={int(sub.iloc[0].n)}, nt={int(sub.iloc[0].nt)})')
    ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / 'sweep_cycles_vs_nb.png', dpi=120)
print(f'\nsaved {RESULTS / "sweep_cycles_vs_nb.png"}')

# Plot 2: speedup vs nb
fig2, ax2 = plt.subplots(figsize=(7, 4))
for layout in layouts:
    sub = df[df['layout'] == layout].sort_values('nb')
    if sub.empty: continue
    ax2.plot(sub['nb'], sub['speedup'], marker='o', label=layout)
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.4, label='break-even')
ax2.set_xscale('log', base=2)
nb_vals = sorted(df['nb'].unique())
ax2.set_xticks(nb_vals); ax2.set_xticklabels(nb_vals)
ax2.set_xlabel('block width nb'); ax2.set_ylabel('blocked / unblocked speedup')
ax2.set_title('WY-blocked speedup over unblocked baseline')
ax2.grid(alpha=0.3); ax2.legend()
fig2.tight_layout()
fig2.savefig(RESULTS / 'sweep_speedup_vs_nb.png', dpi=120)
print(f'saved {RESULTS / "sweep_speedup_vs_nb.png"}')
