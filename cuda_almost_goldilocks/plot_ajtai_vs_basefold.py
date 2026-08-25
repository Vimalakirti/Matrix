#!/usr/bin/env python3
"""
Plot commit-time comparison: Basefold (Goldilocks) vs Ajtai (almost-Goldilocks)
after the three GPU optimizations described in optimize_gpu_ajtai.md.

Both schemes commit to a polynomial with 2^log_n coefficients. Basefold reads
2^log_n Goldilocks field elements. Ajtai reads 2^log_n binary coefficients
packed into 2^(log_n - 6) ring elements (one u64 per ring element).

Data captured on a single A100-SXM4-80GB (sm_80). Min-of-N iterations
(steady state, post-warmup). All times in ms.

Output: ajtai_vs_basefold.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ----- Measured data (post-optimization) -----

log_n   = np.array([14, 16, 18, 20, 22, 24, 26])
n_coefs = 2.0 ** log_n            # 16K, 64K, 256K, 1M, 4M, 16M, 64M

basefold  = np.array([1.25, 1.81, 3.22, 8.49, 45.29, 212.0, 1619.0])
ajtai_b1  = np.array([2.12, 2.13, 2.18, 3.24, 6.75, 23.35, 88.13])
ajtai_b8  = np.array([0.56, 0.57, 0.64, 2.13, 8.29, 32.30, 207.00])   # per-commit
ajtai_b16 = np.array([0.47, 0.51, 0.51, 1.82, 7.17, 29.33, 102.70])   # per-commit

# ----- Two-panel figure -----

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

ax1.loglog(n_coefs, basefold,  'o-', label='Basefold (Goldilocks)',
           color='#1f77b4', linewidth=2.2, markersize=8)
ax1.loglog(n_coefs, ajtai_b1,  's-', label='Ajtai B=1',
           color='#ff7f0e', linewidth=2.2, markersize=8)
ax1.loglog(n_coefs, ajtai_b8,  '^-', label='Ajtai B=8 (per-commit)',
           color='#2ca02c', linewidth=2.2, markersize=8)
ax1.loglog(n_coefs, ajtai_b16, 'D-', label='Ajtai B=16 (per-commit)',
           color='#d62728', linewidth=2.2, markersize=8)

ax1.set_xlabel('Polynomial coefficients  N')
ax1.set_ylabel('Commit time (ms)')
ax1.set_title('(a) Commit time vs polynomial size — single A100')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.92)

ax1.set_xticks(n_coefs)
ax1.set_xticklabels([f'$2^{{{int(k)}}}$' for k in log_n])

# (b) Speedup of Ajtai over Basefold
speedup_b1  = basefold / ajtai_b1
speedup_b8  = basefold / ajtai_b8
speedup_b16 = basefold / ajtai_b16

ax2.loglog(n_coefs, speedup_b1,  's-', color='#ff7f0e',
           linewidth=2.2, markersize=8, label='Ajtai B=1')
ax2.loglog(n_coefs, speedup_b8,  '^-', color='#2ca02c',
           linewidth=2.2, markersize=8, label='Ajtai B=8')
ax2.loglog(n_coefs, speedup_b16, 'D-', color='#d62728',
           linewidth=2.2, markersize=8, label='Ajtai B=16')
ax2.axhline(y=1.0, color='#1f77b4', linestyle='--', linewidth=1.5,
            label='Basefold (parity)')

ax2.set_xlabel('Polynomial coefficients  N')
ax2.set_ylabel('Speedup over Basefold (×)')
ax2.set_title('(b) Ajtai speedup over Basefold — >1× means Ajtai is faster')
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(loc='upper left', fontsize=10, framealpha=0.92)
ax2.set_xticks(n_coefs)
ax2.set_xticklabels([f'$2^{{{int(k)}}}$' for k in log_n])

plt.suptitle('Commit time: Basefold (Goldilocks) vs Ajtai (almost-Goldilocks) — single A100',
             fontsize=13, y=1.02, fontweight='bold')

plt.tight_layout()
out = 'ajtai_vs_basefold.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'wrote {out}')
