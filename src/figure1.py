#!/usr/bin/env python3
"""Figure 1 (extraction quality): QRS-width distribution, compensatory-pause
ratio (from the per-beat file of quality_control.py, so the figure and Table 1
come from one computation), and a grid of randomly sampled extracted beats in
leads II and V2.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEATS as DATA, RESULTS, FIGURES as FIG

QC = f"{RESULTS}/qc"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'savefig.bbox': 'tight'})
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LI = {l: i for i, l in enumerate(LEADS)}

meta = pd.read_csv(f"{DATA}/pvc_meta.csv")
X = np.load(f"{DATA}/pvc_beats.npy")
qc = pd.read_csv(f"{QC}/ptbxl_beat_qc.csv")
ratios = qc[(qc.evaluable) & (~qc.afib.astype(bool))].ratio.dropna().values

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
ax[0, 0].hist(meta.width_ms, bins=30, color='#444')
ax[0, 0].axvline(140, color='r', ls='--', lw=1)
ax[0, 0].set_xlabel('QRS width (ms)')
ax[0, 0].set_ylabel('beats')
ax[0, 0].set_title('A  QRS width')

ax[0, 1].hist(ratios, bins=25, color='#444')
ax[0, 1].axvline(0.9, color='r', ls='--', lw=1)
ax[0, 1].set_xlabel('(pre+post RR)/(2 × median RR)')
ax[0, 1].set_title('B  Compensatory pause')

np.random.seed(1)
sel = np.random.choice(len(X), 6, replace=False)
cells = [ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]]
for cell, i in zip(cells, sel):
    cell.plot(X[i][LI['II']], 'k', lw=.6)
    cell.plot(X[i][LI['V2']], 'r', lw=.6)
    cell.set_xticks([])
    cell.set_yticks([])
cells[0].set_title('C  example PVC beats (II = black, V2 = red)', loc='left',
                   fontsize=8)

fig.savefig(f"{FIG}/Fig1_extraction_quality.tiff", dpi=600,
            pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(f"{FIG}/Fig1_extraction_quality.pdf")
plt.close(fig)
print(f"Fig1 from {len(meta)} beats; pause ratios n={len(ratios)}, "
      f"median {np.median(ratios):.2f}, {100*(ratios>=0.9).mean():.0f}% >=0.9")
print(f"  -> {FIG}/Fig1_extraction_quality.tiff + .pdf")
