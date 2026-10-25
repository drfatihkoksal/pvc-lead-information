#!/usr/bin/env python3
"""Figures 2-5 from the KMAX=12 fits (Fig. 1 is figure1.py).

Each figure is written as 600-dpi LZW TIFF and PDF to config.FIGURES.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, LEADS, ELEC, SUBSETS, SIZES, SUB_IX, induced_leads)
from config import FIGURES as FIG

os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'savefig.bbox': 'tight'})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.tiff", dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)
    print(f"  {name}.tiff (600dpi) + .pdf")


d = np.load(f"{RES}/fits_k12.npz", allow_pickle=True)
Ep, Ec, Up, Uc = d['Ep'], d['Ec'], d['Up'], d['Uc']
mEp, mEc = Ep.mean(0), Ec.mean(0)

# ---------------- Fig 2: per-lead unique information ----------------
C = np.load(f"{RES}/cpsc_internal_k12.npz")['C']
mA, mC = Up.mean(0), C.mean(0)
order = np.argsort(mA)[::-1]
rho = np.array([spearmanr(Up[i], Uc[i]).correlation for i in range(len(Up))])
lA = mA - np.percentile(Up, 2.5, 0); hA = np.percentile(Up, 97.5, 0) - mA
lC = mC - np.percentile(C, 2.5, 0); hC = np.percentile(C, 97.5, 0) - mC
fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(12); w = 0.4
ax.bar(x - w / 2, mA[order], w, yerr=[lA[order], hA[order]], capsize=3,
       label='PTB-XL (internal CV)', color='#1f77b4')
ax.bar(x + w / 2, mC[order], w, yerr=[lC[order], hC[order]], capsize=3,
       label='CPSC-2018 (internal CV)', color='#ff7f0e')
ax.set_xticks(x); ax.set_xticklabels([LEADS[i] for i in order])
ax.legend(frameon=False)
ax.set_ylabel('unique information (reconstruction MSE)')
ax.set_title('Per-lead unique information (mean ± 95% CI; 5×5 CV) — '
             f'transfer Spearman {rho.mean():.2f}')
save(fig, "Fig2_lead_unique_information")

# ---------------- Fig 3: lead-subset Pareto ----------------
fig, ax = plt.subplots(figsize=(11, 6))
for k in range(1, 13):
    ix = np.where(SIZES == k)[0]
    ax.scatter([k] * len(ix), mEp[ix], s=3, color='lightgray', alpha=.4, zorder=1)
front = [np.where(SIZES == k)[0][np.argmin(mEp[SIZES == k])] for k in range(1, 13)]
fk = list(range(1, 13)); fm = [mEp[i] for i in front]
elo = [mEp[i] - np.percentile(Ep[:, i], 2.5) for i in front]
ehi = [np.percentile(Ep[:, i], 97.5) - mEp[i] for i in front]
ax.errorbar(fk, fm, yerr=[elo, ehi], fmt='o-', color='#1f77b4', capsize=3,
            label='best subset (PTB-XL)', zorder=5)
ax.plot(fk, [mEc[i] for i in front], 's--', color='#ff7f0e',
        label='same subset on CPSC', zorder=5)
for k in range(1, 8):
    i = front[k - 1]
    ax.annotate('+'.join(LEADS[j] for j in SUBSETS[i]), (k, mEp[i]), fontsize=7,
                rotation=15, xytext=(3, 4), textcoords='offset points')
ax.set_xlabel('subset size (number of leads)')
ax.set_ylabel('full 12-lead reconstruction MSE')
ax.set_title('Exhaustive lead-subset search: Pareto frontier '
             '(grey = all 4095 subsets)')
ax.set_xticks(range(1, 13)); ax.legend(frameon=False)
save(fig, "Fig3_lead_pareto")

# ---------------- Fig 4: electrode Pareto ----------------
import itertools
best = {}
for n in range(2, 10):
    cand = [(E, SUB_IX[induced_leads(E)])
            for E in itertools.combinations(ELEC, n) if induced_leads(E)]
    if cand:
        best[n] = min(cand, key=lambda c: mEp[c[1]])
ns = sorted(best)
fig, ax = plt.subplots(figsize=(11, 6))
ep = [mEp[best[n][1]] for n in ns]; ec = [mEc[best[n][1]] for n in ns]
elo = [mEp[best[n][1]] - np.percentile(Ep[:, best[n][1]], 2.5) for n in ns]
ehi = [np.percentile(Ep[:, best[n][1]], 97.5) - mEp[best[n][1]] for n in ns]
ax.errorbar(ns, ep, yerr=[elo, ehi], fmt='o-', color='#1f77b4', capsize=3,
            label='best (PTB-XL)')
ax.plot(ns, ec, 's--', color='#ff7f0e', label='same set on CPSC')
for n in ns:
    ax.annotate('+'.join(best[n][0]), (n, mEp[best[n][1]]), fontsize=7,
                rotation=12, xytext=(3, 4), textcoords='offset points')
ax.set_xlabel('number of electrodes')
ax.set_ylabel('full 12-lead reconstruction MSE')
ax.set_title('Electrode-level exhaustive search: Pareto frontier')
ax.set_xticks(ns); ax.legend(frameon=False)
save(fig, "Fig4_electrode_pareto")

# ---------------- Fig 5: PVC vs sinus ----------------
s = np.load(f"{RES}/sinus_k12.npz", allow_pickle=True)
UP, US, PP, PS = s['UP'], s['US'], s['PP'], s['PS']
mUP, mUS = UP.mean(0), US.mean(0)
order = np.argsort(mUP)[::-1]
rs = spearmanr(mUP, mUS).correlation
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
w = 0.4; x = np.arange(12)
lp = mUP - np.percentile(UP, 2.5, 0); hp = np.percentile(UP, 97.5, 0) - mUP
ls = mUS - np.percentile(US, 2.5, 0); hs = np.percentile(US, 97.5, 0) - mUS
ax[0].bar(x - w / 2, mUP[order], w, yerr=[lp[order], hp[order]], capsize=2,
          label='PVC', color='#1f77b4')
ax[0].bar(x + w / 2, mUS[order], w, yerr=[ls[order], hs[order]], capsize=2,
          label='Sinus', color='#ff7f0e')
ax[0].set_xticks(x); ax[0].set_xticklabels([LEADS[i] for i in order])
ax[0].legend(frameon=False)
ax[0].set_ylabel('unique information (MSE)')
ax[0].set_title(f'A  Per-lead unique information (Spearman {rs:.2f})')
mPP, mPS = PP.mean(0), PS.mean(0)
ax[1].plot(range(1, 13), [mPP[SIZES == k].min() for k in range(1, 13)], 'o-',
           color='#1f77b4', label='PVC')
ax[1].plot(range(1, 13), [mPS[SIZES == k].min() for k in range(1, 13)], 's--',
           color='#ff7f0e', label='Sinus')
ax[1].set_xlabel('subset size'); ax[1].set_ylabel('full-12 reconstruction MSE')
ax[1].set_title('B  Pareto frontier'); ax[1].set_xticks(range(1, 13))
ax[1].legend(frameon=False)
save(fig, "Fig5_pvc_vs_sinus")
print("figures ->", FIG)
