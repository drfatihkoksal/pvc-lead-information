#!/usr/bin/env python3
"""Is a low cross-cohort Jaccard cohort disagreement, or a flat optimum?
(Section 3.3)

1. flatness of the optimum: rank and distance of the PTB-XL-optimal subset among
   all subsets of its size, evaluated on CPSC
2. within-cohort selection stability across the 25 fits, the ceiling against
   which the between-cohort Jaccard should be read
3. plane-level (anatomical) agreement of the optimal subsets

Outputs under config.RESULTS/stability.
"""
import os
import sys
import itertools
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, LEADS, ELEC, SUBSETS, SIZES, SUB_IX, induced_leads)
from config import RESULTS

OUT = f"{RESULTS}/stability"
os.makedirs(OUT, exist_ok=True)

PLANE = {}
for l in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']:
    PLANE[l] = 'frontal'
for l in ['V1', 'V2']:
    PLANE[l] = 'septal-anterior'
for l in ['V3', 'V4']:
    PLANE[l] = 'anterior'
for l in ['V5', 'V6']:
    PLANE[l] = 'lateral'
PLANE_OF = [PLANE[l] for l in LEADS]
CLASSES = ['frontal', 'septal-anterior', 'anterior', 'lateral']


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def name(sub):
    return '+'.join(LEADS[i] for i in sub)


def plane_vec(sub):
    c = Counter(PLANE_OF[i] for i in sub)
    return tuple(c[k] for k in CLASSES)


def best_per_size(m):
    return {k: int(np.where(SIZES == k)[0][np.argmin(m[SIZES == k])])
            for k in range(1, 13)}


def elec_sets():
    return [(E, SUB_IX[induced_leads(E)]) for r in range(1, 10)
            for E in itertools.combinations(ELEC, r) if induced_leads(E)]


d = np.load(f"{RES}/fits_k12.npz", allow_pickle=True)
Ep, Ec = d['Ep'], d['Ec']
mEp, mEc = Ep.mean(0), Ec.mean(0)
fp, fc = best_per_size(mEp), best_per_size(mEc)
rec = elec_sets()

out = ["# Flatness and selection stability of the optimal subsets\n"]
P = out.append

# ---------------- flatness ----------------
P("## Flatness of the optimum (lead subsets, evaluated on CPSC)\n")
rows = []
for k in range(1, 13):
    ix = np.where(SIZES == k)[0]
    bp, bc = fp[k], fc[k]
    vals = mEc[ix]
    opt = vals.min()
    rank = int((vals < mEc[bp]).sum()) + 1
    rows.append(dict(
        size=k, ptbxl_optimal=name(SUBSETS[bp]), cpsc_optimal=name(SUBSETS[bc]),
        jaccard=jac(SUBSETS[bp], SUBSETS[bc]),
        mse_cpsc_of_ptbxl_opt=mEc[bp], mse_cpsc_optimal=opt,
        delta_abs=mEc[bp] - opt, delta_pct=100 * (mEc[bp] - opt) / opt,
        rank_of_ptbxl_opt=rank, n_subsets=len(ix),
        percentile=100 * (1 - (rank - 1) / len(ix)),
        n_within_1pct=int((vals <= opt * 1.01).sum()),
        n_within_5pct=int((vals <= opt * 1.05).sum())))
t1 = pd.DataFrame(rows)
t1.to_csv(f"{OUT}/flatness_leads.csv", index=False)
P(t1.round(4).to_markdown(index=False) + "\n")

P("## Flatness of the optimum (electrode subsets, evaluated on CPSC)\n")
rows = []
for n in range(3, 10):
    cand = [r for r in rec if len(r[0]) == n]
    vals = np.array([mEc[i] for _, i in cand])
    bp = min(cand, key=lambda r: mEp[r[1]])
    bc = min(cand, key=lambda r: mEc[r[1]])
    opt = vals.min()
    rank = int((vals < mEc[bp[1]]).sum()) + 1
    rows.append(dict(
        n_electrodes=n, ptbxl_optimal='+'.join(bp[0]),
        cpsc_optimal='+'.join(bc[0]), jaccard=jac(bp[0], bc[0]),
        mse_cpsc_of_ptbxl_opt=mEc[bp[1]], mse_cpsc_optimal=opt,
        delta_pct=100 * (mEc[bp[1]] - opt) / opt,
        rank_of_ptbxl_opt=rank, n_sets=len(cand),
        percentile=100 * (1 - (rank - 1) / len(cand)),
        n_within_1pct=int((vals <= opt * 1.01).sum()),
        n_within_5pct=int((vals <= opt * 1.05).sum())))
t2 = pd.DataFrame(rows)
t2.to_csv(f"{OUT}/flatness_electrodes.csv", index=False)
P(t2.round(4).to_markdown(index=False) + "\n")

# ---------------- within-cohort stability ----------------
def stability(E, label):
    rows = []
    for k in range(1, 13):
        ix = np.where(SIZES == k)[0]
        picks = [tuple(SUBSETS[ix[np.argmin(E[f][ix])]]) for f in range(E.shape[0])]
        cnt = Counter(picks)
        modal, freq = cnt.most_common(1)[0]
        pw = [jac(picks[i], picks[j]) for i in range(len(picks))
              for j in range(i + 1, len(picks))]
        rows.append(dict(size=k, cohort=label, modal_subset=name(modal),
                         modal_freq=freq / len(picks), n_distinct=len(cnt),
                         mean_pairwise_jaccard=float(np.mean(pw))))
    return pd.DataFrame(rows)


P("## Within-cohort selection stability across the 25 fits\n")
s_p = stability(Ep, 'PTB-XL (internal)')
s_t = stability(Ec, 'CPSC (transfer of the same models)')
frames = [s_p, s_t]
if os.path.exists(f"{RES}/cpsc_internal_subsets_k12.npz"):
    Ei = np.load(f"{RES}/cpsc_internal_subsets_k12.npz", allow_pickle=True)['E']
    frames.append(stability(Ei, 'CPSC (internal fits)'))
st = pd.concat(frames)
st.to_csv(f"{OUT}/within_cohort_stability.csv", index=False)
P(st.round(3).to_markdown(index=False) + "\n")

P("### Between-cohort Jaccard read against the within-cohort ceiling\n")
rows = []
for k in range(1, 13):
    within_p = s_p[s_p['size'] == k].mean_pairwise_jaccard.iloc[0]
    within_c = s_t[s_t['size'] == k].mean_pairwise_jaccard.iloc[0]
    rows.append(dict(size=k, between_cohort=jac(SUBSETS[fp[k]], SUBSETS[fc[k]]),
                     within_ptbxl=within_p, within_cpsc=within_c,
                     ceiling=min(within_p, within_c)))
t3 = pd.DataFrame(rows)
t3.to_csv(f"{OUT}/jaccard_vs_ceiling.csv", index=False)
P(t3.round(3).to_markdown(index=False) + "\n")

# ---------------- plane-level agreement ----------------
P("## Plane-level (anatomical) agreement\n")
rows = []
for k in range(1, 13):
    bp, bc = SUBSETS[fp[k]], SUBSETS[fc[k]]
    vp, vc = plane_vec(bp), plane_vec(bc)
    swapped = [(LEADS[i], LEADS[j]) for i in bp if i not in bc
               for j in bc if j not in bp and PLANE_OF[i] == PLANE_OF[j]]
    rows.append(dict(size=k, ptbxl=name(bp), cpsc=name(bc),
                     ptbxl_planes='/'.join(f"{c}:{n}" for c, n in zip(CLASSES, vp) if n),
                     cpsc_planes='/'.join(f"{c}:{n}" for c, n in zip(CLASSES, vc) if n),
                     plane_profile_identical=int(vp == vc),
                     within_plane_swaps='; '.join(f"{a}->{b}" for a, b in swapped)))
t4 = pd.DataFrame(rows)
t4.to_csv(f"{OUT}/plane_agreement.csv", index=False)
P(t4.to_markdown(index=False) + "\n")

n_ident = int(t4[t4['size'].between(1, 6)].plane_profile_identical.sum())
P(f"- Identical plane profile at {n_ident}/6 sizes 1-6, versus "
  f"{int((t3[t3['size'].between(1,6)].between_cohort == 1).sum())}/6 with "
  f"identical subset identity.\n")

txt = "\n".join(out)
open(f"{OUT}/stability_summary.md", "w").write(txt + "\n")
print(txt)
