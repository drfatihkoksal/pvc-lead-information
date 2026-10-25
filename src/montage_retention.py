#!/usr/bin/env python3
"""Electrode-montage retention (Sections 3.4, 3.5 and 4.1).

Retention of a montage is the share of the gap between the limb-only montage
(RA/LA/LL) and the full nine-electrode set that it closes, computed per fit and
summarised as the mean over the 25 fits with a 2.5/97.5 percentile interval.

  1. retention ladder: the PTB-XL-optimal montage at 4-8 electrodes, and
     RA/LA/LL + C2/C4/C6 on PTB-XL and on CPSC (external transfer)
  2. sinus-selected versus PVC-selected six-electrode montage, both applied to
     PVC beats; paired difference computed within fold (positive values favour
     the PVC-selected montage)

PTB-XL values use the held-out PVC errors of sinus_comparison.py (the beats with
a matched sinus reference), so that both comparisons are on the same beats;
CPSC values use the transfer errors of train_eval.py. No retraining.
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, ELEC, SUB_IX, induced_leads
from config import RESULTS

OUT = f"{RESULTS}/montage"
os.makedirs(OUT, exist_ok=True)
LIMB_E = ('RA', 'LA', 'LL')

s = np.load(f"{RES}/sinus_k12.npz", allow_pickle=True)
PP, PS = s['PP'], s['PS']                  # PVC / sinus held-out subset errors
Ec = np.load(f"{RES}/fits_k12.npz", allow_pickle=True)['Ec']

i3 = SUB_IX[induced_leads(LIMB_E)]
i9 = SUB_IX[induced_leads(tuple(ELEC))]


def ix(E):
    return SUB_IX[induced_leads(E)]


def retention(E, j):
    """Per-fit retention (%) of the montage with subset index j."""
    return 100 * (E[:, i3] - E[:, j]) / (E[:, i3] - E[:, i9])


def summary(r):
    return r.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def optimal(E, n):
    """Lowest mean-error montage of n electrodes."""
    m = E.mean(0)
    cand = [Es for Es in itertools.combinations(ELEC, n) if induced_leads(Es)]
    return min(cand, key=lambda Es: m[ix(Es)])


out = ["# Electrode-montage retention\n"]
P = out.append

# ---------- 1. retention ladder ----------
rows = []
for n in range(4, 9):
    Es = optimal(PP, n)
    m, lo, hi = summary(retention(PP, ix(Es)))
    rows.append(dict(cohort='PTB-XL', n_electrodes=n, montage='+'.join(Es),
                     retention=m, ci_lo=lo, ci_hi=hi))
C246 = LIMB_E + ('C2', 'C4', 'C6')
for coh, E in (('PTB-XL', PP), ('CPSC', Ec)):
    m, lo, hi = summary(retention(E, ix(C246)))
    rows.append(dict(cohort=coh, n_electrodes=6, montage='+'.join(C246),
                     retention=m, ci_lo=lo, ci_hi=hi))
t1 = pd.DataFrame(rows)
t1.to_csv(f"{OUT}/retention_ladder.csv", index=False)
P("## Retention ladder (% of the limb-only to nine-electrode gap closed)\n")
P(t1.round(1).to_markdown(index=False) + "\n")

# ---------- 2. sinus-selected vs PVC-selected montage ----------
e_sin, e_pvc = optimal(PS, 6), optimal(PP, 6)
r_sin, r_pvc = retention(PP, ix(e_sin)), retention(PP, ix(e_pvc))
d = r_pvc - r_sin
rows = [dict(selected_on='sinus', montage='+'.join(e_sin), **dict(
            zip(('retention_on_pvc', 'ci_lo', 'ci_hi'), summary(r_sin)))),
        dict(selected_on='PVC', montage='+'.join(e_pvc), **dict(
            zip(('retention_on_pvc', 'ci_lo', 'ci_hi'), summary(r_pvc))))]
t2 = pd.DataFrame(rows)
t2.to_csv(f"{OUT}/sinus_vs_pvc_selected.csv", index=False)
m, lo, hi = summary(d)
P("## Six-electrode montage selected on sinus vs on PVC beats, applied to PVC\n")
P(t2.round(1).to_markdown(index=False) + "\n")
P(f"- Paired difference (PVC-selected minus sinus-selected, within fold): "
  f"{m:+.2f} percentage points [95% CI {lo:+.2f}, {hi:+.2f}]\n")

txt = "\n".join(out)
open(f"{OUT}/montage_summary.md", "w").write(txt + "\n")
print(txt)
