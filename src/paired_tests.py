#!/usr/bin/env python3
"""Paired PVC-versus-sinus tests (Methods 2.4, Section 3.5, Supplementary
Table S1).

Pairing is at the beat level: every PVC beat has a contemporaneous sinus beat
from the same record, and both are evaluated on the fold in which that beat is
held out, under identical splits. Each beat is held out once per seed, so the
five per-seed values are averaged before testing. Beats whose record yielded no
sinus reference are excluded.

  1. per-lead unique information, PVC vs sinus, 12 paired tests, Holm-adjusted
  2. reconstruction error at four clinically relevant configurations,
     Holm-adjusted
  3. precordial-to-limb ratio, paired bootstrap over patients
  4. bootstrap CI for the PVC-vs-sinus Spearman correlation

A global Bonferroni adjustment across the 16 tests of 1 and 2 is reported as a
sensitivity analysis. Reuses the saved KMAX=12 fits; no retraining.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, DEV, MAE, LEADS, SUBSETS, SIZES, SUB_IX, PREC, LIMB,
                    load_pvc, load_sinus, induced_leads)
from config import RESULTS

OUT = f"{RESULTS}/paired_tests"
os.makedirs(OUT, exist_ok=True)
NSEED, NFOLD, KMAX = 5, 5, 12
NBOOT = 2000
RNG = np.random.RandomState(42)

CONFIGS = {
    'best 3-lead subset (III+V2+V5)': ('III', 'V2', 'V5'),
    'best 6-lead subset (I+aVF+V2+V3+V4+V5)': ('I', 'aVF', 'V2', 'V3', 'V4', 'V5'),
    'six-electrode montage (RA/LA/LL+C2/C4/C6)':
        tuple(LEADS[i] for i in induced_leads(('RA', 'LA', 'LL', 'C2', 'C4', 'C6'))),
    'all 12 leads': tuple(LEADS),
}


@torch.no_grad()
def per_beat(m, Xg):
    """(n,12) unique-information error per lead, and (n,) error per config."""
    uniq = np.empty((Xg.shape[0], 12))
    for L in range(12):
        msk = torch.ones(1, 12, 1, device=DEV)
        msk[0, L] = 0
        out = m(Xg * msk)
        uniq[:, L] = ((out[:, L] - Xg[:, L]) ** 2).mean(dim=1).cpu().numpy()
    cfg = {}
    for name, leads in CONFIGS.items():
        msk = torch.zeros(1, 12, 1, device=DEV)
        msk[0, [LEADS.index(l) for l in leads]] = 1
        out = m(Xg * msk)
        cfg[name] = ((out - Xg) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return uniq, cfg


Xp, gp = load_pvc()
Xs = load_sinus()
OK = np.isfinite(Xs).all(axis=(1, 2))   # PVCs whose record yielded no sinus reference
n = len(Xp)
UP = np.zeros((n, 12)); US = np.zeros((n, 12))
CP = {k: np.zeros(n) for k in CONFIGS}; CS = {k: np.zeros(n) for k in CONFIGS}

for seed in range(NSEED):
    for fold, (tr, te) in enumerate(GroupKFold(NFOLD).split(Xp, groups=gp)):
        te = te[OK[te]]
        mp, ms = MAE().to(DEV), MAE().to(DEV)
        mp.load_state_dict(torch.load(f"{RES}/models_k{KMAX}/s{seed}_f{fold}.pt"))
        ms.load_state_dict(torch.load(f"{RES}/models_sinus_k{KMAX}/s{seed}_f{fold}.pt"))
        mp.eval(); ms.eval()
        up, cp = per_beat(mp, torch.tensor(Xp[te]).to(DEV))
        us, cs = per_beat(ms, torch.tensor(Xs[te]).to(DEV))
        UP[te] += up / NSEED; US[te] += us / NSEED
        for k in CONFIGS:
            CP[k][te] += cp[k] / NSEED
            CS[k][te] += cs[k] / NSEED
    print(f"  seed {seed} done", flush=True)

# sinus reference beats are NaN where no clean reference existed
valid = np.isfinite(US).all(1) & np.isfinite(Xs).all((1, 2))
print(f"paired beats: {valid.sum()} of {n}")
UP, US = UP[valid], US[valid]
CPv = {k: CP[k][valid] for k in CONFIGS}
CSv = {k: CS[k][valid] for k in CONFIGS}
pid = gp[valid]
np.savez(f"{OUT}/per_beat.npz", UP=UP, US=US, pid=pid,
         **{f"cfg_pvc_{i}": CPv[k] for i, k in enumerate(CONFIGS)},
         **{f"cfg_sin_{i}": CSv[k] for i, k in enumerate(CONFIGS)})


def median_diff_ci(d, nboot=NBOOT):
    """Median of the paired differences with a bootstrap percentile CI.

    Reported alongside the signed-rank p and the rank-biserial correlation. Note
    that a signed-rank test ranks the magnitudes of the differences, so it can
    reach significance while the CI of the median difference includes zero; the
    two answer different questions and both are given."""
    hl = float(np.median(d))
    idx = RNG.randint(0, len(d), (nboot, len(d)))
    bs = np.median(d[idx], axis=1)
    return hl, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rank_biserial(d):
    r = pd.Series(np.abs(d)).rank().values
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return (rp - rm) / (rp + rm)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


out = ["# Paired PVC-versus-sinus tests\n",
       f"Beat-level pairing, n = {len(UP)} matched PVC/sinus pairs "
       f"(same record, identical splits, averaged over {NSEED} seeds).\n"]
P = out.append

# ---- 1. per-lead unique information ----
rows = []
for L in range(12):
    d = UP[:, L] - US[:, L]
    md, lo, hi = median_diff_ci(d)
    st = wilcoxon(UP[:, L], US[:, L])
    rows.append(dict(lead=LEADS[L], pvc=UP[:, L].mean(), sinus=US[:, L].mean(),
                     mean_ratio=UP[:, L].mean() / US[:, L].mean(),
                     median_diff=md, md_lo=lo, md_hi=hi,
                     rank_biserial=rank_biserial(d), p=st.pvalue))
t1 = pd.DataFrame(rows)
t1['p_holm'] = holm(t1.p.values)
P("## 1. Per-lead unique information, PVC vs sinus\n")
P(t1.round(4).to_markdown(index=False) + "\n")
P(f"- Significant after Holm correction: "
  f"{int((t1.p_holm<0.05).sum())}/12 leads; PVC error higher in "
  f"{int((t1.median_diff>0).sum())}/12.\n")

# ---- 2. configurations ----
rows = []
for k in CONFIGS:
    d = CPv[k] - CSv[k]
    md, lo, hi = median_diff_ci(d)
    st = wilcoxon(CPv[k], CSv[k])
    rows.append(dict(configuration=k, pvc=CPv[k].mean(), sinus=CSv[k].mean(),
                     mean_ratio=CPv[k].mean() / CSv[k].mean(), median_diff=md,
                     md_lo=lo, md_hi=hi, rank_biserial=rank_biserial(d),
                     p=st.pvalue))
t2 = pd.DataFrame(rows)
t2['p_holm'] = holm(t2.p.values)
# global Bonferroni across both families (12 + 4 tests), sensitivity analysis
n_tests = len(t1) + len(t2)
for t in (t1, t2):
    t['p_bonferroni_global'] = np.minimum(t.p.values * n_tests, 1.0)
t1.to_csv(f"{OUT}/per_lead_tests.csv", index=False)
t2.to_csv(f"{OUT}/configuration_tests.csv", index=False)
P("## 2. Reconstruction error at clinically relevant configurations\n")
P(t2.round(4).to_markdown(index=False) + "\n")

# ---- 3/4. patient-level bootstrap ----
pats = np.unique(pid)
groups = {p: np.where(pid == p)[0] for p in pats}
ratios_p, ratios_s, rhos = [], [], []
for _ in range(NBOOT):
    take = np.concatenate([groups[p] for p in RNG.choice(pats, len(pats), True)])
    mp_, ms_ = UP[take].mean(0), US[take].mean(0)
    ratios_p.append(mp_[PREC].mean() / mp_[LIMB].mean())
    ratios_s.append(ms_[PREC].mean() / ms_[LIMB].mean())
    rhos.append(spearmanr(mp_, ms_).correlation)
ratios_p, ratios_s, rhos = map(np.array, (ratios_p, ratios_s, rhos))
diff = ratios_p - ratios_s
mP, mS = UP.mean(0), US.mean(0)
P("## 3. Precordial-to-limb ratio (paired bootstrap over patients, "
  f"{NBOOT} resamples)\n")
P(f"- PVC {mP[PREC].mean()/mP[LIMB].mean():.1f}x "
  f"[{np.percentile(ratios_p,2.5):.1f}, {np.percentile(ratios_p,97.5):.1f}]; "
  f"sinus {mS[PREC].mean()/mS[LIMB].mean():.1f}x "
  f"[{np.percentile(ratios_s,2.5):.1f}, {np.percentile(ratios_s,97.5):.1f}]")
P(f"- Paired difference {diff.mean():+.2f} "
  f"[{np.percentile(diff,2.5):+.2f}, {np.percentile(diff,97.5):+.2f}]\n")
P("## 4. PVC-vs-sinus Spearman correlation with bootstrap CI\n")
P(f"- Spearman {spearmanr(mP, mS).correlation:.3f} "
  f"[{np.percentile(rhos,2.5):.3f}, {np.percentile(rhos,97.5):.3f}]\n")
np.savez(f"{OUT}/bootstrap.npz", ratios_p=ratios_p, ratios_s=ratios_s, rhos=rhos)

txt = "\n".join(out)
open(f"{OUT}/paired_tests_summary.md", "w").write(txt + "\n")
print(txt)
