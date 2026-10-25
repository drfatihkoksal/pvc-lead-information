#!/usr/bin/env python3
"""Pareto frontiers, unique information and sensitivity analysis (Table 2,
Sections 3.2-3.4, Methods 2.2).

Reads the fits saved by train_eval.py (and, if present, cpsc_internal_cv.py) and
emits, under config.FITS:
  pareto_leads.csv        Table 2 Panel A (KMAX=12)
  pareto_electrodes.csv   Table 2 Panel B (KMAX=12)
  unique_info.csv         Fig. 2 numbers, both cohorts, both models
  sensitivity_k8.csv      Pareto agreement of the KMAX=8 model with KMAX=12
  rank_control.csv        algebraic rank of each optimal subset
  wilcoxon_9v12.csv       best-8 / best-9 subsets versus all 12 leads
  summary.md              headline numbers
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, LEADS, ELEC, SUBSETS, SIZES, SUB_IX, PREC, LIMB,
                    induced_leads, subset_rank)

ALL12 = SUB_IX[tuple(range(12))]


def load(kmax):
    d = np.load(f"{RES}/fits_k{kmax}.npz", allow_pickle=True)
    return {k: d[k] for k in ('Ep', 'Ec', 'Up', 'Uc')}


def ci(a, axis=0):
    return np.percentile(a, 2.5, axis=axis), np.percentile(a, 97.5, axis=axis)


def hodges_lehmann(d):
    """HL median difference + distribution-free 95% CI from Walsh averages."""
    n = len(d)
    w = np.sort(np.array([(d[i] + d[j]) / 2
                          for i in range(n) for j in range(i, n)]))
    m = len(w)
    crit = n * (n + 1) / 4 - 1.959964 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    k = max(int(np.floor(crit)), 0)
    return float(np.median(w)), float(w[k]), float(w[m - 1 - k])


def rank_biserial(d):
    st = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
    r = np.abs(d)
    ranks = pd.Series(r).rank().values
    rp = ranks[d > 0].sum()
    rm = ranks[d < 0].sum()
    return (rp - rm) / (rp + rm), st.pvalue


def pareto(mE):
    """Best subset index per size 1..12."""
    return {k: int(np.where(SIZES == k)[0][np.argmin(mE[SIZES == k])])
            for k in range(1, 13)}


def elec_records(mEp, mEc):
    rec = []
    for r in range(1, 10):
        for E in itertools.combinations(ELEC, r):
            lv = induced_leads(E)
            if not lv:
                continue
            i = SUB_IX[lv]
            rec.append((E, lv, mEp[i], mEc[i], i))
    return rec


def main():
    prim, sens = load(12), load(8)
    out = []
    P = out.append

    mEp, mEc = prim['Ep'].mean(0), prim['Ec'].mean(0)
    sEp, sEc = sens['Ep'].mean(0), sens['Ec'].mean(0)
    fr_prim, fr_sens = pareto(mEp), pareto(sEp)
    fr_prim_c = pareto(mEc)

    P("# Pareto frontiers, unique information and sensitivity analysis\n")
    P(f"Fits: {prim['Ep'].shape[0]} (5 patient-grouped folds x 5 seeds), "
      "PTB-XL trained; CPSC = external transfer of the same models.\n")

    # ---------- Table 2 Panel A ----------
    rows = []
    for k in range(1, 13):
        b = fr_prim[k]
        lo, hi = ci(prim['Ep'][:, b])
        bc = fr_prim_c[k]
        jac = len(set(SUBSETS[b]) & set(SUBSETS[bc])) / len(set(SUBSETS[b]) | set(SUBSETS[bc]))
        rows.append(dict(size=k,
                         subset_ptbxl='+'.join(LEADS[i] for i in SUBSETS[b]),
                         mse_ptbxl=mEp[b], ci_lo=lo, ci_hi=hi,
                         mse_cpsc_same_subset=mEc[b],
                         subset_cpsc='+'.join(LEADS[i] for i in SUBSETS[bc]),
                         mse_cpsc_optimal=mEc[bc], jaccard=jac,
                         rank=subset_rank(SUBSETS[b])))
    pa = pd.DataFrame(rows)
    pa.to_csv(f"{RES}/pareto_leads.csv", index=False)
    P("## Table 2 Panel A — lead subsets\n")
    P(pa.round(4).to_markdown(index=False) + "\n")

    # ---------- Table 2 Panel B ----------
    rec = elec_records(mEp, mEc)
    rows = []
    best_el = {}
    for n in range(3, 10):
        cand = [r for r in rec if len(r[0]) == n]
        b = min(cand, key=lambda r: r[2])
        bc = min(cand, key=lambda r: r[3])
        best_el[n] = b
        lo, hi = ci(prim['Ep'][:, b[4]])
        jac = len(set(b[0]) & set(bc[0])) / len(set(b[0]) | set(bc[0]))
        rows.append(dict(n_electrodes=n, set_ptbxl='+'.join(b[0]),
                         leads='+'.join(LEADS[j] for j in b[1]),
                         mse_ptbxl=b[2], ci_lo=lo, ci_hi=hi,
                         mse_cpsc_same_set=b[3], set_cpsc='+'.join(bc[0]),
                         mse_cpsc_optimal=bc[3], jaccard=jac))
    pb = pd.DataFrame(rows)
    pb.to_csv(f"{RES}/pareto_electrodes.csv", index=False)
    P("## Table 2 Panel B — electrode subsets\n")
    P(pb.round(4).to_markdown(index=False) + "\n")

    # ---------- six-electrode retention ----------
    i3 = SUB_IX[induced_leads(('RA', 'LA', 'LL'))]
    i9 = SUB_IX[induced_leads(tuple(ELEC))]
    i6 = SUB_IX[induced_leads(best_el[6][0])]
    ret_fits = 100 * (prim['Ep'][:, i3] - prim['Ep'][:, i6]) / (prim['Ep'][:, i3] - prim['Ep'][:, i9])
    ret_mean = 100 * (mEp[i3] - mEp[i6]) / (mEp[i3] - mEp[i9])
    ret_lo, ret_hi = ci(ret_fits)
    s6set = min([r for r in elec_records(sEp, sEc) if len(r[0]) == 6],
                key=lambda r: r[2])[0]
    s6 = SUB_IX[induced_leads(s6set)]
    ret_sens = 100 * (sEp[i3] - sEp[s6]) / (sEp[i3] - sEp[i9])
    P("## Six-electrode retention (gap between 3-electrode and 9-electrode montage closed)\n")
    P(f"- KMAX=12 (primary): **{ret_mean:.1f}%** [95% CI {ret_lo:.1f}-{ret_hi:.1f}] "
      f"with {'+'.join(best_el[6][0])}")
    P(f"- KMAX=8  (sensitivity): {ret_sens:.1f}%\n")

    # ---------- unique information ----------
    cols = {'lead': LEADS,
            'ptbxl_k12': prim['Up'].mean(0), 'ptbxl_k12_lo': ci(prim['Up'])[0],
            'ptbxl_k12_hi': ci(prim['Up'])[1],
            'transfer_k12': prim['Uc'].mean(0),
            'ptbxl_k8': sens['Up'].mean(0), 'transfer_k8': sens['Uc'].mean(0)}
    Cprim = Csens = None
    if os.path.exists(f"{RES}/cpsc_internal_k12.npz"):
        Cprim = np.load(f"{RES}/cpsc_internal_k12.npz")['C']
        Csens = np.load(f"{RES}/cpsc_internal_k8.npz")['C']
        cols.update(cpsc_int_k12=Cprim.mean(0), cpsc_int_k12_lo=ci(Cprim)[0],
                    cpsc_int_k12_hi=ci(Cprim)[1], cpsc_int_k8=Csens.mean(0))
    ui = pd.DataFrame(cols)
    ui.to_csv(f"{RES}/unique_info.csv", index=False)
    rho_p = spearmanr(ui.ptbxl_k12, ui.ptbxl_k8).correlation
    rho_t = spearmanr(ui.transfer_k12, ui.transfer_k8).correlation
    rho_transfer_prim = spearmanr(ui.ptbxl_k12, ui.transfer_k12).correlation
    rho_transfer_sens = spearmanr(ui.ptbxl_k8, ui.transfer_k8).correlation
    rho_fits = np.array([spearmanr(prim['Up'][i], prim['Uc'][i]).correlation
                         for i in range(prim['Up'].shape[0])])
    pl_prim = ui.ptbxl_k12[PREC].mean() / ui.ptbxl_k12[LIMB].mean()
    P("## Per-lead unique information\n")
    P(ui.round(4).to_markdown(index=False) + "\n")
    P(f"- Spearman KMAX=8 vs KMAX=12: PTB-XL {rho_p:.3f}, CPSC-transfer {rho_t:.3f}")
    P(f"- Precordial:limb ratio (KMAX=12): PTB-XL {pl_prim:.1f}x" +
      (f", CPSC internal CV {ui.cpsc_int_k12[PREC].mean()/ui.cpsc_int_k12[LIMB].mean():.1f}x"
       if Cprim is not None else ""))
    if Cprim is not None:
        rho_c = spearmanr(ui.cpsc_int_k12, ui.cpsc_int_k8).correlation
        P(f"- Spearman KMAX=8 vs KMAX=12, CPSC internal CV: {rho_c:.3f}")
    P(f"- PTB-XL->CPSC transfer Spearman: KMAX=12 {rho_transfer_prim:.3f} "
      f"[95% CI over fits {np.percentile(rho_fits,2.5):.3f}, "
      f"{np.percentile(rho_fits,97.5):.3f}], KMAX=8 {rho_transfer_sens:.3f}")
    top6 = ((-prim['Up']).argsort(1).argsort(1) < 6).mean(0)
    P(f"- Precordial leads in the top six across fits: "
      f"{100*top6[PREC].min():.0f}-{100*top6[PREC].max():.0f}%; "
      f"limb leads {100*top6[LIMB].min():.0f}-{100*top6[LIMB].max():.0f}%\n")

    # ---------- sensitivity model: Pareto agreement ----------
    rows = []
    for k in range(1, 13):
        bp, bs = fr_prim[k], fr_sens[k]
        sp, ss = set(SUBSETS[bp]), set(SUBSETS[bs])
        rows.append(dict(size=k,
                         subset_k8='+'.join(LEADS[i] for i in SUBSETS[bs]),
                         subset_k12='+'.join(LEADS[i] for i in SUBSETS[bp]),
                         jaccard=len(sp & ss) / len(sp | ss),
                         mse_k8_ptbxl=sEp[bs], mse_k12_ptbxl=mEp[bp],
                         mse_k8_on_k12_subset=sEp[bp],
                         identical=int(sp == ss)))
    sv = pd.DataFrame(rows)
    sv.to_csv(f"{RES}/sensitivity_k8.csv", index=False)
    P("## Sensitivity model (KMAX=8) vs primary (KMAX=12): Pareto agreement\n")
    P(sv.round(4).to_markdown(index=False) + "\n")
    P(f"- Identical optimal subset at {int(sv.identical.sum())}/12 sizes; "
      f"mean Jaccard {sv.jaccard.mean():.3f}\n")

    # ---------- rank control ----------
    rows = []
    for k in range(1, 13):
        b = fr_prim[k]
        rows.append(dict(size=k, subset='+'.join(LEADS[i] for i in SUBSETS[b]),
                         rank=subset_rank(SUBSETS[b]), mse_ptbxl=mEp[b],
                         max_rank_at_size=max(subset_rank(SUBSETS[i])
                                              for i in np.where(SIZES == k)[0])))
    rc = pd.DataFrame(rows)
    rc.to_csv(f"{RES}/rank_control.csv", index=False)
    P("## Algebraic rank of the optimal subsets\n")
    P(rc.round(4).to_markdown(index=False) + "\n")
    r8 = rc[rc['size'] == 8].iloc[0]
    n_full8 = sum(subset_rank(SUBSETS[i]) == 8 for i in np.where(SIZES == 8)[0])
    P(f"- Optimal 8-lead subset {r8.subset} has rank {r8['rank']} "
      f"(full rank = 8); {n_full8} of {int((SIZES == 8).sum())} eight-lead "
      f"subsets reach full rank.")
    P(f"- MSE at sizes 8..12: " +
      ", ".join(f"{k}:{mEp[fr_prim[k]]:.4f}" for k in range(8, 13)) + "\n")

    # ---------- best-8 / best-9 vs all 12 ----------
    rows = []
    for coh, E in (('PTB-XL', prim['Ep']), ('CPSC', prim['Ec'])):
        mE = E.mean(0)
        fr = pareto(mE)
        for k in (8, 9):
            b = fr[k]
            d = E[:, b] - E[:, ALL12]          # negative = subset better
            hl, lo, hi = hodges_lehmann(d)
            rb, p = rank_biserial(d)
            rows.append(dict(cohort=coh, comparison=f"best-{k} vs all-12",
                             subset='+'.join(LEADS[i] for i in SUBSETS[b]),
                             mse_subset=mE[b], mse_all12=mE[ALL12],
                             hl_diff=hl, hl_lo=lo, hl_hi=hi,
                             rank_biserial=rb, p=p,
                             pct_of_all12=100 * mE[b] / mE[ALL12]))
    wx = pd.DataFrame(rows)
    wx.to_csv(f"{RES}/wilcoxon_9v12.csv", index=False)
    P("## Paired comparison of best-8 / best-9 subsets vs all 12 leads\n")
    P(wx.round(5).to_markdown(index=False) + "\n")
    P("(negative HL difference = the subset reconstructs better than all 12 "
      "leads; subsets of full algebraic rank are equivalent in the signal, so a "
      "difference here reflects the reconstruction model)\n")

    txt = "\n".join(out)
    with open(f"{RES}/summary.md", "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
