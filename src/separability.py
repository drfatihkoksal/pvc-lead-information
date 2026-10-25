#!/usr/bin/env python3
"""Is morphological separability preserved by the six-electrode montage?
(Methods 2.4, Sections 3.5 and 4.1)

Reconstruction MSE measures how faithfully a beat is recovered, not whether two
beats that differ remain distinguishable after reconstruction, so separability
is measured directly. For each seed, every PVC beat is reconstructed from
RA/LA/LL + C2/C4/C6 (nine leads) by the model of the fold in which it is held
out, giving an out-of-fold reconstruction of the whole cohort.

Within-record (reported): for each record contributing at least four beats, all
pairwise beat distances (1 - Pearson r, deliberately not the squared-error
geometry the autoencoder minimises) are computed on the true and the
reconstructed beats and compared by Spearman correlation.

Between-record (supplementary output): AUC for separating same-record from
different-record beat pairs, on the true signals and on the reconstructions.

    python3 separability.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, DEV, DATA, MAE, norm, load_pvc, induced_leads, LEADS

NSEED, NFOLD, KMAX = 5, 5, 12
MONTAGE = ['RA', 'LA', 'LL', 'C2', 'C4', 'C6']
RNG = np.random.RandomState(20260819)
MAXPAIR = 40000


def recon(m, X, keep):
    """Reconstruct X from only the leads in `keep`."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i + 512]).to(DEV)
            mask = torch.zeros_like(xb)
            mask[:, keep, :] = 1.0
            out.append(m(xb * mask).cpu().numpy())
    return np.concatenate(out)


def auc_same_record(X, rec):
    """AUC for separating same-record from different-record beat pairs."""
    n = len(X)
    F = X.reshape(n, -1)
    i = RNG.randint(0, n, MAXPAIR)
    j = RNG.randint(0, n, MAXPAIR)
    ok = i != j
    i, j = i[ok], j[ok]
    d = np.linalg.norm(F[i] - F[j], axis=1)
    y = (rec[i] == rec[j]).astype(int)
    if y.sum() < 20 or (1 - y).sum() < 20:
        return np.nan
    return roc_auc_score(y, -d)



def corr_dist(F):
    """1 - Pearson r between beats. Deliberately not Euclidean: the autoencoder
    minimises squared error in this space, so a Euclidean test would be scored in
    the very geometry the reconstruction was optimised for."""
    Z = F - F.mean(1, keepdims=True)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Z @ Z.T


def within_record(Xt, Xr, rec, min_beats=4):
    """Is *within-record* morphological structure preserved?

    Multiform ectopy is a within-record question: one patient, two ectopic foci,
    two QRS morphologies. The between-record test cannot speak to it, because it
    is solved by anatomy, electrode placement and torso geometry — patient
    identity, not beat morphology. Here, for each record with enough beats, we
    ask whether the ordering of pairwise beat distances survives reconstruction.
    Clustering was not used: with a median of two beats per record, and only 19
    records reaching six, no cluster structure can be estimated per record.
    """
    from scipy.stats import spearmanr
    rho, nb = [], []
    for r in np.unique(rec):
        m = rec == r
        k = int(m.sum())
        if k < min_beats:
            continue
        A = corr_dist(Xt[m].reshape(k, -1))
        B = corr_dist(Xr[m].reshape(k, -1))
        iu = np.triu_indices(k, 1)
        s = spearmanr(A[iu], B[iu]).statistic
        if np.isfinite(s):
            rho.append(s)
            nb.append(k)
    return np.array(rho), np.array(nb)


def auc_same_record_corr(X, rec):
    """The between-record test, scored with correlation distance."""
    n = len(X)
    F = X.reshape(n, -1)
    D = corr_dist(F)
    i = RNG.randint(0, n, MAXPAIR)
    j = RNG.randint(0, n, MAXPAIR)
    ok = i != j
    i, j = i[ok], j[ok]
    y = (rec[i] == rec[j]).astype(int)
    if y.sum() < 20:
        return np.nan
    return roc_auc_score(y, -D[i, j])


def main():
    Xp, gp = load_pvc()
    rec = pd.read_csv(f"{DATA}/pvc_meta.csv").ecg_id.values
    keep = list(induced_leads(MONTAGE))
    print(f"montage {'+'.join(MONTAGE)} induces {len(keep)} leads: "
          f"{[LEADS[k] for k in keep]}", flush=True)

    true_auc, rec_auc = [], []
    corr_true, corr_rec = [], []
    wrho, nrecs, allrho, allnb = [], [], [], []
    for seed in range(NSEED):
        # Each beat is held out exactly once per seed, so one seed yields an
        # out-of-fold reconstruction of the whole cohort. Pooling them is what
        # makes the within-record analysis possible: a single held-out fold
        # leaves only four records with enough beats.
        Xr = np.empty_like(Xp)
        for fold, (_, te) in enumerate(GroupKFold(NFOLD).split(Xp, groups=gp)):
            m = MAE().to(DEV)
            m.load_state_dict(torch.load(f"{RES}/models_k{KMAX}/s{seed}_f{fold}.pt"))
            m.eval()
            Xr[te] = recon(m, Xp[te], keep)
        a1, a2 = auc_same_record(Xp, rec), auc_same_record(Xr, rec)
        c1, c2 = auc_same_record_corr(Xp, rec), auc_same_record_corr(Xr, rec)
        wv, nb = within_record(Xp, Xr, rec)
        allnb.append(nb)
        w, nr = wv.mean(), len(wv)
        allrho.append(wv)
        true_auc.append(a1); rec_auc.append(a2)
        corr_true.append(c1); corr_rec.append(c2)
        wrho.append(w); nrecs.append(nr)
        print(f"  seed {seed}: AUC true {a1:.4f} rec {a2:.4f} | corr-AUC "
              f"{c1:.4f}/{c2:.4f} | within-record rho {w:.4f} ({nr} records)",
              flush=True)

    t, r = np.array(true_auc), np.array(rec_auc)
    ct, cr = np.array(corr_true), np.array(corr_rec)
    wr = np.array(wrho)

    def ci(a, k=3):
        return (f"{a.mean():.{k}f} [{np.percentile(a, 2.5):.{k}f}, "
                f"{np.percentile(a, 97.5):.{k}f}]")

    print("\n--- between-record (identity), Euclidean distance")
    print("  true 12-lead   :", ci(t))
    print("  reconstructed  :", ci(r))
    print("  difference     :", ci(t - r), "(positive = loss after reduction)")
    print("--- between-record, correlation distance (not the trained geometry)")
    print("  true 12-lead   :", ci(ct))
    print("  reconstructed  :", ci(cr))
    print("  difference     :", ci(ct - cr), "(positive = loss after reduction)")
    print("--- within-record (morphology), correlation distance")
    print(f"  distance-rank preservation, Spearman: {ci(wr)}")
    pool = np.concatenate(allrho)
    print(f"  records contributing: {np.mean(nrecs):.0f} "
          f"(those with >=4 beats)")
    print(f"  across records: median {np.median(pool):.3f}, "
          f"IQR {np.percentile(pool,25):.3f}-{np.percentile(pool,75):.3f}, "
          f"2.5-97.5 pct {np.percentile(pool,2.5):.3f}-{np.percentile(pool,97.5):.3f}")
    print(f"  records with rho > 0.8: {100*(pool>0.8).mean():.1f}%")
    nb = np.concatenate(allnb)
    from scipy.stats import spearmanr
    six = pool[nb >= 6]
    print(f"  restricted to records with >=6 beats (n={len(six)//NSEED} per seed): "
          f"median {np.median(six):.3f}, IQR {np.percentile(six,25):.3f}-"
          f"{np.percentile(six,75):.3f}")
    print(f"  rho vs beat count, Spearman: {spearmanr(pool, nb).statistic:+.3f}")
    print(f"  share of records with exactly 4 beats: {100*(nb==4).mean():.0f}%")

    pd.DataFrame({"seed": range(len(t)), "auc_true": t, "auc_recon": r,
                  "auc_true_corr": ct, "auc_recon_corr": cr,
                  "within_rho": wr, "n_records": nrecs}
                 ).to_csv(f"{RES}/separability.csv", index=False)
    print(f"saved -> {RES}/separability.csv")


if __name__ == "__main__":
    main()
