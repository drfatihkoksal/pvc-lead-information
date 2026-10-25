#!/usr/bin/env python3
"""PVC versus matched sinus beats (Fig. 5, Section 3.5).

Matched beats from the same records under identical patient-grouped folds
(5 folds x 5 seeds, KMAX=12). The PVC-side fits are the ones saved by
train_eval.py (identical seed/fold/data/sampler); only the sinus-side models are
trained here.
"""
import os
import sys
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, DEV, MAE, SUBSETS, load_pvc, load_sinus, train,
                       uniq_info, eval_all)

NSEED, NFOLD, KMAX = 5, 5, 12

Xp, gp = load_pvc()
Xs = load_sinus()
# A PVC keeps its slot even when no clean sinus reference existed in that record;
# those slots are NaN. Training on them drives the loss to NaN and silently
# destroys the model weights, so they are excluded from the paired analysis.
OK = np.isfinite(Xs).all(axis=(1, 2))
print(f"beats with a usable sinus reference: {OK.sum()}/{len(Xs)}", flush=True)
os.makedirs(f"{RES}/models_sinus_k{KMAX}", exist_ok=True)

UP, US, PP, PS = [], [], [], []
t0 = time.time()
for seed in range(NSEED):
    for fold, (tr, te) in enumerate(GroupKFold(NFOLD).split(Xp, groups=gp)):
        tr, te = tr[OK[tr]], te[OK[te]]
        Xte_p = torch.tensor(Xp[te]).to(DEV)
        Xte_s = torch.tensor(Xs[te]).to(DEV)
        mp = MAE().to(DEV)
        mp.load_state_dict(torch.load(f"{RES}/models_k{KMAX}/s{seed}_f{fold}.pt"))
        mp.eval()
        ms = train(Xs[tr], seed, KMAX)
        torch.save(ms.state_dict(), f"{RES}/models_sinus_k{KMAX}/s{seed}_f{fold}.pt")
        UP.append(uniq_info(mp, Xte_p))
        US.append(uniq_info(ms, Xte_s))
        PP.append(eval_all(mp, Xte_p))
        PS.append(eval_all(ms, Xte_s))
        print(f"  sinus seed {seed} fold {fold} done ({time.time()-t0:.0f}s)",
              flush=True)

np.savez(f"{RES}/sinus_k{KMAX}.npz", UP=np.array(UP), US=np.array(US),
         PP=np.array(PP), PS=np.array(PS),
         subsets=np.array(SUBSETS, dtype=object))
print(f"saved -> {RES}/sinus_k{KMAX}.npz ({time.time()-t0:.0f}s)")
