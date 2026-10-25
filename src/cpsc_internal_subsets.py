#!/usr/bin/env python3
"""CPSC-trained fits storing the full 4095-subset error vector on each held-out
CPSC fold (record-grouped 5-fold x 5 seeds, KMAX=12), so that the modal optimal
subset and the within-cohort Jaccard ceiling can be computed inside CPSC exactly
as they are inside PTB-XL (selection_stability.py).
"""
import os
import sys
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, DEV, SUBSETS, load_cpsc, train, uniq_info, eval_all

NSEED, NFOLD, KMAX = 5, 5, 12

Xc, gc = load_cpsc()
os.makedirs(f"{RES}/models_cpsc_k{KMAX}", exist_ok=True)
E, U = [], []
t0 = time.time()
for seed in range(NSEED):
    for fold, (tr, te) in enumerate(GroupKFold(NFOLD).split(Xc, groups=gc)):
        m = train(Xc[tr], seed, KMAX)
        torch.save(m.state_dict(), f"{RES}/models_cpsc_k{KMAX}/s{seed}_f{fold}.pt")
        Xte = torch.tensor(Xc[te]).to(DEV)
        E.append(eval_all(m, Xte))
        U.append(uniq_info(m, Xte))
        print(f"  cpsc-internal subsets seed {seed} fold {fold} "
              f"({time.time()-t0:.0f}s)", flush=True)

np.savez(f"{RES}/cpsc_internal_subsets_k{KMAX}.npz", E=np.array(E), U=np.array(U),
         subsets=np.array(SUBSETS, dtype=object))
print(f"saved -> {RES}/cpsc_internal_subsets_k{KMAX}.npz ({time.time()-t0:.0f}s)")
