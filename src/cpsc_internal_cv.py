#!/usr/bin/env python3
"""CPSC internal cross-validation of per-lead unique information (CPSC bars of
Fig. 2; CPSC precordial-to-limb ratio in Section 3.2), for both mask samplers.

Record-grouped 5-fold x 5 seeds on CPSC. PTB-XL internal CV and the
PTB-XL->CPSC transfer come from train_eval.py (Up / Uc).
"""
import os
import sys
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, DEV, load_cpsc, train, uniq_info

NSEED, NFOLD = 5, 5

Xc, gc = load_cpsc()
for kmax in (12, 8):
    C = []
    t0 = time.time()
    for seed in range(NSEED):
        for fold, (tr, te) in enumerate(GroupKFold(NFOLD).split(Xc, groups=gc)):
            m = train(Xc[tr], seed, kmax)
            C.append(uniq_info(m, torch.tensor(Xc[te]).to(DEV)))
            print(f"  cpsc-internal k<={kmax} seed {seed} fold {fold} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{RES}/cpsc_internal_k{kmax}.npz", C=np.array(C))
    print(f"saved -> {RES}/cpsc_internal_k{kmax}.npz", flush=True)
