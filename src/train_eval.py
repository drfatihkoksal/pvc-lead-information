#!/usr/bin/env python3
"""Train the reconstruction model and evaluate every lead subset (Sections
3.2-3.4).

Protocol (Methods 2.4): patient-grouped 5-fold CV x 5 seeds = 25 fits on PTB-XL.
Each fit is evaluated on (a) its held-out PTB-XL fold and (b) the full CPSC
cohort (external transfer, PTB-XL-trained model). The primary (KMAX=12) and
sensitivity (KMAX=8) models are fitted on identical folds and seeds.

    python3 train_eval.py          # both models
    python3 train_eval.py 12       # primary only

Outputs (per model, KMAX in {8,12}) -> config.FITS/fits_k{KMAX}.npz
    Ep (25,4095) subset MSE on held-out PTB-XL      Ec (25,4095) on CPSC
    Up (25,12)   per-lead unique info on PTB-XL     Uc (25,12)   on CPSC
"""
import os
import sys
import time
import json
import subprocess
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, DEV, SUBSETS, load_pvc, load_cpsc, train,
                       uniq_info, eval_all)

NSEED, NFOLD = 5, 5


def run(kmax):
    Xp, gp = load_pvc()
    Xc, _ = load_cpsc()
    Xc_g = torch.tensor(Xc).to(DEV)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(f"{RES}/models_k{kmax}", exist_ok=True)

    Ep, Ec, Up, Uc, tags = [], [], [], [], []
    t0 = time.time()
    for seed in range(NSEED):
        gkf = GroupKFold(NFOLD)
        for fold, (tr, te) in enumerate(gkf.split(Xp, groups=gp)):
            m = train(Xp[tr], seed, kmax)
            Xte = torch.tensor(Xp[te]).to(DEV)
            Up.append(uniq_info(m, Xte))
            Uc.append(uniq_info(m, Xc_g))
            Ep.append(eval_all(m, Xte))
            Ec.append(eval_all(m, Xc_g))
            tags.append((seed, fold))
            torch.save(m.state_dict(), f"{RES}/models_k{kmax}/s{seed}_f{fold}.pt")
            print(f"  k<={kmax} seed {seed} fold {fold} done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(f"{RES}/fits_k{kmax}.npz", Ep=np.array(Ep), Ec=np.array(Ec),
             Up=np.array(Up), Uc=np.array(Uc), tags=np.array(tags),
             subsets=np.array(SUBSETS, dtype=object))
    print(f"k<={kmax}: {len(Ep)} fits in {time.time()-t0:.0f}s -> "
          f"{RES}/fits_k{kmax}.npz", flush=True)
    return time.time() - t0


if __name__ == "__main__":
    kmaxes = [int(a) for a in sys.argv[1:]] or [12, 8]
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                   else "cpu"), "numpy": np.__version__,
           "nseed": NSEED, "nfold": NFOLD, "protocol": "GroupKFold(5) x seeds 0-4"}
    runtimes = {}
    for k in kmaxes:
        runtimes[f"k{k}_seconds"] = run(k)
    env.update(runtimes)
    with open(f"{RES}/run_env.json", "w") as f:
        json.dump(env, f, indent=2)
