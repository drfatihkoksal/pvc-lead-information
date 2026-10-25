#!/usr/bin/env python3
"""Shared data loading, reconstruction model, training and evaluation.

Masked convolutional autoencoder (Methods 2.2): three 1-D convolutional encoder
blocks (48/96/96 channels, kernels 7/5/5, batch-norm + ReLU) and a 1x1
convolutional decoder to 12 leads; 150 epochs, Adam 1e-3, batch 256, MSE loss;
each beat normalised to unit SD across all leads jointly.

During training each beat keeps a random subset of k leads, k drawn uniformly
from 1..KMAX and the leads uniformly at random, the rest zeroed:
    KMAX=12 -> k ~ Uniform{1..12}  (primary model)
    KMAX=8  -> k ~ Uniform{1..8}   (sensitivity model)
"""
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import BEATS as DATA, FITS as RES, DEV

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ELEC = ['RA', 'LA', 'LL', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
LIX = {l: i for i, l in enumerate(LEADS)}
PREC = [6, 7, 8, 9, 10, 11]
LIMB = [0, 1, 2, 3, 4, 5]

SUBSETS = [s for r in range(1, 13) for s in itertools.combinations(range(12), r)]
SIZES = np.array([len(s) for s in SUBSETS])
SUB_IX = {s: i for i, s in enumerate(SUBSETS)}


def norm(X):
    Y = np.empty_like(X)
    for i in range(len(X)):
        Y[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-6)
    return Y


def load_pvc():
    X = norm(np.load(f"{DATA}/pvc_beats.npy"))
    m = pd.read_csv(f"{DATA}/pvc_meta.csv")
    return X, m.patient_id.values


def load_sinus():
    return norm(np.load(f"{DATA}/normal_beats.npy"))


def load_cpsc():
    X = norm(np.load(f"{DATA}/cpsc_pvc_beats.npy"))
    m = pd.read_csv(f"{DATA}/cpsc_pvc_meta.csv")
    return X, m.rec.values


class MAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.Sequential(
            nn.Conv1d(12, 48, 7, 1, 3), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 96, 5, 1, 2), nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, 96, 5, 1, 2), nn.BatchNorm1d(96), nn.ReLU())
        s.dec = nn.Conv1d(96, 12, 1)

    def forward(s, x):
        return s.dec(s.enc(x))


def train(Xtr, seed, kmax):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = MAE().to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    lf = nn.MSELoss()
    Xg = torch.tensor(Xtr).to(DEV)
    idx = np.arange(len(Xtr))
    bs = 256
    for ep in range(150):
        m.train()
        np.random.shuffle(idx)
        for j in range(0, len(idx), bs):
            xb = Xg[idx[j:j + bs]]
            B = xb.shape[0]
            msk = torch.zeros(B, 12, 1, device=DEV)
            for i in range(B):
                k = np.random.randint(1, kmax + 1)
                msk[i, np.random.choice(12, k, replace=False)] = 1
            out = m(xb * msk)
            l = lf(out, xb)
            opt.zero_grad()
            l.backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def uniq_info(m, Xg):
    """Unique information of lead L: MSE on L when L alone is removed."""
    r = np.empty(12)
    for L in range(12):
        msk = torch.ones(1, 12, 1, device=DEV)
        msk[0, L] = 0
        out = m(Xg * msk)
        r[L] = ((out[:, [L]] - Xg[:, [L]]) ** 2).mean().item()
    return r


@torch.no_grad()
def eval_all(m, Xg, chunk=1024):
    """Full-12-lead reconstruction MSE for each of the 4095 non-empty subsets."""
    e = np.zeros(len(SUBSETS))
    n = Xg.shape[0]
    for i, sub in enumerate(SUBSETS):
        msk = torch.zeros(1, 12, 1, device=DEV)
        msk[0, list(sub)] = 1
        tot = 0.0
        for j in range(0, n, chunk):
            xb = Xg[j:j + chunk]
            out = m(xb * msk)
            tot += ((out - xb) ** 2).mean(dim=(1, 2)).sum().item()
        e[i] = tot / n
    return e


def induced_leads(E):
    """Electrode subset -> tuple of lead indices obtainable from it."""
    E = set(E)
    out = []
    if {'RA', 'LA'} <= E:
        out.append(LIX['I'])
    if {'RA', 'LL'} <= E:
        out.append(LIX['II'])
    if {'LA', 'LL'} <= E:
        out.append(LIX['III'])
    if {'RA', 'LA', 'LL'} <= E:
        out += [LIX['aVR'], LIX['aVL'], LIX['aVF']]
        for i, C in enumerate(['C1', 'C2', 'C3', 'C4', 'C5', 'C6']):
            if C in E:
                out.append(LIX['V' + str(i + 1)])
    return tuple(sorted(set(out)))


# Lead vectors over the 8 independent sources (I, II, V1..V6): rank-8 algebra
LEAD_BASIS = np.zeros((12, 8))
LEAD_BASIS[LIX['I']] = [1, 0, 0, 0, 0, 0, 0, 0]
LEAD_BASIS[LIX['II']] = [0, 1, 0, 0, 0, 0, 0, 0]
LEAD_BASIS[LIX['III']] = [-1, 1, 0, 0, 0, 0, 0, 0]
LEAD_BASIS[LIX['aVR']] = [-0.5, -0.5, 0, 0, 0, 0, 0, 0]
LEAD_BASIS[LIX['aVL']] = [1, -0.5, 0, 0, 0, 0, 0, 0]
LEAD_BASIS[LIX['aVF']] = [-0.5, 1, 0, 0, 0, 0, 0, 0]
for i in range(6):
    LEAD_BASIS[LIX['V' + str(i + 1)], 2 + i] = 1


def subset_rank(sub):
    return int(np.linalg.matrix_rank(LEAD_BASIS[list(sub)]))
