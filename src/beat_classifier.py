#!/usr/bin/env python3
"""MIT-BIH-supervised PVC beat classifier.

A small 1-D CNN is trained on the MIT-BIH Arrhythmia Database beat annotations.
Positive class is 'V' (premature ventricular contraction); negatives are the
non-ventricular beat labels; fusion ('F'), ventricular escape ('E'), flutter
waves ('!') and the unclassifiable or non-beat annotation symbols are
excluded as ambiguous.

Negatives also include **detections that are not beats at all**. At inference
every peak the detector finds is scored, including detector artifacts, so a
model trained only on annotated complexes would never have seen one. Detector
peaks in MIT-BIH that match no annotation (about 3%) are therefore added as
negatives.

Channels are selected **by name**, not by position: MIT-BIH channel 0 is MLII in
45 of 48 records but channel 1 is V1 in 40 and V5 in only 2. The model uses the
MLII and V1 channels of the 40 records carrying that pair, and is applied to
leads II and V1 of the 12-lead cohorts. Signals are resampled from 360 to 500 Hz
by polyphase resampling (25/18) so that the window matches the cohorts'
sampling rate. Input is those two channels over ±200 ms, per-channel z-scored,
together with the record's own dominant beat template and the beat-minus-template
residual, plus the RR context. The template matters: in a record with bundle
branch block the conducted beat is itself wide, and a model given only the beat
learns "wide equals ectopic". A ventricular ectopic beat is defined by differing
from the patient's own conducted morphology, which is a relative judgement and
needs the reference.

Performance is estimated with record-wise 5-fold cross-validation, so no record
contributes to both training and test. The operating threshold is the lowest
threshold whose held-out PPV reaches TARGET_PPV, or the threshold of highest
held-out PPV if none does; the final model is then fitted on all records.

Outputs (config.CLASSIFIER): beatnet.pt, model_card.json, operating_curve.csv.
"""
import glob
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold
from beat_extraction import detect as _detect
from config import MITDB, CLASSIFIER as OUT, DEV

warnings.filterwarnings('ignore')

os.makedirs(OUT, exist_ok=True)
FS = 500
HALF = 100                    # ±200 ms at 500 Hz
CHANNELS = ['MLII', 'V1']       # selected by name; see the docstring
APPLIED_TO = ['II', 'V1']        # the 12-lead leads these map onto at inference
POS = {'V'}
AMBIG = {'F', 'E', '!', 'Q', '+', '~', '|', '"', 'x', '[', ']'}
TARGET_PPV = 0.90


class BeatNet(nn.Module):
    """Morphology of the two leads plus the RR context of the beat.

    Time is pooled to 25 steps rather than to a single value: a PVC differs from
    a sinus beat in the *shape* of the complex, which an average over the whole
    window destroys."""

    def __init__(s, n_ctx=3):
        super().__init__()
        s.f = nn.Sequential(
            nn.Conv1d(6, 32, 7, 1, 3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, 1, 2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(25))
        s.h = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 25 + n_ctx, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))

    def forward(s, x, ctx):
        z = s.f(x)
        return s.h(torch.cat([z.flatten(1), ctx], 1)).squeeze(-1)


def zscore(x):
    m = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - m) / (sd + 1e-6)


def build_dataset():
    """Beats from the records that carry every channel in CHANNELS, read by name."""
    recs = [f[:-4] for f in sorted(glob.glob(f"{MITDB}/*.atr"))
            if os.path.exists(f[:-4] + '.hea')]
    X, y, g, C = [], [], [], []
    used = []
    n_nonbeat = 0
    for rec in recs:
        h = wfdb.rdheader(rec)
        if not all(c in h.sig_name for c in CHANNELS):
            continue
        used.append(os.path.basename(rec))
        r = wfdb.rdrecord(rec)
        a = wfdb.rdann(rec, 'atr')
        idx = [h.sig_name.index(c) for c in CHANNELS]
        sig = resample_poly(r.p_signal[:, idx], 25, 18, axis=0).astype(np.float32)
        # detector peaks with no annotation nearby: non-beats, used as negatives
        _sf12 = np.zeros((sig.shape[0], 12), dtype=np.float32)
        _sf12[:, 1] = sig[:, 0]
        _sf12[:, 6] = sig[:, 1] if sig.shape[1] > 1 else 0
        _f, _vm, _pk, _ = _detect(_sf12)
        pos_all = np.array([int(s * FS / r.fs) for s in a.sample])
        rr_all = np.diff(pos_all) / FS
        med = np.median(rr_all) if len(rr_all) else 1.0
        for i, (s, sym) in enumerate(zip(a.sample, a.symbol)):
            if sym in AMBIG:
                continue
            p = int(s * FS / r.fs)
            if p - HALF < 0 or p + HALF > len(sig):
                continue
            pre = rr_all[i - 1] / med if i > 0 else 1.0
            post = rr_all[i] / med if i < len(rr_all) else 1.0
            X.append(sig[p - HALF:p + HALF].T)
            C.append([np.clip(pre, 0, 3), np.clip(post, 0, 3),
                      np.clip(pre + post, 0, 6) / 2])
            y.append(1 if sym in POS else 0)
            g.append(os.path.basename(rec))
        # non-beat negatives
        rr_med = np.median(np.diff(_pk) / FS) if len(_pk) > 2 else 1.0
        for i, p in enumerate(_pk):
            if len(pos_all) and np.abs(pos_all - p).min() <= int(0.15 * FS):
                continue
            if p - HALF < 0 or p + HALF > len(sig):
                continue
            pre = ((p - _pk[i - 1]) / FS / rr_med) if i > 0 else 1.0
            post = ((_pk[i + 1] - p) / FS / rr_med) if i < len(_pk) - 1 else 1.0
            X.append(sig[p - HALF:p + HALF].T)
            C.append([np.clip(pre, 0, 3), np.clip(post, 0, 3),
                      np.clip(pre + post, 0, 6) / 2])
            y.append(0)
            g.append(os.path.basename(rec))
            n_nonbeat += 1
    print(f"records carrying {CHANNELS}: {len(used)}; "
          f"non-beat negatives added: {n_nonbeat}")
    X = zscore(np.stack(X).astype(np.float32))
    X = add_template(X, np.array(g))
    return X, np.array(C, dtype=np.float32), np.array(y), np.array(g)


def add_template(X, g):
    """Append the record's dominant beat template and the beat-minus-template
    residual, giving 6 channels: beat(2), template(2), residual(2)."""
    out = np.empty((len(X), 6, X.shape[2]), dtype=np.float32)
    for rec in np.unique(g):
        m = g == rec
        tmpl = np.median(X[m], axis=0)
        out[m, 0:2] = X[m]
        out[m, 2:4] = tmpl
        out[m, 4:6] = X[m] - tmpl
    return out


def train_one(Xtr, Ctr, ytr, seed=0, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = BeatNet().to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    pos_w = torch.tensor([np.sqrt((ytr == 0).sum() / max((ytr == 1).sum(), 1))],
                         dtype=torch.float32, device=DEV)
    lf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xg = torch.tensor(Xtr).to(DEV)
    Cg = torch.tensor(Ctr).to(DEV)
    yg = torch.tensor(ytr, dtype=torch.float32).to(DEV)
    idx = np.arange(len(Xtr))
    bs = 512
    for ep in range(epochs):
        m.train()
        np.random.shuffle(idx)
        for j in range(0, len(idx), bs):
            b = idx[j:j + bs]
            xb = Xg[b]
            # ±30 ms jitter, so the model tolerates detector-vs-annotation offset
            sh = np.random.randint(-15, 16)
            xb = torch.roll(xb, shifts=int(sh), dims=-1)
            loss = lf(m(xb, Cg[b]), yg[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def predict(m, X, C, bs=4096):
    out = []
    for j in range(0, len(X), bs):
        xb = torch.tensor(X[j:j + bs]).to(DEV)
        cb = torch.tensor(C[j:j + bs]).to(DEV)
        out.append(torch.sigmoid(m(xb, cb)).cpu().numpy())
    return np.concatenate(out)


def main():
    X, C, y, g = build_dataset()
    print(f"beats {len(X)} | PVC {int(y.sum())} | records {len(np.unique(g))}",
          flush=True)

    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g)):
        m = train_one(X[tr], C[tr], y[tr], seed=k)
        oof[te] = predict(m, X[te], C[te])
        print(f"  fold {k} done", flush=True)

    rows = []
    for t in np.round(np.concatenate([np.arange(0.05, 1.0, 0.05),
                                      [0.96, 0.97, 0.98, 0.99]]), 3):
        pred = oof >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        rows.append(dict(threshold=t, sensitivity=100 * tp / (tp + fn),
                         ppv=100 * tp / max(tp + fp, 1), tp=tp, fp=fp, fn=fn))
    curve = pd.DataFrame(rows)
    curve.to_csv(f"{OUT}/operating_curve.csv", index=False)
    print(curve.round(1).to_string(index=False))

    ok = curve[curve.ppv >= 100 * TARGET_PPV]
    chosen = (ok.iloc[0] if len(ok)
              else curve.iloc[curve.ppv.idxmax()])
    thr = float(chosen.threshold)
    print(f"\nchosen threshold {thr}: sensitivity {chosen.sensitivity:.1f}%, "
          f"PPV {chosen.ppv:.1f}% (record-wise CV, n={len(y)})")

    final = train_one(X, C, y, seed=99)
    torch.save(final.state_dict(), f"{OUT}/beatnet.pt")
    json.dump(dict(threshold=thr, sensitivity=float(chosen.sensitivity),
                   ppv=float(chosen.ppv), n_beats=int(len(y)),
                   n_pvc=int(y.sum()), n_records=int(len(np.unique(g))),
                   fs=FS, half_window=HALF, channels=CHANNELS,
                   applied_to=APPLIED_TO,
                   resample='polyphase 25/18, 360 -> 500 Hz',
                   records_used=int(len(np.unique(g))),
                   positive_labels=sorted(POS), excluded=sorted(AMBIG)),
              open(f"{OUT}/model_card.json", 'w'), indent=2)
    print(f"saved -> {OUT}/beatnet.pt")


if __name__ == '__main__':
    main()
