#!/usr/bin/env python3
"""PVC and matched sinus beat extraction from PTB-XL and CPSC-2018.

Detection: band-pass 0.5-40 Hz, vector magnitude over the 12 leads, peaks at
least 0.32 s apart above the 75th percentile, followed by the Pan-Tompkins
T-wave discrimination rule adapted to the vector-magnitude signal: a candidate
that follows the last accepted complex by less than T_WINDOW and is smaller than
T_AMP times both that complex and the running median accepted complex is a T
wave and is rejected. Two independent amplitude conditions are required, so the
rule is conservative: a ventricular ectopic beat is wide *and* tall in vector
magnitude and is not rejected.

Classification: every accepted complex is scored by the MIT-BIH-supervised
classifier of beat_classifier.py (leads II and V1, the record's own dominant
template, RR context) and kept as a PVC at or above the classifier's operating
threshold. The classifier is applied uniformly, including in records labelled
with atrial fibrillation or flutter.

Each PVC is stored as a ±200 ms 12-lead window; in PTB-XL a sinus reference beat
from the same record (the median-position complex with classifier probability
< 0.05) is stored alongside it, or NaN where the record has none.

Outputs (config.BEATS): pvc_beats.npy, pvc_meta.csv, normal_beats.npy,
cpsc_pvc_beats.npy, cpsc_pvc_meta.csv.
"""
import ast
import glob
import os
import warnings
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from config import PTBXL as PTB, CPSC, BEATS as OUT, DEV

warnings.filterwarnings('ignore')

os.makedirs(OUT, exist_ok=True)

FS = 500
WIN = int(0.20 * FS)
B, A = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], 'band')
PVC_SNOMED = "164884008"
AFIB_SNOMED = {"164889003", "164890007"}

T_WINDOW = 0.50      # a complex this soon after the last accepted one ...
T_AMP = 0.5          # ... and this much smaller than it, and than the running
                     # median accepted amplitude, is a T wave


def qrs_width(vm, p):
    s = int(0.16 * FS)
    seg = vm[max(0, p - s):p + s]
    if seg.size == 0:
        return 0
    idx = np.where(seg > seg.max() * 0.2)[0]
    return (idx[-1] - idx[0]) / FS if len(idx) > 1 else 0


def detect(sig):
    """Vector-magnitude detection with T-wave rejection.

    Returns the filtered signal, the vector magnitude and the accepted peaks."""
    sf = filtfilt(B, A, sig, axis=0)
    vm = np.sqrt((sf ** 2).sum(1))
    pk, _ = find_peaks(vm, distance=int(0.32 * FS), height=np.percentile(vm, 75))
    if len(pk) == 0:
        return sf, vm, pk, np.zeros(0, bool)
    accepted, rejected = [], np.zeros(len(pk), bool)
    for i, p in enumerate(pk):
        if accepted:
            last = pk[accepted[-1]]
            run_med = np.median(vm[pk[accepted]])
            if ((p - last) / FS < T_WINDOW and vm[p] < T_AMP * vm[last]
                    and vm[p] < T_AMP * run_med):
                rejected[i] = True
                continue
        accepted.append(i)
    return sf, vm, pk[accepted], rejected


_NET = None
_THR = None
LEAD_II, LEAD_V1 = 1, 6   # the leads the MIT-BIH MLII and V1 channels map onto


def _load_net():
    """The MIT-BIH-supervised classifier and its operating threshold."""
    global _NET, _THR
    if _NET is None:
        import json
        import torch
        from beat_classifier import BeatNet, OUT as COUT
        _NET = BeatNet().to(DEV)
        _NET.load_state_dict(torch.load(f"{COUT}/beatnet.pt", map_location=DEV))
        _NET.eval()
        _THR = json.load(open(f"{COUT}/model_card.json"))['threshold']
    return _NET, _THR


def classify(sf, pk):
    """PVC probability for every detected complex, from the two-lead morphology,
    the record's dominant template and the RR context."""
    import torch
    net, thr = _load_net()
    H = WIN
    ok = (pk >= H) & (pk < len(sf) - H)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return np.zeros(len(pk)), thr
    beats = np.stack([sf[p - H:p + H, [LEAD_II, LEAD_V1]].T for p in pk[idx]])
    beats = beats.astype(np.float32)
    m = beats.mean(-1, keepdims=True)
    s = beats.std(-1, keepdims=True)
    beats = (beats - m) / (s + 1e-6)
    tmpl = np.median(beats, axis=0)
    x = np.concatenate([beats, np.repeat(tmpl[None], len(beats), 0),
                        beats - tmpl], axis=1).astype(np.float32)
    rr = np.diff(pk) / FS
    med = np.median(rr) if len(rr) else 1.0
    ctx = []
    for i in idx:
        pre = rr[i - 1] / med if i > 0 else 1.0
        post = rr[i] / med if i < len(rr) else 1.0
        ctx.append([np.clip(pre, 0, 3), np.clip(post, 0, 3),
                    np.clip(pre + post, 0, 6) / 2])
    ctx = np.array(ctx, dtype=np.float32)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(x).to(DEV),
                              torch.tensor(ctx).to(DEV))).cpu().numpy()
    prob = np.zeros(len(pk))
    prob[idx] = p
    return prob, thr


def extract(sig, afib=False):
    """Return (peaks, candidate indices, widths, reference sinus peak, n_rejected)."""
    sf, vm, pk, rej = detect(sig)
    if len(pk) < 4:
        return None
    widths = np.array([qrs_width(vm, p) for p in pk])
    prob, thr = classify(sf, pk)
    cand = np.where(prob >= thr)[0]
    norm_i = np.where(prob < 0.05)[0]
    norm_p = pk[norm_i[len(norm_i) // 2]] if len(norm_i) else None
    return pk, cand, widths, norm_p, int(rej.sum()), sf


def main():
    # ---------------- PTB-XL ----------------
    df = pd.read_csv(f"{PTB}/ptbxl_database.csv", index_col='ecg_id')
    df['scp'] = df.scp_codes.apply(ast.literal_eval)
    pvc_df = df[df.scp.apply(lambda d: 'PVC' in d)]
    print(f"PTB-XL PVC records: {len(pvc_df)}")
    
    beats, metas, normals = [], [], []
    skipped = n_rej = 0
    for ecg_id, row in pvc_df.iterrows():
        try:
            sig, _ = wfdb.rdsamp(f"{PTB}/{row.filename_hr}")
        except Exception:
            skipped += 1; continue
        if sig.shape[1] != 12:
            skipped += 1; continue
        afib = ('AFIB' in row.scp) or ('AFLT' in row.scp)
        res = extract(sig, afib)
        if res is None:
            skipped += 1; continue
        pk, cand, widths, norm_p, nr, sfil = res
        n_rej += nr
        for ci in cand:
            p = pk[ci]
            if p - WIN < 0 or p + WIN > len(sig):
                continue
            beats.append(sig[p - WIN:p + WIN, :].T.astype(np.float32))
            if norm_p is not None and norm_p - WIN >= 0 and norm_p + WIN <= len(sig):
                normals.append(sig[norm_p - WIN:norm_p + WIN, :].T.astype(np.float32))
            else:
                normals.append(np.full((12, 2 * WIN), np.nan, np.float32))
            metas.append(dict(
                ecg_id=ecg_id, patient_id=row.patient_id, pos_s=round(p / FS, 3),
                width_ms=round(1000 * widths[ci]), afib=afib,
                mi=any(k in row.scp for k in ['IMI', 'ASMI', 'ALMI', 'ILMI', 'LMI',
                                              'AMI', 'PMI', 'IPMI', 'IPLMI']),
                lvh='LVH' in row.scp, sr='SR' in row.scp, n_codes=len(row.scp)))
    
    np.save(f"{OUT}/pvc_beats.npy", np.stack(beats))
    np.save(f"{OUT}/normal_beats.npy", np.stack(normals))
    mp = pd.DataFrame(metas)
    mp.to_csv(f"{OUT}/pvc_meta.csv", index=False)
    print(f"PTB-XL: {len(beats)} beats, {mp.ecg_id.nunique()} records, "
          f"{mp.patient_id.nunique()} patients, {n_rej} T waves rejected, "
          f"median width {mp.width_ms.median():.0f} ms")
    
    # ---------------- CPSC ----------------
    def read_dx(hea):
        for ln in open(hea):
            if ln.startswith("#Dx"):
                return set(ln.split(":")[1].replace(" ", "").strip().split(","))
        return set()
    
    
    recs = []
    for hea in glob.glob(f"{CPSC}/*.hea"):
        dx = read_dx(hea)
        if PVC_SNOMED in dx:
            recs.append((os.path.basename(hea)[:-4], dx))
    print(f"CPSC PVC records: {len(recs)}")
    
    beats, metas = [], []
    skipped = n_rej = 0
    for name, dx in recs:
        try:
            sig, _ = wfdb.rdsamp(f"{CPSC}/{name}")
        except Exception:
            skipped += 1; continue
        if sig.shape[1] != 12:
            skipped += 1; continue
        res = extract(sig, bool(dx & AFIB_SNOMED))
        if res is None:
            skipped += 1; continue
        pk, cand, widths, _, nr, sfil = res
        n_rej += nr
        for ci in cand:
            p = pk[ci]
            if p - WIN < 0 or p + WIN > len(sig):
                continue
            beats.append(sig[p - WIN:p + WIN, :].T.astype(np.float32))
            metas.append(dict(rec=name, pos_s=round(p / FS, 3),
                              width_ms=round(1000 * widths[ci]),
                              afib=bool(dx & AFIB_SNOMED)))
    
    np.save(f"{OUT}/cpsc_pvc_beats.npy", np.stack(beats))
    mc = pd.DataFrame(metas)
    mc.to_csv(f"{OUT}/cpsc_pvc_meta.csv", index=False)
    print(f"CPSC: {len(beats)} beats, {mc.rec.nunique()} records, "
          f"{n_rej} T waves rejected, median width {mc.width_ms.median():.0f} ms")


if __name__ == '__main__':
    main()
