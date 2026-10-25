#!/usr/bin/env python3
"""Extraction quality control, with one definition for both cohorts (Table 1,
Section 3.1).

On the beats actually used in the study:

  - proportion of beats with vector-magnitude QRS width <140 ms;
  - compensatory-pause ratio (pre_RR + post_RR) / (2 x median_RR), its
    distribution and the proportion >= 0.9, excluding AF/AFL-labelled records;
  - the same restricted to isolated ectopy, since a couplet or run has no
    compensatory pause by definition;
  - the evaluable fraction, since a beat at a record edge has no pre- or post-RR.

Peaks are re-detected with the detector of beat_extraction.py (including T-wave
rejection), so the RR context is the one the extraction used, and the extracted
beats are matched back by their recorded position.

Outputs (config.RESULTS/qc): ptbxl_beat_qc.csv, cpsc_beat_qc.csv, qc_summary.md.
"""
import ast
import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PTBXL as PTB, CPSC, BEATS as DATA, RESULTS

OUT = f"{RESULTS}/qc"
os.makedirs(OUT, exist_ok=True)

FS = 500
B, A = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], 'band')
PVC_SNOMED = "164884008"
AFIB_SNOMED = {"164889003", "164890007"}


def qrs_width(vm, p):
    s = int(0.16 * FS)
    seg = vm[max(0, p - s):p + s]
    if seg.size == 0:
        return 0
    idx = np.where(seg > seg.max() * 0.2)[0]
    return (idx[-1] - idx[0]) / FS if len(idx) > 1 else 0


from beat_extraction import detect as _detect


def peaks_of(sig):
    """The extraction detector (with T-wave rejection), so the RR context here is
    the same one the extraction used."""
    _sf, vm, pk, _rej = _detect(sig)
    return vm, pk


def pause_for_record(sig, positions_s):
    """One row per requested beat: pause ratio, evaluability, and whether the
    beat is isolated (neither neighbouring complex is itself an extracted PVC).
    A couplet or a run has no compensatory pause by definition, so the ratio is
    only interpretable for isolated ectopy."""
    vm, pk = peaks_of(sig)
    if len(pk) < 4:
        return [dict(evaluable=False, ratio=np.nan, isolated=np.nan,
                     reason='fewer than 4 peaks') for _ in positions_s]
    rr = np.diff(pk) / FS
    medrr = np.median(rr)
    matched = {}
    for j, pos in enumerate(positions_s):
        p = int(round(pos * FS))
        ci = int(np.argmin(np.abs(pk - p)))
        matched[j] = ci if abs(pk[ci] - p) <= int(0.05 * FS) else None
    flagged = {c for c in matched.values() if c is not None}
    rows = []
    for j, pos in enumerate(positions_s):
        ci = matched[j]
        if ci is None:
            rows.append(dict(evaluable=False, ratio=np.nan, isolated=np.nan,
                             reason='beat not re-detected'))
            continue
        iso = not ((ci - 1 in flagged) or (ci + 1 in flagged))
        if ci == 0 or ci >= len(rr):
            rows.append(dict(evaluable=False, ratio=np.nan, isolated=iso,
                             reason='no pre- or post-RR (record edge)'))
        else:
            rows.append(dict(evaluable=True, isolated=iso,
                             ratio=(rr[ci - 1] + rr[ci]) / (2 * medrr),
                             reason=''))
    return rows


# ---------------- PTB-XL ----------------
meta_p = pd.read_csv(f"{DATA}/pvc_meta.csv")
df = pd.read_csv(f"{PTB}/ptbxl_database.csv", index_col='ecg_id')
df['scp'] = df.scp_codes.apply(ast.literal_eval)
rows = []
for ecg_id, g in meta_p.groupby('ecg_id'):
    try:
        sig, _ = wfdb.rdsamp(f"{PTB}/{df.loc[ecg_id,'filename_hr']}")
    except Exception:
        for _ in range(len(g)):
            rows.append(dict(evaluable=False, ratio=np.nan, reason='unreadable'))
        continue
    for r, row in zip(pause_for_record(sig, g.pos_s.values), g.itertuples()):
        r.update(ecg_id=ecg_id, width_ms=row.width_ms, afib=row.afib,
                 sr=row.sr, mi=row.mi)
        rows.append(r)
qp = pd.DataFrame(rows)
qp.to_csv(f"{OUT}/ptbxl_beat_qc.csv", index=False)

# ---------------- CPSC ----------------
meta_c = pd.read_csv(f"{DATA}/cpsc_pvc_meta.csv")
rows = []
for rec, g in meta_c.groupby('rec'):
    try:
        sig, _ = wfdb.rdsamp(f"{CPSC}/{rec}")
    except Exception:
        for _ in range(len(g)):
            rows.append(dict(evaluable=False, ratio=np.nan, reason='unreadable'))
        continue
    for r, row in zip(pause_for_record(sig, g.pos_s.values), g.itertuples()):
        r.update(rec=rec, width_ms=row.width_ms, afib=row.afib,
                 duration_s=sig.shape[0] / FS)
        rows.append(r)
qc = pd.DataFrame(rows)
qc.to_csv(f"{OUT}/cpsc_beat_qc.csv", index=False)

# ---------------- report ----------------
out = ["# Extraction quality control\n"]
P = out.append


def block(name, q, meta):
    P(f"## {name}\n")
    P(f"- Beats: {len(q)}")
    P(f"- QRS width, median (mean): {meta.width_ms.median():.0f} "
      f"({meta.width_ms.mean():.0f}) ms")
    P(f"- Beats with width <140 ms: **{100*(meta.width_ms<140).mean():.1f}%** "
      f"({int((meta.width_ms<140).sum())}/{len(meta)})")
    nonaf = q[~q.afib.astype(bool)]
    ev = nonaf[nonaf.evaluable]
    P(f"- Non-AF beats: {len(nonaf)}; evaluable for the pause ratio: "
      f"**{len(ev)}/{len(nonaf)} ({100*len(ev)/len(nonaf):.1f}%)**")
    if len(ev):
        P(f"- Pause ratio (pre+post RR)/(2 x median RR): median "
          f"**{ev.ratio.median():.2f}** (IQR {ev.ratio.quantile(.25):.2f}–"
          f"{ev.ratio.quantile(.75):.2f}); **{100*(ev.ratio>=0.9).mean():.0f}%** ≥0.9")
        iso, cons = ev[ev.isolated == True], ev[ev.isolated == False]
        P(f"- Isolated ectopy ({len(iso)} beats): median "
          f"{iso.ratio.median():.2f}; **{100*(iso.ratio>=0.9).mean():.0f}%** ≥0.9")
        P(f"- Consecutive ectopy ({len(cons)} beats, "
          f"{100*len(cons)/len(ev):.0f}% of evaluable): "
          f"{100*(cons.ratio>=0.9).mean():.0f}% ≥0.9 — a couplet or run has no "
          f"compensatory pause by definition")
    nonev = nonaf[~nonaf.evaluable]
    if len(nonev):
        P(f"- Reasons a beat was not evaluable: " +
          ", ".join(f"{k} {v}" for k, v in nonev.reason.value_counts().items()))
    P("")


block("PTB-XL — all PVC-labelled records", qp, meta_p)
block("CPSC-2018 — all PVC-labelled records", qc, meta_c)

# PTB-XL clean sinus-rhythm subset (sinus rhythm, no AF/AFL, no MI label)
clean = qp[(qp.sr.astype(bool)) & (~qp.afib.astype(bool)) & (~qp.mi.astype(bool))]
ev = clean[clean.evaluable]
P("## PTB-XL — clean sinus-rhythm subset\n")
P(f"- Beats {len(clean)}, evaluable {len(ev)} ({100*len(ev)/max(len(clean),1):.1f}%); "
  f"median ratio {ev.ratio.median():.2f}; **{100*(ev.ratio>=0.9).mean():.0f}%** ≥0.9\n")

P("## Record duration (why the evaluable fraction differs)\n")
P(f"- CPSC recording duration: median {qc.duration_s.median():.1f} s "
  f"(range {qc.duration_s.min():.1f}–{qc.duration_s.max():.1f} s); "
  f"PTB-XL records are uniformly 10 s.\n")

txt = "\n".join(out)
open(f"{OUT}/qc_summary.md", "w").write(txt + "\n")
print(txt)
