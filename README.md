# pvc-lead-information

Analysis code for:

> Köksal F. Reconstructive information content of ECG leads and electrodes in
> premature ventricular contractions: an exhaustive, dual-population study.
> *Journal of Electrocardiology*, 2026.

The pipeline selects premature ventricular contractions (PVCs) from two public
12-lead cohorts with an MIT-BIH-supervised beat classifier, trains a masked
convolutional autoencoder that reconstructs all 12 leads from any subset, and
evaluates all 4095 lead subsets and all 256 electrode subsets exhaustively.

## Data

All three databases are public; none is redistributed here.

| Database | Version | Used for | Environment variable |
|---|---|---|---|
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) | 1.0.1 | primary cohort | `PTBXL_DIR` (folder holding `ptbxl_database.csv`) |
| [CPSC-2018](https://physionet.org/content/challenge-2020/), as distributed with the PhysioNet/Computing in Cardiology Challenge 2020 | records A0001–A6877, WFDB format (`Training_WFDB`) | external cohort | `CPSC_DIR` |
| [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) | 1.0.0 | training the beat classifier | `MITDB_DIR` |

Unset variables default to `raw/ptb-xl-1.0.1`, `raw/Training_WFDB` and
`raw/mitdb-1.0.0` under the repository root. All outputs go to `WORK_DIR`
(default `work/`).

## Requirements

Python 3.13 with the packages in `requirements.txt`, and a CUDA GPU (set
`DEVICE=cpu` to run without one, much more slowly). The published results were
produced on an NVIDIA GeForce RTX 5090 with PyTorch 2.11.0 / CUDA 12.8; one set
of 25 reconstruction-model fits takes about 7 minutes there.

```bash
pip install -r requirements.txt
export PTBXL_DIR=/path/to/ptb-xl-1.0.1
export CPSC_DIR=/path/to/Training_WFDB
export MITDB_DIR=/path/to/mitdb/1.0.0
./run_all.sh
```

## Pipeline

| Step | Script | Paper |
|---|---|---|
| Beat classifier, trained on MIT-BIH with record-wise cross-validation | `src/beat_classifier.py` | Methods 2.1 |
| Beat detection with T-wave rejection, PVC selection, matched sinus beats | `src/beat_extraction.py` | Methods 2.1 |
| Extraction quality control | `src/quality_control.py` | Section 3.1, Table 1 |
| Figure 1 | `src/figure1.py` | Fig. 1 |
| Reconstruction model, 5-fold × 5-seed CV, all 4095 subsets, PTB-XL → CPSC transfer | `src/train_eval.py` (shared code in `src/common.py`) | Methods 2.2–2.4 |
| CPSC internal cross-validation | `src/cpsc_internal_cv.py`, `src/cpsc_internal_subsets.py` | Sections 3.2–3.3 |
| PVC versus matched sinus beats | `src/sinus_comparison.py` | Section 3.5 |
| Pareto frontiers, unique information, sensitivity model, algebraic rank | `src/pareto_analysis.py` | Table 2, Sections 3.2–3.4 |
| Figures 2–5 | `src/figures.py` | Figs. 2–5 |
| Flatness and within-cohort stability of the optimal subsets | `src/selection_stability.py` | Section 3.3 |
| Montage retention ladder; sinus- versus PVC-selected montage | `src/montage_retention.py` | Sections 3.4–3.5 |
| Paired PVC-versus-sinus tests, Holm and global Bonferroni | `src/paired_tests.py` | Section 3.5, Supplementary Table S1 |
| Within-record separability after reduction | `src/separability.py` | Sections 3.5, 4.1 |

`run_all.sh` runs them in this order.

GPU training is not bit-for-bit deterministic, so a re-run can differ from the
published values in the last reported digit, and a retrained beat classifier
can select a slightly different beat set.

## License

Code: MIT (see `LICENSE`). The databases are subject to their own licences on
PhysioNet.
