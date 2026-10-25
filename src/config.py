#!/usr/bin/env python3
"""Paths and device, read from environment variables.

    PTBXL_DIR         PTB-XL 1.0.1 root (the folder holding ptbxl_database.csv)
    CPSC_DIR          CPSC-2018 records in WFDB format, as distributed with the
                      PhysioNet/Computing in Cardiology Challenge 2020
    MITDB_DIR         MIT-BIH Arrhythmia Database 1.0.0
    WORK_DIR          where extracted beats, results and figures are written
    DEVICE            torch device, default 'cuda'

Relative defaults resolve against the repository root.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(var, default):
    return os.path.abspath(os.path.join(ROOT, os.environ.get(var, default)))


PTBXL = _path('PTBXL_DIR', 'raw/ptb-xl-1.0.1')
CPSC = _path('CPSC_DIR', 'raw/Training_WFDB')
MITDB = _path('MITDB_DIR', 'raw/mitdb-1.0.0')
WORK = _path('WORK_DIR', 'work')
DEV = os.environ.get('DEVICE', 'cuda')

BEATS = f"{WORK}/beats"                       # extracted PVC and sinus beats
RESULTS = f"{WORK}/results"
CLASSIFIER = f"{RESULTS}/classifier"          # beat classifier weights + card
FITS = f"{RESULTS}/fits"                      # reconstruction-model fits
FIGURES = f"{WORK}/figures"
