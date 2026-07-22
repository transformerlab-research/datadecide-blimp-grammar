#!/usr/bin/env python3
"""Reproduce the headline result: Spearman rank correlation between BLiMP grammatical
competence and DataDecide's published olmes_10 QA aggregate, at 90M and 20M.

Reads ../data/blimp-{90m-v2,20m}-full.csv (this repo) and the public DataDecide eval
bundle (allenai/DataDecide-eval-results, macro_avg parquet), which it downloads once.

Usage:  python analysis/qa_join.py        (run from repo root or analysis/)
Deps:   pandas numpy scipy pyarrow requests
"""
import os, json, urllib.request
import pandas as pd, numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PARQUET_URL = ("https://huggingface.co/datasets/allenai/DataDecide-eval-results/"
               "resolve/main/data/macro_avg-00000-of-00001.parquet")
PARQUET = os.path.join(HERE, "macro_avg.parquet")

# DataDecide recipe display-name -> our recipe slug
NAME = {
 'C4':'c4','DCLM-Baseline':'dclm-baseline','DCLM-Baseline (QC 10%)':'dclm-baseline-qc-10p',
 'DCLM-Baseline (QC 20%)':'dclm-baseline-qc-20p','DCLM-Baseline (QC 7%, FW2)':'dclm-baseline-qc-7p-fw2',
 'DCLM-Baseline (QC 7%, FW3)':'dclm-baseline-qc-7p-fw3','DCLM-Baseline (QC FW 10%)':'dclm-baseline-qc-fw-10p',
 'DCLM-Baseline (QC FW 3%)':'dclm-baseline-qc-fw-3p','DCLM-Baseline 25% / Dolma 75%':'dclm-baseline-25p-dolma1.7-75p',
 'DCLM-Baseline 50% / Dolma 50%':'dclm-baseline-50p-dolma1.7-50p','DCLM-Baseline 75% / Dolma 25%':'dclm-baseline-75p-dolma1.7-25p',
 'Dolma1.6++':'dolma1_6plus','Dolma1.7':'dolma1_7','Dolma1.7 (no Flan)':'dolma1_7-no-flan',
 'Dolma1.7 (no Reddit)':'dolma1_7-no-reddit','Dolma1.7 (no code)':'dolma1_7-no-code',
 'Dolma1.7 (no math, code)':'dolma1_7-no-math-code','Falcon':'falcon','Falcon+CC':'falcon-and-cc',
 'Falcon+CC (QC 10%)':'falcon-and-cc-qc-10p','Falcon+CC (QC 20%)':'falcon-and-cc-qc-20p',
 'Falcon+CC (QC Orig 10%)':'falcon-and-cc-qc-orig-10p','Falcon+CC (QC Tulu 10%)':'falcon-and-cc-qc-tulu-10p',
 'FineWeb-Edu':'fineweb-edu','FineWeb-Pro':'fineweb-pro'}

def qa_ranking(params):
    if not os.path.exists(PARQUET):
        print("downloading DataDecide eval bundle (~140 MB, once)...")
        urllib.request.urlretrieve(PARQUET_URL, PARQUET)
    d = pd.read_parquet(PARQUET)
    d = d[(d.params == params) & (d.task == 'olmes_10_macro_avg')]
    d = d[d.step == d.step.max()]
    d['qa'] = d.metrics.map(lambda m: json.loads(m)['primary_metric'])
    q = d.groupby('data').qa.mean()
    q.index = [NAME[i] for i in q.index]
    return q

def spearman_ci(x, y, reps=10000, seed=0):
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = [stats.spearmanr(x.values[i], y.values[i]).statistic
          for i in (rng.integers(0, len(x), len(x)) for _ in range(reps)) if len(set(i)) > 3]
    return rho, p, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

for size, csv in [('90M', 'blimp-90m-v2-full.csv'), ('20M', 'blimp-20m-full.csv')]:
    bl = pd.read_csv(os.path.join(DATA, csv)).set_index('recipe')
    j = pd.DataFrame({'blimp': bl.macro5_no_detn, 'qa': qa_ranking(size)}).dropna()
    rho, p, lo, hi = spearman_ci(j.blimp, j.qa)
    print(f"\n{size}: BLiMP grammar (5-paradigm, saturation-free) vs DataDecide olmes_10 QA")
    print(f"  Spearman rho = {rho:+.3f}  p = {p:.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n = {len(j)}")
