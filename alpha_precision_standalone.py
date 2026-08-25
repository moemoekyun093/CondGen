"""
Standalone alpha-Precision / beta-Recall (Alaa et al. 2022), WITHOUT synthcity.

Reproduces the reference evaluation script's actual behavior exactly:
  - target column is folded into num_col_idx (regression) or cat_col_idx
    (classification) and is NOT dropped (their GenericDataLoader(...) call passes
    no target_column=, so synthcity's target-drop logic never triggers)
  - categoricals: cleaned ("3.0"->"3", whitespace-stripped), one-hot encoded,
    encoder fit on REAL only
  - MinMaxScaler fit on REAL only, applied to [numerics | one-hot categoricals]

Then reproduces synthcity's AlphaPrecision.metrics() (naive/no-embedding case) exactly:
  - alphas = linspace(0, 1, 30)
  - center = mean(X_real); Radii = quantile(||X_real - center||, alphas)
  - alpha_precision_curve[k] = mean(||X_syn - center|| <= Radii[k])
  - real_to_real   = each real point's distance to its nearest OTHER real point
  - real_to_synth  = each real point's distance to its nearest synthetic point
  - real_synth_closest_d = distance of those nearest synthetic points to synth_center
  - closest_synth_Radii = quantile(real_synth_closest_d, alphas)
  - beta_coverage_curve[k] = mean((real_to_synth <= real_to_real)
                                   & (real_synth_closest_d <= closest_synth_Radii[k]))
  - authenticity = mean(real_to_real[real_to_synth_args] < real_to_synth)
  - Delta_precision_alpha = 1 - sum(|alphas - alpha_precision_curve|) / sum(alphas)
  - Delta_coverage_beta   = 1 - sum(|alphas - beta_coverage_curve|) / sum(alphas)

One deliberate deviation from the reference script: OneHotEncoder uses
handle_unknown='ignore' (not the sklearn default 'error'), so unseen synthetic
categories map to an all-zero row instead of crashing the whole run.

Requires len(X) == len(X_syn) (synthcity's own constraint) -- the larger set is
randomly subsampled (seeded) to match.

USAGE:
    python alpha_precision_standalone.py \
        --generated_data_folder artifacts/generated_data \
        --target_folder artifacts/metrics_results \
        --pattern "*TabDiffFTPeriodic*.csv"
"""
import argparse
import fnmatch
import json
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ORIGINAL_DATA_FOLDER = 'data'
NAMES = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'diabetes']


def find_related_dataset(file_path):
    for name in sorted(NAMES, key=len, reverse=True):
        if name in file_path:
            return name
    raise ValueError(f"Unknown dataset for file: {file_path}")


def find_files(starting_folder, pattern):
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full = os.path.join(root, filename)
            if fnmatch.fnmatch(full, pattern):
                matches.append(full)
    return matches


def build_features(real_df, syn_df, info):
    """
    Matches the reference script's actual behavior: target column is folded into
    num_col_idx (regression) or cat_col_idx (classification) and is NOT dropped,
    because their GenericDataLoader(...) call passes no target_column=, so
    synthcity's _normalize_covariates never strips it (hasattr(X,'target_column')
    is False in that usage). MinMaxScaler is still applied afterwards, fit on real.
    """
    real_df = real_df.copy(); syn_df = syn_df.copy()
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))

    num_idx = list(info['num_col_idx'])
    cat_idx = list(info['cat_col_idx'])
    tgt_idx = list(info['target_col_idx'])
    if info['task_type'] == 'regression':
        num_idx = num_idx + tgt_idx     # target folded into numerics
    else:
        cat_idx = cat_idx + tgt_idx     # target folded into categoricals
    # NOT dropped -- kept in, matching the reference script's GenericDataLoader usage

    num_real = real_df[num_idx].to_numpy().astype(float) if num_idx else np.zeros((len(real_df), 0))
    num_syn = syn_df[num_idx].to_numpy().astype(float) if num_idx else np.zeros((len(syn_df), 0))

    if cat_idx:
        cat_real = real_df[cat_idx].to_numpy().astype(str)
        cat_syn = syn_df[cat_idx].to_numpy().astype(str)

        def clean(a):
            out = a.astype(str)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    if out[i, j].endswith('.0'):
                        out[i, j] = out[i, j][:-2]
            return np.char.strip(out.astype(str))

        cat_real = clean(cat_real); cat_syn = clean(cat_syn)
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        enc.fit(cat_real)
        oh_real = enc.transform(cat_real)
        oh_syn = enc.transform(cat_syn)
    else:
        oh_real = np.zeros((len(real_df), 0)); oh_syn = np.zeros((len(syn_df), 0))

    X_real = np.concatenate([num_real, oh_real], axis=1)
    X_syn = np.concatenate([num_syn, oh_syn], axis=1)

    # MinMaxScaler, fit on REAL only -- matches synthcity's _normalize_covariates
    scaler = MinMaxScaler().fit(X_real)
    return scaler.transform(X_real), scaler.transform(X_syn)


def alpha_precision_beta_recall_authenticity(X, X_syn, n_steps=30, seed=0):
    """
    Exact port of synthcity's AlphaPrecision.metrics(), naive (no embedding) case.
    Requires len(X) == len(X_syn); the caller subsamples to match.
    Returns (Delta_precision_alpha, Delta_coverage_beta, authenticity).
    """
    if len(X) != len(X_syn):
        raise RuntimeError("X and X_syn must have the same length (synthcity's own constraint)")

    emb_center = np.mean(X, axis=0)
    alphas = np.linspace(0, 1, n_steps)

    Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)

    synth_center = np.mean(X_syn, axis=0)

    alpha_precision_curve = []
    beta_coverage_curve = []

    synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
    real_to_real, _ = nbrs_real.kneighbors(X)

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

    # nearest OTHER real point (skip self, hence column 1 not 0)
    real_to_real = real_to_real[:, 1].squeeze()
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()

    real_synth_closest = X_syn[real_to_synth_args]
    real_synth_closest_d = np.sqrt(np.sum((real_synth_closest - synth_center) ** 2, axis=1))
    closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

    for k in range(len(Radii)):
        precision_audit_mask = synth_to_center <= Radii[k]
        alpha_precision = np.mean(precision_audit_mask)

        beta_coverage = np.mean(
            (real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])
        )

        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)

    authen = real_to_real[real_to_synth_args] < real_to_synth
    authenticity = np.mean(authen)

    Delta_precision_alpha = 1 - np.sum(
        np.abs(np.array(alphas) - np.array(alpha_precision_curve))
    ) / np.sum(alphas)

    Delta_coverage_beta = 1 - np.sum(
        np.abs(np.array(alphas) - np.array(beta_coverage_curve))
    ) / np.sum(alphas)

    return float(Delta_precision_alpha), float(Delta_coverage_beta), float(authenticity)


def evaluate_one(real_path, syn_path, info_path, seed=0):
    with open(info_path) as f:
        info = json.load(f)
    real_df = pd.read_csv(real_path)
    syn_df = pd.read_csv(syn_path)
    X_real, X_syn = build_features(real_df, syn_df, info)
    if np.isnan(X_syn).any():
        print(f"  synthetic has NaN, skipping {syn_path}")
        return None, None, None

    # synthcity's metrics() requires equal-length X and X_syn; subsample the larger.
    rng = np.random.RandomState(seed)
    n = min(len(X_real), len(X_syn))
    if len(X_real) > n:
        X_real = X_real[rng.choice(len(X_real), n, replace=False)]
    if len(X_syn) > n:
        X_syn = X_syn[rng.choice(len(X_syn), n, replace=False)]

    return alpha_precision_beta_recall_authenticity(X_real, X_syn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='*.csv')
    ap.add_argument('--target_folder', default='artifacts/metrics_results')
    ap.add_argument('--generated_data_folder', default='artifacts/generated_data')
    ap.add_argument('--seed', type=int, default=0, help='subsample seed when real/synthetic sizes differ')
    args = ap.parse_args()

    os.makedirs(args.target_folder, exist_ok=True)
    files = find_files(args.generated_data_folder, args.pattern)
    print(f"Pattern: {args.pattern}")
    print(f"Found {len(files)} files")

    for gen in files:
        ds = find_related_dataset(gen)
        real_path = os.path.join(ORIGINAL_DATA_FOLDER, ds, 'train.csv')
        info_path = os.path.join(ORIGINAL_DATA_FOLDER, ds, 'info.json')

        a_dir = os.path.join(os.path.dirname(gen).replace(args.generated_data_folder, args.target_folder), 'alpha_precision')
        b_dir = os.path.join(os.path.dirname(gen).replace(args.generated_data_folder, args.target_folder), 'beta_recall')
        c_dir = os.path.join(os.path.dirname(gen).replace(args.generated_data_folder, args.target_folder), 'authenticity')
        os.makedirs(a_dir, exist_ok=True); os.makedirs(b_dir, exist_ok=True); os.makedirs(c_dir, exist_ok=True)
        a_path = os.path.join(a_dir, os.path.basename(gen).replace('.csv', '.json'))
        b_path = os.path.join(b_dir, os.path.basename(gen).replace('.csv', '.json'))
        c_path = os.path.join(c_dir, os.path.basename(gen).replace('.csv', '.json'))

        if os.path.exists(a_path) and os.path.exists(b_path) and os.path.exists(c_path):
            print(f"  exists, skipping {gen}")
            continue

        try:
            a, b, c = evaluate_one(real_path, syn_path=gen, info_path=info_path, seed=args.seed)
        except Exception as e:
            print(f"  error {gen}: {type(e).__name__}: {e}")
            continue
        if a is None:
            continue

        with open(a_path, 'w') as f:
            json.dump({'alpha_precision': a}, f, indent=4)
        with open(b_path, 'w') as f:
            json.dump({'beta_recall': b}, f, indent=4)
        with open(c_path, 'w') as f:
            json.dump({'authenticity': c}, f, indent=4)
        print(f"  {ds}: alpha_precision={a:.4f} beta_recall={b:.4f} authenticity={c:.4f} -> {os.path.basename(gen)}")


if __name__ == "__main__":
    main()