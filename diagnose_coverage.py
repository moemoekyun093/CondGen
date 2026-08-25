"""
Diagnose LOW BETA-RECALL (mode collapse / missing coverage): compares real vs synthetic
data column-by-column to find WHERE the synthetic distribution fails to cover real data.

Two checks:
  1. CATEGORICAL COVERAGE: for each categorical column, how many of the real data's
     unique values actually appear in the synthetic data, and how far off are the
     category frequencies? Flags columns where synthetic collapses onto a subset.
  2. NUMERICAL COVERAGE: for each numerical column, compares the [min,max] RANGE
     covered, plots real vs synthetic histograms, and reports a two-sample KS
     statistic (large KS = distributions differ a lot) plus how much of the real
     data's range/tails the synthetic data actually reaches.

Outputs:
  - a printed per-column summary table (numbers)
  - PNG histogram overlays for the worst-covered numerical columns
  - a printed ranked list of "most collapsed" columns to focus on

USAGE (run from the TabDiff repo root, so data/{dataset}/ is reachable):
    python diagnose_coverage.py --dataname beijing \
        --synthetic_csv tabdiff/result/beijing/ft_periodic_seed0/<epoch>/samples.csv \
        --out_dir coverage_diagnostics/beijing

    python diagnose_coverage.py --dataname diabetes \
        --synthetic_csv tabdiff/result/diabetes/ft_periodic_seed0/<epoch>/samples.csv \
        --out_dir coverage_diagnostics/diabetes
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def load_info(dataname):
    with open(f"data/{dataname}/info.json") as f:
        return json.load(f)


def load_real(dataname):
    return pd.read_csv(f"data/{dataname}/train.csv")


def get_col_groups(info, real_df):
    """Column indices -> names, split by type. Handles the num/cat/target ordering
    convention used elsewhere in this project (target goes into num for regression,
    into cat for classification)."""
    num_idx = list(info['num_col_idx'])
    cat_idx = list(info['cat_col_idx'])
    tgt_idx = list(info['target_col_idx'])
    if info['task_type'] == 'regression':
        num_idx = num_idx + tgt_idx
    else:
        cat_idx = cat_idx + tgt_idx

    cols = list(real_df.columns)
    num_cols = [cols[i] for i in num_idx if i < len(cols)]
    cat_cols = [cols[i] for i in cat_idx if i < len(cols)]
    return num_cols, cat_cols


def categorical_coverage(real_df, syn_df, cat_cols):
    print("\n" + "=" * 90)
    print("CATEGORICAL COVERAGE  (does synthetic reach all the real categories, at similar rates?)")
    print("=" * 90)
    print(f"{'column':<30} {'real_uniq':>10} {'syn_uniq':>9} {'missing':>8} {'coverage%':>10} {'max_freq_gap':>13}")
    print("-" * 90)

    rows = []
    for col in cat_cols:
        real_vals = real_df[col].astype(str)
        syn_vals = syn_df[col].astype(str)

        real_uniq = set(real_vals.unique())
        syn_uniq = set(syn_vals.unique())
        missing = real_uniq - syn_uniq
        coverage_pct = 100.0 * (len(real_uniq) - len(missing)) / max(len(real_uniq), 1)

        # frequency comparison for the values both share
        real_freq = real_vals.value_counts(normalize=True)
        syn_freq = syn_vals.value_counts(normalize=True)
        common = real_freq.index.intersection(syn_freq.index)
        if len(common) > 0:
            freq_gap = (real_freq[common] - syn_freq[common]).abs().max()
        else:
            freq_gap = float('nan')

        rows.append({
            'column': col, 'real_uniq': len(real_uniq), 'syn_uniq': len(syn_uniq),
            'missing': len(missing), 'coverage_pct': coverage_pct, 'max_freq_gap': freq_gap,
            'missing_values_sample': list(missing)[:5],
        })
        print(f"{str(col):<30} {len(real_uniq):>10} {len(syn_uniq):>9} {len(missing):>8} "
              f"{coverage_pct:>9.1f}% {freq_gap:>13.4f}")

    df = pd.DataFrame(rows).sort_values('coverage_pct')
    print("\nWorst-covered categorical columns (lowest coverage % first):")
    print(df[['column', 'real_uniq', 'syn_uniq', 'missing', 'coverage_pct']].head(10).to_string(index=False))
    return df


def numerical_coverage(real_df, syn_df, num_cols, out_dir=None, plot_worst_n=6):
    print("\n" + "=" * 90)
    print("NUMERICAL COVERAGE  (does synthetic reach the same range/spread as real?)")
    print("=" * 90)
    print(f"{'column':<25} {'real_range':>22} {'syn_range':>22} {'range_cov%':>10} {'KS_stat':>8}")
    print("-" * 90)

    rows = []
    for col in num_cols:
        r = pd.to_numeric(real_df[col], errors='coerce').dropna().to_numpy()
        s = pd.to_numeric(syn_df[col], errors='coerce').dropna().to_numpy()
        if len(r) == 0 or len(s) == 0:
            continue

        r_min, r_max = r.min(), r.max()
        s_min, s_max = s.min(), s.max()

        # how much of the real RANGE does synthetic actually reach?
        real_span = r_max - r_min if r_max > r_min else 1e-9
        overlap_lo = max(r_min, s_min)
        overlap_hi = min(r_max, s_max)
        overlap = max(0.0, overlap_hi - overlap_lo)
        range_cov_pct = 100.0 * overlap / real_span

        ks_stat, _ = ks_2samp(r, s)

        rows.append({
            'column': col, 'r_min': r_min, 'r_max': r_max, 's_min': s_min, 's_max': s_max,
            'range_cov_pct': range_cov_pct, 'ks_stat': ks_stat,
        })
        print(f"{str(col):<25} [{r_min:>8.2f},{r_max:>8.2f}] [{s_min:>8.2f},{s_max:>8.2f}] "
              f"{range_cov_pct:>9.1f}% {ks_stat:>8.3f}")

    df = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    print("\nWorst-covered numerical columns (highest KS statistic = most different, first):")
    print(df[['column', 'range_cov_pct', 'ks_stat']].head(10).to_string(index=False))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            worst = df.head(plot_worst_n)['column'].tolist()
            for col in worst:
                r = pd.to_numeric(real_df[col], errors='coerce').dropna()
                s = pd.to_numeric(syn_df[col], errors='coerce').dropna()
                plt.figure(figsize=(6, 4))
                bins = np.histogram_bin_edges(np.concatenate([r, s]), bins=50)
                plt.hist(r, bins=bins, alpha=0.5, density=True, label='real')
                plt.hist(s, bins=bins, alpha=0.5, density=True, label='synthetic')
                plt.title(f"{col}: real vs synthetic")
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(out_dir, f"hist_{str(col).replace('/', '_')}.png")
                plt.savefig(fname, dpi=120)
                plt.close()
                print(f"  saved {fname}")
        except ImportError:
            print("  (matplotlib not available, skipping histogram plots)")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--synthetic_csv', required=True)
    ap.add_argument('--out_dir', default=None, help='if set, saves histogram PNGs here')
    args = ap.parse_args()

    info = load_info(args.dataname)
    real_df = load_real(args.dataname)
    syn_df = pd.read_csv(args.synthetic_csv)

    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))

    num_cols, cat_cols = get_col_groups(info, real_df)
    print(f"Dataset: {args.dataname}   numerical cols: {len(num_cols)}   categorical cols: {len(cat_cols)}")

    if cat_cols:
        categorical_coverage(real_df, syn_df, cat_cols)
    if num_cols:
        numerical_coverage(real_df, syn_df, num_cols, out_dir=args.out_dir)


if __name__ == "__main__":
    main()