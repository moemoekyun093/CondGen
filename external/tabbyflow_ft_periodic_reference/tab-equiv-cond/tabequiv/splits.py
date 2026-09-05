"""[ATTIC from 2026-08-25] Frozen 60/20/20 splits, TRAIN-only column statistics, and the split manifest.

**The project split is now data90/ (see load_split below and tabequiv/splits90.py).**
Everything else in this module (make/write split, stats, manifest) describes the RETIRED
60/20/20 protocol and operates on data/<ds>/attic_60_20_20/.

These indices are the SINGLE SOURCE OF TRUTH for all downstream work: queries,
conditional references, and any later training/selection all index into them.

Mechanism (Task 1a). Reuses tabsyn/TabDiff's category-coverage retry
(``baselines/tabsyn/process_dataset.py::train_val_test_split``, imported read-only in
spirit -- the routine is reimplemented here because the original is hardwired to a
two-way 90/10 split and to DataFrame slicing):

* shuffle indices under ``np.random.seed(seed)`` starting at 1234;
* if any categorical level is missing from ANY split, increment the seed and reshuffle;
* record the FINAL seed actually used.

COVERAGE POLICY (and why it is not "all three splits"). tabsyn checks coverage on the
TRAIN split only, and that is the strongest guarantee actually achievable here: a level
occurring FEWER THAN 3 TIMES in the full table cannot appear in all three splits, no
matter the seed. Measured on these datasets:

    shoppers  Browser=9 appears ONCE in 12330 rows; TrafficType 17 and 12 also once
    default   PAY_2/PAY_4/PAY_5/PAY_6 each carry 1-2 such levels
    adult     native.country carries one
    diabetes  diag_1/diag_2/diag_3 carry 144/201/195 such levels

Requiring three-way coverage therefore fails by construction (verified: shoppers
exhausted 500 seeds). The retry enforces TRAIN coverage -- the tabsyn guarantee, and the
one that matters, since column statistics and predicate sizing are TRAIN-only -- and the
manifest additionally RECORDS val/test level gaps per column so any query conditioned on
a level missing from a split can be identified rather than silently mis-realized.

No target stratification -- plain shuffle, matching the dominant convention across the
six baselines (see ``docs/protocol_survey.md``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

SEED0 = 1234
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20      # test takes the remainder
DATASETS = ("shoppers", "magic", "default", "news", "beijing", "adult", "diabetes")

#: adult ships an official test split; we honour it and carve val from the train part.
PROVIDED_TEST = {"adult"}


# ---------------------------------------------------------------------------
def canonical_source_rows(ds: str) -> int:
    """Row count of the CANONICAL source table -- the number the split MUST cover.

    Guard against silent scope loss (Task 2). Preference order:
      1. the dataset's own raw table where one is shipped whole
         (diabetes ships ``diabetic_data.csv`` with all 101,766 rows);
      2. otherwise the sum of every provided partition CSV.
    """
    d = DATA / ds
    raw = {"diabetes": "diabetic_data.csv"}.get(ds)
    if raw and (d / raw).exists():
        return sum(1 for _ in (d / raw).open()) - 1
    n = 0
    for part in ("train.csv", "val.csv", "test.csv"):
        if (d / part).exists():
            n += sum(1 for _ in (d / part).open()) - 1
    return n


def load_frame(ds: str) -> tuple[pd.DataFrame, dict]:
    """Full table (EVERY provided partition concatenated), plus info.

    BUG FIXED HERE (Task 1): this previously concatenated ``train.csv`` + ``test.csv``
    only. diabetes is the sole dataset that also ships ``val.csv``, so its table was
    silently truncated to 61,059 + 20,354 = 81,413 of 101,766 rows -- the whole val
    partition (20,353 rows) was dropped from the split, the stats and the query suite.
    Every partition present is now concatenated, in train, val, test order.

    The concatenation order defines the row order the frozen indices refer to, so it
    must stay stable.
    """
    d = DATA / ds
    info = json.loads((d / "info.json").read_text())
    frames, sizes = [], {}
    for part in ("train", "val", "test"):
        f = d / f"{part}.csv"
        if f.exists():
            df_p = pd.read_csv(f)
            frames.append(df_p)
            sizes[part] = int(len(df_p))
    full = pd.concat(frames, axis=0, ignore_index=True)
    info = dict(info)
    info["provided_partition_sizes"] = sizes
    info["n_provided_train"] = sizes.get("train", 0)
    info["n_provided_test"] = sizes.get("test", 0)
    if "metadata" not in info:      # datasets staged outside the TabDiff/TabSyn lineage (fb-comments): synthesize
        cols = info["column_names"]; num = set(info["num_col_idx"]); cat = set(info["cat_col_idx"])
        tgt = set(info.get("target_col_idx", []))
        info["metadata"] = {"columns": {str(i): ({"sdtype": "numerical", "computer_representation": "Float"}
                                                 if (i in num or (i in tgt and info.get("task_type") == "regression"))
                                                 else {"sdtype": "categorical"}) for i in range(len(cols))}}
        info.setdefault("column_info", {str(i): {} for i in range(len(cols))})
        info.setdefault("idx_mapping", {str(i): i for i in range(len(cols))})
        info.setdefault("inverse_idx_mapping", {str(i): i for i in range(len(cols))})
        info.setdefault("idx_name_mapping", {str(i): c for i, c in enumerate(cols)})
        info.setdefault("int_col_idx", []); info.setdefault("int_columns", []); info.setdefault("int_col_idx_wrt_num", [])
    return full, info


def cat_columns(df: pd.DataFrame, info: dict) -> list[str]:
    """Categorical + target columns by NAME (the target is categorical for binclass)."""
    names = list(df.columns)
    idx = list(info["cat_col_idx"])
    if info.get("task_type") != "regression":
        idx = idx + list(info.get("target_col_idx", []))
    return [names[i] for i in sorted(set(idx)) if i < len(names)]


def _covered(df: pd.DataFrame, parts: dict[str, np.ndarray], cols: list[str]) -> bool:
    """True iff every categorical level of the FULL table appears in TRAIN.

    This is tabsyn's criterion (``process_dataset.py::train_val_test_split``) and the
    only one satisfiable in general -- see the module docstring on levels occurring
    fewer than three times. val/test gaps are reported, not retried.
    """
    for c in cols:
        allv = set(df[c].astype(str))
        if set(df[c].astype(str).iloc[parts["train"]]) != allv:
            return False
    return True



def _n_missing(df, parts, cols) -> int:
    """Number of (column, level) pairs of the full table absent from TRAIN."""
    miss = 0
    for c in cols:
        allv = set(df[c].astype(str))
        miss += len(allv - set(df[c].astype(str).iloc[parts["train"]]))
    return miss


def _best_effort(df, cols, pool, n_train, test_idx, max_tries, n_val=None):
    """Seed minimising TRAIN level gaps, when full coverage is unreachable.

    Some tables cannot satisfy even tabsyn's train-only criterion: a level occurring
    ONCE lands in train with probability 0.60, and diabetes carries 353 such levels
    across diag_1/diag_2/diag_3 and others, so P(all in train) ~ 0.6^353 ~ 5e-79.
    Rather than fail (which would leave the dataset unusable) or silently drop the
    check, we pick the seed with the FEWEST missing train levels and record the
    residual in the manifest as ``train_level_gaps``.
    """
    best = None
    for t in range(max_tries):
        seed = SEED0 + t
        idx = pool.copy()
        np.random.seed(seed)
        np.random.shuffle(idx)
        if test_idx is None:
            parts = {"train": idx[:n_train],
                     "val": idx[n_train:n_train + n_val],
                     "test": idx[n_train + n_val:]}
        else:
            parts = {"train": idx[:n_train], "val": idx[n_train:], "test": test_idx}
        m = _n_missing(df, parts, cols)
        if best is None or m < best[0]:
            best = (m, seed, parts)
        if m == 0:
            break
    return best[1], best[2]


def coverage_report(df, parts, cols) -> dict:
    """Per-split, per-column level counts -- persisted so the check is auditable."""
    out = {}
    for c in cols:
        allv = set(df[c].astype(str))
        out[c] = {"n_levels_full": len(allv), **{
            k: {"n_levels": len(set(df[c].astype(str).iloc[idx])),
                "covers_all": set(df[c].astype(str).iloc[idx]) == allv}
            for k, idx in parts.items()}}
    return out


def make_split(ds: str, max_tries: int = 500) -> dict:
    """-> dict with the three index arrays, the final seed, sizes and coverage."""
    df, info = load_frame(ds)
    cols = cat_columns(df, info)
    n = len(df)

    if ds in PROVIDED_TEST:
        # Honour the provided test split; carve val out of the provided train rows only.
        n_tr_prov = info["n_provided_train"]
        test_idx = np.arange(n_tr_prov, n)
        pool = np.arange(n_tr_prov)
        # 60/20 of the NON-test rows -> 75/25 of the pool
        n_train = int(round(len(pool) * (TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC))))
        seed = SEED0
        for _ in range(max_tries):
            rng_idx = pool.copy()
            np.random.seed(seed)
            np.random.shuffle(rng_idx)
            parts = {"train": rng_idx[:n_train], "val": rng_idx[n_train:],
                     "test": test_idx}
            if _covered(df, parts, cols):
                break
            seed += 1
        else:
            seed, parts = _best_effort(df, cols, pool, n_train, test_idx, max_tries)
        scheme = (f"provided test ({len(test_idx)} rows); remaining {len(pool)} rows "
                  f"split {n_train}/{len(pool)-n_train} = "
                  f"{n_train/len(pool):.2%}/{1-n_train/len(pool):.2%} train/val")
    else:
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        seed = SEED0
        for _ in range(max_tries):
            idx = np.arange(n)
            np.random.seed(seed)
            np.random.shuffle(idx)
            parts = {"train": idx[:n_train],
                     "val": idx[n_train:n_train + n_val],
                     "test": idx[n_train + n_val:]}
            if _covered(df, parts, cols):
                break
            seed += 1
        else:
            seed, parts = _best_effort(df, cols, np.arange(n), n_train, None,
                                       max_tries, n_val=n_val)
        scheme = f"60/20/20 of all {n} rows"

    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return {
        "dataset": ds, "seed_start": SEED0, "seed_final": int(seed),
        "seed_increments": int(seed - SEED0), "scheme": scheme,
        "n_total": int(n),
        "n_source_rows": int(canonical_source_rows(ds)),
        "guard_ok": bool(int(n) == int(canonical_source_rows(ds))),
        "sizes": {k: int(len(v)) for k, v in parts.items()},
        "fractions": {k: round(len(v) / n, 6) for k, v in parts.items()},
        "cat_columns_checked": cols,
        "coverage_criterion": "train must contain every level of the full table "
                              "(tabsyn's criterion); val/test gaps recorded not retried",
        "coverage": coverage_report(df, parts, cols),
        "rare_levels": {                      # levels that CANNOT reach all 3 splits
            c: int((df[c].astype(str).value_counts() < 3).sum())
            for c in cols if int((df[c].astype(str).value_counts() < 3).sum())
        },
        "singleton_levels": {                 # levels appearing exactly once
            c: int((df[c].astype(str).value_counts() == 1).sum())
            for c in cols if int((df[c].astype(str).value_counts() == 1).sum())
        },
        "train_level_gaps": int(_n_missing(df, parts, cols)),
        "data_sha256": h.hexdigest(),
        "_parts": parts,
    }


def write_split(ds: str) -> dict:
    m = make_split(ds)
    # Task 2: fail loudly rather than let a partition go missing again.
    if not m["guard_ok"]:
        raise RuntimeError(
            f"{ds}: n_total={m['n_total']} != n_source_rows={m['n_source_rows']} "
            f"-- the split does not cover the canonical table")
    out = DATA / ds / ATTIC_SPLIT_DIRNAME / "splits"
    out.mkdir(parents=True, exist_ok=True)
    parts = m.pop("_parts")
    for k, v in parts.items():
        np.save(out / f"{k}_idx.npy", np.sort(v))
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


def load_split(ds: str) -> dict[str, np.ndarray]:
    """THE project split (2026-08-25 onward): ``data90/<ds>/splits`` -- AA_code-identical
    90/10 trainval/test (TabDiff's native partition) with a SEEDED 10%-of-train val carve
    (train 81% core / val 9% / test 10%). Indices index the frame from :func:`load_frame`.
    Row-for-row identity with AA_code is verified by ``scripts/verify_data90.py``.

    Returns ``train`` / ``val`` / ``test``. The 90% ``trainval`` (= the papers' training
    set) is available from :func:`tabequiv.splits90.load_split90`.

    The former 60/20/20 split is ATTIC: :func:`load_split_attic`.
    """
    from tabequiv.splits90 import load_split90
    sp = load_split90(ds)
    return {k: sp[k] for k in ("train", "val", "test")}


ATTIC_SPLIT_DIRNAME = "attic_60_20_20"


def load_split_attic(ds: str) -> dict[str, np.ndarray]:
    """The RETIRED frozen 60/20/20 split (``data/<ds>/attic_60_20_20/splits``). Only for
    reproducing the 2026-08 pilot; never for new work."""
    out = DATA / ds / ATTIC_SPLIT_DIRNAME / "splits"
    if not out.exists():
        raise FileNotFoundError(f"{out}: the attic 60/20/20 split is not present")
    return {k: np.load(out / f"{k}_idx.npy") for k in ("train", "val", "test")}


# ---------------------------------------------------------------------------
# 1b. TRAIN-ONLY column statistics
# ---------------------------------------------------------------------------
QUANTILES = np.round(np.linspace(0.0, 1.0, 101), 4)   # percentile grid for the CDF


def compute_stats(ds: str) -> dict:
    """Numeric quantile function + moments, categorical frequencies -- TRAIN ONLY.

    Leakage rule: nothing here may touch val or test. The query generator sizes its
    predicates from these numbers, so any val/test information leaking in would make the
    downstream selectivity targets self-fulfilling.
    """
    df, info = load_frame(ds)
    tr = load_split_attic(ds)["train"]      # ATTIC stats; data90 has its own (splits90.compute_stats90)
    sub = df.iloc[tr]
    names = list(df.columns)
    num_idx = list(info["num_col_idx"])
    cat_idx = list(info["cat_col_idx"])
    if info.get("task_type") == "regression":
        num_idx = sorted(set(num_idx) | set(info.get("target_col_idx", [])))
    else:
        cat_idx = sorted(set(cat_idx) | set(info.get("target_col_idx", [])))

    numeric = {}
    for i in num_idx:
        c = names[i]
        v = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if not len(v):
            continue
        numeric[c] = {
            "col_idx": int(i), "n": int(len(v)),
            "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "quantile_levels": QUANTILES.tolist(),
            "quantile_values": np.quantile(v, QUANTILES).tolist(),
        }

    categorical = {}
    for i in cat_idx:
        c = names[i]
        vc = sub[c].astype(str).value_counts()
        categorical[c] = {
            "col_idx": int(i), "n": int(vc.sum()),
            "n_levels": int(len(vc)),
            "freq": {str(k): int(x) for k, x in vc.items()},
            "prob": {str(k): float(x / vc.sum()) for k, x in vc.items()},
        }
    return {"dataset": ds, "computed_on": "train", "n_train": int(len(tr)),
            "numeric": numeric, "categorical": categorical}


def write_stats(ds: str) -> dict:
    st = compute_stats(ds)
    out = DATA / ds / "stats"
    out.mkdir(parents=True, exist_ok=True)
    (out / "numeric.json").write_text(json.dumps(
        {"dataset": ds, "computed_on": "train", "columns": st["numeric"]}, indent=2) + "\n")
    (out / "categorical.json").write_text(json.dumps(
        {"dataset": ds, "computed_on": "train", "columns": st["categorical"]}, indent=2) + "\n")
    return st


if __name__ == "__main__":
    import sys
    for ds in (sys.argv[1:] or list(DATASETS)):
        m = write_split(ds)
        st = write_stats(ds)
        tr_bad = [c for c, v in m["coverage"].items() if not v["train"]["covers_all"]]
        vt_bad = [c for c, v in m["coverage"].items()
                  if not (v["val"]["covers_all"] and v["test"]["covers_all"])]
        print(f"{ds:10s} seed={m['seed_final']:5d} (+{m['seed_increments']:2d}) "
              f"sizes={m['sizes']} num={len(st['numeric']):2d} cat={len(st['categorical']):2d} "
              f"train_gaps={len(tr_bad)} val/test_gaps={len(vt_bad)}")
