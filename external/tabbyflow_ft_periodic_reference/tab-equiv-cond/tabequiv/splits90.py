"""Paper-faithful 90/10 split + a SEEDED 10%-of-train validation carve, under ``data90/``.

This is a second, fully DETACHED split root. Nothing here reads or writes anything under
``data/<ds>/{splits,stats,queries,queries_full}`` (the 60/20/20 pilot root); only the raw
CSVs and ``info.json`` are shared. Runs must never mix the two: every consumer of this
module gets its indices from ``load_split90`` and ``ROOT`` only.

Layout: **train 81% / val 9% / test 10%** of the full table, built as the baselines do:

1. **The published 90/10 is the PROVIDED partition, taken verbatim.** ``data/<ds>/
   train.csv`` + ``test.csv`` are the OUTPUT of tabsyn's ``process_dataset.py`` (row
   counts equal ``int(n*0.9)`` / the rest, rows shuffled relative to the raw file) --
   the exact rows TabSyn/TabDiff/ef-vfm/tab-flow/tabpc were published on. They are NOT
   reshuffled: re-running tabsyn's seed-1234 loop on the concatenation would apply the
   same seed to a different row order and yield a different split.
2. **tabpc rule** (``baselines/tabpc/src/util.py:232``): carve ``int(n_train * 0.9)``
   rows of the provided train as the core, the remainder is val. tabpc's carve is an
   UNSEEDED ``np.random.permutation``; here it is seeded (1234, bumped on retry) and
   recorded.
3. **Coverage is enforced on the 81% CORE**: the models that hold val out (tabpc,
   TabSyn-VAE) must still see every level. Only the carve is retried -- train+val vs
   test membership never changes.
4. **Provided val partitions are honoured** (diabetes ships TabDiff's train/val/test):
   no carve at all. adult keeps its official test; fb-comments keeps rtdl's partition.

Which rows a model TRAINS on (core 81% vs core+val 90%) is a training-stage decision
and deliberately not encoded here: both ``train_idx`` and ``trainval_idx`` are stored so
either composition is one load away. Column statistics are computed on the 81% core so
that predicate sizing touches nothing any model holds out.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabequiv.splits import (REPO, DATA, DATASETS, SEED0, PROVIDED_TEST,  # noqa: E402
                             QUANTILES, canonical_source_rows, cat_columns,
                             coverage_report, load_frame, _n_missing)

ROOT = REPO / "data90"                       # the detached root
ALL_DATASETS = tuple(DATASETS) + ("fb-comments",)
TRAIN90_FRAC = 0.9                           # tabsyn: int(n * 0.9)
CORE_FRAC = 0.9                              # tabpc: int(n_train * 0.9)
LAYOUT = "train 81% (core) / val 9% / test 10%"


# ---------------------------------------------------------------------------
def _carve(pool: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """tabpc's carve on a SEEDED permutation: core = first int(n*0.9), val = the rest."""
    np.random.seed(seed)
    perm = pool[np.random.permutation(len(pool))]
    n_core = int(len(pool) * CORE_FRAC)
    return perm[:n_core], perm[n_core:]


def make_split90(ds: str, max_tries: int = 500) -> dict:
    """train/val/test index arrays honouring the PROVIDED partitions.

    ``data/<ds>/train.csv`` + ``test.csv`` are already the published split: for the
    tabsyn-lineage datasets they are the OUTPUT of tabsyn's ``process_dataset.py``
    (row counts match ``int(n*0.9)`` exactly and the rows are shuffled relative to the
    raw file), adult ships its official test, diabetes ships TabDiff's train/val/test,
    fb-comments ships rtdl's train/test. Re-shuffling their concatenation would produce
    a DIFFERENT split under the same seed, so nothing is reshuffled here:

    * test  = the provided test rows, always;
    * val   = the provided val rows where a val partition exists (diabetes), else a
      seeded tabpc carve (``int(n*0.9)``) of the provided train, retried over seeds
      until the 81% core holds every categorical level (best-effort if impossible);
    * train = the rest of the provided train (the core).

    The retry only moves rows between train and val -- train+val vs test membership is
    the published split for every dataset.
    """
    df, info = load_frame(ds)
    cols = cat_columns(df, info)
    n = len(df)
    sizes = info["provided_partition_sizes"]
    n_tr, n_va = sizes.get("train", 0), sizes.get("val", 0)
    train_prov = np.arange(n_tr)                       # concat order: train, val, test
    test_idx = np.arange(n_tr + n_va, n)

    # fb-comments: rtdl's val+test were merged into test.csv in val,test order
    # (info.json: "concatenated in train,val,test order"). Recover rtdl's own 80/10/10
    # from the recorded sizes instead of carving -- a real held-out val, paper-faithful
    # to the rtdl/TabDDPM release, and the closest of all eight to the 81/9/10 target.
    rtdl = info.get("source", {}).get("rtdl_info")
    if rtdl and not n_va and n_tr == rtdl["train_size"] \
            and len(test_idx) == rtdl["val_size"] + rtdl["test_size"]:
        n_va = rtdl["val_size"]
        test_idx = np.arange(n_tr + n_va, n)
        provided_val_note = "rtdl val recovered from the head of test.csv"
    else:
        provided_val_note = None

    seed_final, best = None, None
    if n_va:                                           # provided val: keep it verbatim
        parts = {"train": train_prov, "val": np.arange(n_tr, n_tr + n_va),
                 "test": test_idx}
        scheme = (f"provided train/val/test kept verbatim "
                  f"({n_tr}/{n_va}/{len(test_idx)}); no carve, no reshuffle"
                  + (f" [{provided_val_note}]" if provided_val_note else ""))
        seed_final = None
    else:
        parts = None
        for t in range(max_tries):
            seed = SEED0 + t
            core, val = _carve(train_prov, seed)
            p = {"train": core, "val": val, "test": test_idx}
            m = _n_missing(df, p, cols)                # gaps in the 81% CORE
            if best is None or m < best[0]:
                best = (m, seed, p)
            if m == 0:
                seed_final, parts = seed, p
                break
        if parts is None:                              # best effort
            _, seed_final, parts = best
        scheme = (f"provided train ({n_tr}) / test ({len(test_idx)}) kept; val = "
                  f"seeded tabpc carve int({n_tr}*{CORE_FRAC}) -> "
                  f"{len(parts['train'])}/{len(parts['val'])} core/val")

    n_all_tr = n_tr + n_va
    tabsyn_90_10 = (ds not in PROVIDED_TEST and not n_va
                    and n_all_tr == int(n * TRAIN90_FRAC))
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return {
        "dataset": ds, "root": str(ROOT.relative_to(REPO)), "layout": LAYOUT,
        "trainval_test_from_provided_partitions": True,
        "provided_partition_sizes": sizes,
        "provided_split_is_tabsyn_90_10": bool(tabsyn_90_10),
        "seed_start": SEED0,
        "seed_final": (None if seed_final is None else int(seed_final)),
        "seed_increments": (None if seed_final is None else int(seed_final - SEED0)),
        "scheme": scheme,
        "n_total": int(n),
        "n_source_rows": int(canonical_source_rows(ds)),
        "guard_ok": bool(int(n) == int(canonical_source_rows(ds))),
        "sizes": {k: int(len(v)) for k, v in parts.items()},
        "n_trainval": int(len(parts["train"]) + len(parts["val"])),
        "fractions": {k: round(len(v) / n, 6) for k, v in parts.items()},
        "cat_columns_checked": cols,
        "coverage_criterion": "the 81% CORE train must contain every level of the full "
                              "table; only the train/val carve is retried, train+val vs "
                              "test membership is the provided split",
        "coverage": coverage_report(df, parts, cols),
        "rare_levels": {
            c: int((df[c].astype(str).value_counts() < 3).sum())
            for c in cols if int((df[c].astype(str).value_counts() < 3).sum())
        },
        "singleton_levels": {
            c: int((df[c].astype(str).value_counts() == 1).sum())
            for c in cols if int((df[c].astype(str).value_counts() == 1).sum())
        },
        "train_level_gaps": int(_n_missing(df, parts, cols)),
        "data_sha256": h.hexdigest(),
        "_parts": parts,
    }


def write_split90(ds: str) -> dict:
    m = make_split90(ds)
    if not m["guard_ok"]:
        raise RuntimeError(
            f"{ds}: n_total={m['n_total']} != n_source_rows={m['n_source_rows']} "
            f"-- the split does not cover the canonical table")
    out = ROOT / ds / "splits"
    out.mkdir(parents=True, exist_ok=True)
    parts = m.pop("_parts")
    for k, v in parts.items():
        np.save(out / f"{k}_idx.npy", np.sort(v))
    np.save(out / "trainval_idx.npy",
            np.sort(np.concatenate([parts["train"], parts["val"]])))
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


def load_split90(ds: str) -> dict[str, np.ndarray]:
    """train (81% core), val (9%), test (10%), trainval (90% = tabsyn's train)."""
    out = ROOT / ds / "splits"
    if not out.exists():
        raise FileNotFoundError(f"{out} -- run tabequiv/splits90.py first "
                                f"(this is the data90 root, NOT data/<ds>/splits)")
    return {k: np.load(out / f"{k}_idx.npy")
            for k in ("train", "val", "test", "trainval")}


# ---------------------------------------------------------------------------
# column statistics -- 81% CORE only
# ---------------------------------------------------------------------------
def compute_stats90(ds: str) -> dict:
    """Numeric quantile function + moments, categorical frequencies -- CORE train only.

    Sizing predicates from the core means no held-out row (val for tabpc/TabSyn-VAE,
    test for everyone) leaks into the selectivity targets.
    """
    df, info = load_frame(ds)
    tr = load_split90(ds)["train"]
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
            "col_idx": int(i), "n": int(vc.sum()), "n_levels": int(len(vc)),
            "freq": {str(k): int(x) for k, x in vc.items()},
            "prob": {str(k): float(x / vc.sum()) for k, x in vc.items()},
        }
    return {"dataset": ds, "computed_on": "train (81% core)", "n_train": int(len(tr)),
            "numeric": numeric, "categorical": categorical}


def write_stats90(ds: str) -> dict:
    st = compute_stats90(ds)
    out = ROOT / ds / "stats"
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("numeric", "categorical"):
        (out / f"{kind}.json").write_text(json.dumps(
            {"dataset": ds, "computed_on": st["computed_on"], "columns": st[kind]},
            indent=2) + "\n")
    return st


def load_stats90(ds: str) -> tuple[dict, dict]:
    d = ROOT / ds / "stats"
    return (json.loads((d / "numeric.json").read_text())["columns"],
            json.loads((d / "categorical.json").read_text())["columns"])


if __name__ == "__main__":
    import sys
    for ds in (sys.argv[1:] or list(ALL_DATASETS)):
        m = write_split90(ds)
        st = write_stats90(ds)
        vt_bad = [c for c, v in m["coverage"].items()
                  if not (v["val"]["covers_all"] and v["test"]["covers_all"])]
        print(f"{ds:12s} seed={m['seed_final']} (+{m['seed_increments']}) "
              f"provided_is_tabsyn_90_10={m['provided_split_is_tabsyn_90_10']!s:5s} "
              f"sizes={m['sizes']} core_gaps={m['train_level_gaps']} "
              f"val/test_gap_cols={len(vt_bad)} "
              f"num={len(st['numeric'])} cat={len(st['categorical'])}")
