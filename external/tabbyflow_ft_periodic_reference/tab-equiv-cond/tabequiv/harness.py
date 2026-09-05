"""Training / checkpoint / selection harness (Task 4).

SKELETON + smoke test only -- this module does not launch full multi-backbone training.

Policy implemented here:

* **EMA** with fixed decay (``beta`` in 0.999-0.9999), following the ``ema_pytorch``
  contract used by DEFT: ``update_after_step`` (no averaging during early warmup, where
  weights move fast and would poison the average) and ``update_every`` (update every N
  optimizer steps). Sampling/evaluation uses the EMA weights.
  For reference, TabDiff's own trainer uses a plain 0.997 decay applied every step
  (``TabDiff/tabdiff/trainer.py:32,222``); this harness makes the schedule explicit.
* **Checkpoint every 500 steps**, storing per-checkpoint weights (both raw and EMA).
* **At every checkpoint, run the FULL metric union on VAL**, unconditional, and append a
  row to a CSV.
* **No metric is hardcoded as the selection criterion.** Every metric key is stored, so
  "best val model" can be decided afterwards by whichever metric turns out to matter.
  :func:`select_best` is a helper over the stored CSV, not a policy baked into training.

Splits come from ``data/<ds>/splits/`` (``tabequiv/splits.py``) -- the single source of
truth. VAL is used for checkpoint selection; TEST is never touched here.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    """Fixed-decay exponential moving average of a module's parameters.

    Mirrors ``ema_pytorch.EMA``'s scheduling knobs:

    * ``beta``            -- decay per update (0.999-0.9999 typical);
    * ``update_after_step`` -- steps to skip before averaging begins;
    * ``update_every``    -- apply an update only every N steps.

    The shadow model is a detached deep copy; ``copy_to`` swaps the averaged weights
    into a live model for evaluation/sampling.
    """

    def __init__(self, model: torch.nn.Module, beta: float = 0.999,
                 update_after_step: int = 100, update_every: int = 10):
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0,1), got {beta}")
        self.beta = float(beta)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.step = 0
        self.n_updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> bool:
        """Advance one step; -> True if an averaging update was applied."""
        self.step += 1
        if self.step <= self.update_after_step:
            # warmup: track the live weights exactly, so the average never starts from
            # a stale initialization
            self.ema_model.load_state_dict(model.state_dict())
            return False
        if self.step % self.update_every:
            return False
        for pe, pm in zip(self.ema_model.parameters(), model.parameters()):
            pe.mul_(self.beta).add_(pm.detach(), alpha=1.0 - self.beta)
        for be, bm in zip(self.ema_model.buffers(), model.buffers()):
            be.copy_(bm)
        self.n_updates += 1
        return True

    def state_dict(self):
        return {"beta": self.beta, "step": self.step, "n_updates": self.n_updates,
                "ema": self.ema_model.state_dict()}


# ---------------------------------------------------------------------------
@dataclass
class HarnessConfig:
    dataset: str
    backbone: str
    checkpoint_every: int = 500
    max_steps: int = 2000
    ema_beta: float = 0.999
    ema_update_after_step: int = 100
    ema_update_every: int = 10
    n_val_sample: int = 2000        # rows generated per checkpoint eval
    fast_metrics: bool = False      # full union by default (the stated policy)
    out_dir: str = ""

    def resolved_out(self) -> Path:
        p = Path(self.out_dir) if self.out_dir else \
            REPO / "experiments" / "06_harness" / f"{self.dataset}__{self.backbone}"
        p.mkdir(parents=True, exist_ok=True)
        return p


class CheckpointRecorder:
    """Per-checkpoint: save weights, evaluate the FULL metric union on VAL, append CSV.

    Every metric key is stored. Selection happens later (:func:`select_best`) so no
    single metric is privileged at training time.
    """

    def __init__(self, cfg: HarnessConfig, info: dict, val_frame: pd.DataFrame):
        self.cfg = cfg
        self.info = info
        self.val = val_frame
        self.out = cfg.resolved_out()
        (self.out / "checkpoints").mkdir(exist_ok=True)
        self.rows: list[dict] = []
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    def _metrics(self):
        from tabequiv.eval import suite
        return suite.FAST_METRICS if self.cfg.fast_metrics else suite.ALL_METRICS

    def record(self, step: int, model, ema: EMA | None, sample_fn) -> dict:
        """``sample_fn(model, n) -> DataFrame`` in the val frame's column order."""
        from tabequiv.eval import suite

        t0 = time.time()
        eval_model = ema.ema_model if ema is not None else model
        ckpt = self.out / "checkpoints" / f"step_{step:07d}.pt"
        torch.save({"step": step,
                    "model": model.state_dict(),
                    "ema": (ema.state_dict() if ema is not None else None)}, ckpt)
        t_save = time.time() - t0

        t1 = time.time()
        syn = sample_fn(eval_model, self.cfg.n_val_sample)
        t_sample = time.time() - t1

        t2 = time.time()
        res = suite.evaluate_frames(self.val, syn, self.info,
                                    metrics=self._metrics(), skip_on_error=True)
        t_eval = time.time() - t2

        row = {"step": step, "checkpoint": str(ckpt.relative_to(self.out)),
               "used_ema": ema is not None,
               "seconds_save": round(t_save, 3),
               "seconds_sample": round(t_sample, 3),
               "seconds_eval": round(t_eval, 3),
               "seconds_total": round(time.time() - t0, 3),
               **{k: v for k, v in res.items()}}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out / "val_metrics.csv", index=False)
        return row


def select_best(csv_path, metric_key: str, mode: str = "min") -> dict:
    """Pick the best checkpoint by ANY stored metric, after the fact.

    Deliberately not called during training: the harness stores everything so the
    criterion can be chosen once it is known which metric matters.
    """
    df = pd.read_csv(csv_path)
    if metric_key not in df.columns:
        raise KeyError(f"{metric_key!r} not in {csv_path}; "
                       f"available: {[c for c in df.columns if '/' in c][:20]}")
    d = df.dropna(subset=[metric_key])
    if d.empty:
        raise ValueError(f"no non-null values for {metric_key!r}")
    i = d[metric_key].idxmin() if mode == "min" else d[metric_key].idxmax()
    return df.loc[i].to_dict()


# ---------------------------------------------------------------------------
def load_val_frame(dataset: str):
    """VAL rows of the full table, in original column order, + info."""
    from tabequiv.splits import load_frame, load_split
    df, info = load_frame(dataset)
    idx = load_split(dataset)["val"]
    return df.iloc[idx].reset_index(drop=True), info
