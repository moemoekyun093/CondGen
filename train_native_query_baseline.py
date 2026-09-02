"""Train a dataset-agnostic DiffPuter-release or GReaT query baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from tabdiff.baseline_data import load_baseline_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("diffputer", "great"), required=True)
    parser.add_argument("--dataname")
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
    parser.add_argument("--info-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument("--great-root", default="baselines/great")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hid-dim", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-0", type=float, default=1e-4)
    parser.add_argument("--beta-t", type=float, default=2e-2)
    parser.add_argument("--llm", default="distilgpt2")
    parser.add_argument("--float-precision", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_diffputer(args, table, output_dir: Path) -> None:
    harpoon_root = Path(args.harpoon_root).resolve()
    sys.path.insert(0, str(harpoon_root))
    from model import MLPDiffusion
    from utils import calc_diffusion_hyperparams

    epochs = args.epochs if args.epochs is not None else 1000
    batch_size = args.batch_size if args.batch_size is not None else 1024
    encoded = table.encode(table.train)
    mean, std = table.standardization()
    tensor = torch.tensor((encoded - mean) / std, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True, num_workers=4)
    device = torch.device(args.device)
    model = MLPDiffusion(encoded.shape[1], args.hid_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50)
    diffusion = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_t)
    alpha_bar = diffusion["Alpha_bar"].to(device)
    for epoch in range(1, epochs + 1):
        total = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            time = torch.randint(args.timesteps, (len(batch),), device=device)
            noise = torch.randn_like(batch)
            noisy = torch.sqrt(alpha_bar[time, None]) * batch + torch.sqrt(
                1 - alpha_bar[time, None]
            ) * noise
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(noisy, time), noise)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step(total)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch={epoch:05d} loss={total / max(1, batches):.6f}", flush=True)
    config = {
        "method": "diffputer_harpoon_repaint",
        "dataname": args.dataname,
        "hid_dim": args.hid_dim,
        "timesteps": args.timesteps,
        "beta_0": args.beta_0,
        "beta_t": args.beta_t,
        "encoded_dim": encoded.shape[1],
    }
    torch.save({"state_dict": model.state_dict(), "config": config}, output_dir / "model.pt")
    (output_dir / "adapter.json").write_text(json.dumps(config, indent=2) + "\n")


def train_great(args, table, output_dir: Path) -> None:
    sys.path.insert(0, str(Path(args.great_root).resolve()))
    from be_great import GReaT

    epochs = args.epochs if args.epochs is not None else 5
    batch_size = args.batch_size if args.batch_size is not None else 32
    model = GReaT(
        llm=args.llm,
        batch_size=batch_size,
        epochs=epochs,
        fp16=args.device.startswith("cuda"),
        float_precision=args.float_precision,
        dataloader_num_workers=4,
        report_to=[],
        save_steps=100000,
        experiment_dir=str(output_dir / "trainer"),
        save_strategy="no",
        logging_strategy="epoch",
        logging_first_step=False,
    )
    model.fit(table.train.loc[:, table.model_columns])
    model.save(str(output_dir / "model"))
    metadata = {
        "method": "great_native_imputation",
        "dataname": args.dataname,
        "llm": args.llm,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    (output_dir / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    table = load_baseline_table(
        dataname=args.dataname,
        train_data=args.train_data,
        test_data=args.test_data,
        info_file=args.info_file,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "diffputer":
        train_diffputer(args, table, output_dir)
    else:
        train_great(args, table, output_dir)
    print(f"Saved {args.method} model to {output_dir}")


if __name__ == "__main__":
    main()
