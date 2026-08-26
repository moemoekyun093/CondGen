"""Train HARPOON's released OHE DDPM on a prepared TabDiff dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from harpoon_runtime import load_harpoon


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", choices=("shoppers",), default="shoppers")
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument("--hid-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-0", type=float, default=0.0001)
    parser.add_argument("--beta-t", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.hid_dim, args.batch_size, args.epochs, args.timesteps) <= 0:
        raise ValueError("model size, batch size, epochs, and timesteps must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    harpoon_root = (PROJECT_ROOT / args.harpoon_root).resolve()
    previous_cwd = Path.cwd()
    os.chdir(harpoon_root)
    try:
        Preprocessor, MLPDiffusion, calc_diffusion_hyperparams = load_harpoon(
            harpoon_root
        )
        prepper = Preprocessor(args.dataname)
        train = prepper.encodeDf("OHE", prepper.df_train)
        numerical_count = prepper.numerical_indices_np_end
        mean = np.concatenate(
            [np.mean(train[:, :numerical_count], axis=0), np.zeros(train.shape[1] - numerical_count)]
        )
        std = np.concatenate(
            [np.std(train[:, :numerical_count], axis=0), np.ones(train.shape[1] - numerical_count)]
        )
        std[std == 0] = 1.0
        train_tensor = torch.tensor((train - mean) / std, dtype=torch.float32)

        device = torch.device(args.device)
        model = MLPDiffusion(train.shape[1], args.hid_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=50
        )
        diffusion = calc_diffusion_hyperparams(
            args.timesteps, args.beta_0, args.beta_t
        )
        alpha_bar = diffusion["Alpha_bar"].to(device)
        loader = DataLoader(
            train_tensor,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )

        progress = tqdm(range(args.epochs), desc="Training HARPOON OHE DDPM")
        for _ in progress:
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                time = torch.randint(args.timesteps, (len(batch),), device=device)
                noise = torch.randn_like(batch)
                noisy = (
                    torch.sqrt(alpha_bar[time, None]) * batch
                    + torch.sqrt(1 - alpha_bar[time, None]) * noise
                )
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(noisy, time), noise)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step(total_loss)
            progress.set_postfix(loss=total_loss)

        checkpoint = Path("saved_models") / args.dataname / "diffputer_selfmade.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved {checkpoint}")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
