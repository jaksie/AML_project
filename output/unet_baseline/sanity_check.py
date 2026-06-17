#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_model
from preprocessing import load_dataset
from utils import load_config


def main():
    cfg = load_config()

    output_dir = Path(cfg["paths"]["output_dir"])
    checkpoint_path = output_dir / "model" / "last_model.pt"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device, flush=True)
    print("checkpoint:", checkpoint_path, flush=True)

    data = load_dataset(cfg["paths"]["dataset_npz"])

    model = build_model(
        input_shape=cfg["model"]["input_shape"],
        base_filters=cfg["model"]["base_filters"],
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    mean = data["mean"]
    std = data["std"]

    indices = [0, 100, 500, 1000, 3000]

    for idx in indices:
        x_norm = data["X_test"][idx : idx + 1].to(device)

        with torch.no_grad():
            pred_norm = model(x_norm)

        pred_norm = pred_norm.cpu().numpy()[0, 0]
        pred_raw = pred_norm * std + mean

        x_raw = data["X_raw_test"][idx, :, :, 0]
        y_raw = data["Y_raw_test"][idx, :, :, 0]

        baseline_err = x_raw - y_raw
        model_err = pred_raw - y_raw

        baseline_mae = np.mean(np.abs(baseline_err))
        model_mae = np.mean(np.abs(model_err))

        vmin = min(x_raw.min(), pred_raw.min(), y_raw.min())
        vmax = max(x_raw.max(), pred_raw.max(), y_raw.max())

        err_abs = max(
            abs(baseline_err).max(),
            abs(model_err).max(),
        )

        print(
            f"idx={idx:5d} "
            f"x_raw=[{x_raw.min():.2f}, {x_raw.max():.2f}] "
            f"y_raw=[{y_raw.min():.2f}, {y_raw.max():.2f}] "
            f"pred_raw=[{pred_raw.min():.2f}, {pred_raw.max():.2f}] "
            f"baseline_MAE={baseline_mae:.4f} K "
            f"model_MAE={model_mae:.4f} K",
            flush=True,
        )

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        im0 = axes[0].imshow(x_raw, vmin=vmin, vmax=vmax)
        axes[0].set_title("Input X_raw")

        axes[1].imshow(pred_raw, vmin=vmin, vmax=vmax)
        axes[1].set_title("U-Net prediction")

        axes[2].imshow(y_raw, vmin=vmin, vmax=vmax)
        axes[2].set_title("Target Y_raw")

        im3 = axes[3].imshow(baseline_err, vmin=-err_abs, vmax=err_abs, cmap="coolwarm")
        axes[3].set_title("Baseline error")

        axes[4].imshow(model_err, vmin=-err_abs, vmax=err_abs, cmap="coolwarm")
        axes[4].set_title("Model error")

        for ax in axes:
            ax.axis("off")

        fig.colorbar(im0, ax=axes[:3], shrink=0.75, label="T2 [K]")
        fig.colorbar(im3, ax=axes[3:], shrink=0.75, label="error [K]")

        fig.suptitle(
            f"Test patch idx={idx} | "
            f"baseline MAE={baseline_mae:.4f} K | "
            f"model MAE={model_mae:.4f} K"
        )

        out = figure_dir / f"quick_eval_idx_{idx:05d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("saved:", out, flush=True)


if __name__ == "__main__":
    main()