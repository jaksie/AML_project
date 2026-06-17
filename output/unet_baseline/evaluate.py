#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json

import numpy as np
import torch

from model import build_model
from preprocessing import load_dataset
from utils import load_config


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def bias(pred, target):
    return float(np.mean(pred - target))


def gradient_mae(pred, target):
    pred_dy = np.diff(pred, axis=1)
    target_dy = np.diff(target, axis=1)

    pred_dx = np.diff(pred, axis=2)
    target_dx = np.diff(target, axis=2)

    dy_mae = np.mean(np.abs(pred_dy - target_dy))
    dx_mae = np.mean(np.abs(pred_dx - target_dx))

    return float(0.5 * (dy_mae + dx_mae))


def compute_metrics(pred, target):
    return {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "bias": bias(pred, target),
        "gradient_mae": gradient_mae(pred, target),
    }


def predict(model, x_test, mean, std, device, batch_size):
    model.eval()
    preds = []

    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            end = start + batch_size

            x = x_test[start:end].to(device)
            pred_norm = model(x)

            pred_raw = pred_norm.cpu().numpy()[:, 0, :, :] * std + mean
            preds.append(pred_raw)

    return np.concatenate(preds, axis=0)


def load_test_leads(dataset_path):
    d = np.load(dataset_path, allow_pickle=True)
    test_mask = d["test_mask"].astype(bool)

    if "patch_lead_hours" in d.files:
        return d["patch_lead_hours"][test_mask].astype(int)

    if "lead_hours" in d.files:
        lead_hours = d["lead_hours"].astype(int)

        if len(lead_hours) == len(test_mask):
            return lead_hours[test_mask]

    return None


def write_metrics_csv(path, rows):
    fieldnames = [
        "experiment",
        "model_type",
        "lead",
        "mae",
        "rmse",
        "bias",
        "gradient_mae",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main(name=None, checkpoint=None):
    if name is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", default="experiment")
        parser.add_argument("--checkpoint", default=None)
        args = parser.parse_args()

        name = args.name
        checkpoint = args.checkpoint

    cfg = load_config()

    output_dir = Path(cfg["paths"]["output_dir"])
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        Path(checkpoint)
        if checkpoint is not None
        else output_dir / "model" / "last_model.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("experiment:", name, flush=True)
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

    mean = data["mean"]
    std = data["std"]

    x_test = data["X_test"]
    x_raw = data["X_raw_test"][:, :, :, 0]
    y_raw = data["Y_raw_test"][:, :, :, 0]

    pred_raw = predict(
        model=model,
        x_test=x_test,
        mean=mean,
        std=std,
        device=device,
        batch_size=cfg["training"]["batch_size"],
    )

    baseline_metrics = compute_metrics(x_raw, y_raw)
    model_metrics = compute_metrics(pred_raw, y_raw)

    summary = {
        "experiment": name,
        "checkpoint": str(checkpoint_path),
        "n_test": int(len(y_raw)),
        "baseline": baseline_metrics,
        "model": model_metrics,
        "improvement": {
            "mae": baseline_metrics["mae"] - model_metrics["mae"],
            "rmse": baseline_metrics["rmse"] - model_metrics["rmse"],
            "gradient_mae": (
                baseline_metrics["gradient_mae"]
                - model_metrics["gradient_mae"]
            ),
        },
    }

    rows = []

    rows.append({
        "experiment": name,
        "model_type": "baseline",
        "lead": "all",
        **baseline_metrics,
    })

    rows.append({
        "experiment": name,
        "model_type": "unet",
        "lead": "all",
        **model_metrics,
    })

    leads = load_test_leads(cfg["paths"]["dataset_npz"])

    if leads is not None:
        for lead in sorted(np.unique(leads)):
            mask = leads == lead

            baseline_lead_metrics = compute_metrics(x_raw[mask], y_raw[mask])
            model_lead_metrics = compute_metrics(pred_raw[mask], y_raw[mask])

            rows.append({
                "experiment": name,
                "model_type": "baseline",
                "lead": int(lead),
                **baseline_lead_metrics,
            })

            rows.append({
                "experiment": name,
                "model_type": "unet",
                "lead": int(lead),
                **model_lead_metrics,
            })

    json_path = eval_dir / f"{name}_metrics.json"
    csv_path = eval_dir / f"{name}_metrics.csv"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_metrics_csv(csv_path, rows)

    print()
    print("Overall metrics:")
    print(
        f"baseline "
        f"MAE={baseline_metrics['mae']:.4f} K "
        f"RMSE={baseline_metrics['rmse']:.4f} K "
        f"bias={baseline_metrics['bias']:.4f} K "
        f"grad_MAE={baseline_metrics['gradient_mae']:.4f}"
    )
    print(
        f"unet     "
        f"MAE={model_metrics['mae']:.4f} K "
        f"RMSE={model_metrics['rmse']:.4f} K "
        f"bias={model_metrics['bias']:.4f} K "
        f"grad_MAE={model_metrics['gradient_mae']:.4f}"
    )

    print()
    print("Improvement over baseline:")
    print(f"MAE improvement:       {summary['improvement']['mae']:.4f} K")
    print(f"RMSE improvement:      {summary['improvement']['rmse']:.4f} K")
    print(f"gradient improvement:  {summary['improvement']['gradient_mae']:.4f}")

    print()
    print("saved:", json_path)
    print("saved:", csv_path)


if __name__ == "__main__":
    main()