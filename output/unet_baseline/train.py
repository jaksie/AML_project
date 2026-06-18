#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import build_model
from preprocessing import load_dataset, make_loaders
from utils import load_config, set_seed

'''
Residual learning proved to be a step up in the performance (in comparison to the direct unet). MAE and RMSE were significantly lower, but Gradient MAE stood in place. I added provisional gradient loss tracking but it resulted in no significant improvement whatsoever (tried it with lambda ranging from .1 to .5). Going back to the standard Loss = MAE(Y, \hat Y) wouldnt impair the model but instead improve the readability of the training loop :/
'''

def gradient_loss(pred, target):

    # assumed tensor shape = [B, C, H, W]

    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    loss_dy = torch.mean(torch.abs(pred_dy - target_dy))
    loss_dx = torch.mean(torch.abs(pred_dx - target_dx))

    return 0.5 * (loss_dy + loss_dx)


def compute_loss(pred, target, pixel_loss_fn, gradient_loss_weight):
    pixel = pixel_loss_fn(pred, target) # standard MAE, pixel by pixel
    grad = gradient_loss(pred, target)
    total = pixel + gradient_loss_weight * grad

    return total, pixel, grad


def train_one_epoch(
    model,
    loader,
    pixel_loss_fn,
    optimizer,
    device,
    gradient_loss_weight,
):
    model.train()

    total_losses = []
    pixel_losses = []
    gradient_losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        total_loss, pixel_loss, grad_loss = compute_loss(
            pred=pred,
            target=y,
            pixel_loss_fn=pixel_loss_fn,
            gradient_loss_weight=gradient_loss_weight,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        pixel_losses.append(pixel_loss.item())
        gradient_losses.append(grad_loss.item())

    return {
        "total": float(np.mean(total_losses)),
        "pixel": float(np.mean(pixel_losses)),
        "gradient": float(np.mean(gradient_losses)),
    }


def validate(
    model,
    loader,
    pixel_loss_fn,
    device,
    gradient_loss_weight,
):
    model.eval()

    total_losses = []
    pixel_losses = []
    gradient_losses = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            total_loss, pixel_loss, grad_loss = compute_loss(
                pred=pred,
                target=y,
                pixel_loss_fn=pixel_loss_fn,
                gradient_loss_weight=gradient_loss_weight,
            )

            total_losses.append(total_loss.item())
            pixel_losses.append(pixel_loss.item())
            gradient_losses.append(grad_loss.item())

    return {
        "total": float(np.mean(total_losses)),
        "pixel": float(np.mean(pixel_losses)),
        "gradient": float(np.mean(gradient_losses)),
    }


def main(name="experiment"):
    cfg = load_config()
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["paths"]["output_dir"])
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir / "tensorboard" / name)

    data = load_dataset(cfg["paths"]["dataset_npz"])
    train_loader, val_loader, _ = make_loaders(
        data,
        batch_size=cfg["training"]["batch_size"],
    )

    gradient_loss_weight = float(cfg["training"].get("gradient_loss_weight", 0.0))
    patience = int(cfg["training"].get("early_stopping_patience", 0))
    min_delta = float(cfg["training"].get("early_stopping_min_delta", 0.0))

    print("experiment:", name, flush=True)
    print("device:", device, flush=True)
    print("epochs:", cfg["training"]["epochs"], flush=True)
    print("train batches:", len(train_loader), flush=True)
    print("val batches:", len(val_loader), flush=True)
    print("gradient_loss_weight:", gradient_loss_weight, flush=True)
    print("early_stopping_patience:", patience, flush=True) # added early stopping when Taurus started malfunctioning and had to train models locally. In the end, it almost always ran for the entirety of the loop
    print("early_stopping_min_delta:", min_delta, flush=True)

    model = build_model(
        input_shape=cfg["model"]["input_shape"],
        base_filters=cfg["model"]["base_filters"],
    ).to(device)

    pixel_loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_checkpoint_path = model_dir / "best_model.pt"
    last_checkpoint_path = model_dir / "last_model.pt"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            pixel_loss_fn=pixel_loss_fn,
            optimizer=optimizer,
            device=device,
            gradient_loss_weight=gradient_loss_weight,
        )

        val_losses = validate(
            model=model,
            loader=val_loader,
            pixel_loss_fn=pixel_loss_fn,
            device=device,
            gradient_loss_weight=gradient_loss_weight,
        )

        print(
            f"epoch {epoch}/{cfg['training']['epochs']} "
            f"train_total={train_losses['total']:.6f} "
            f"train_pixel={train_losses['pixel']:.6f} "
            f"train_grad={train_losses['gradient']:.6f} "
            f"val_total={val_losses['total']:.6f} "
            f"val_pixel={val_losses['pixel']:.6f} "
            f"val_grad={val_losses['gradient']:.6f}",
            flush=True,
        )

        writer.add_scalar("loss/train_total", train_losses["total"], epoch)
        writer.add_scalar("loss/train_pixel", train_losses["pixel"], epoch)
        writer.add_scalar("loss/train_gradient", train_losses["gradient"], epoch)

        writer.add_scalar("loss/val_total", val_losses["total"], epoch)
        writer.add_scalar("loss/val_pixel", val_losses["pixel"], epoch)
        writer.add_scalar("loss/val_gradient", val_losses["gradient"], epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        improved = val_losses["total"] < best_val_loss - min_delta

        if improved:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_checkpoint_path)

            print(
                f"saved best_model.pt "
                f"epoch={epoch} "
                f"val_total={val_losses['total']:.6f} "
                f"val_pixel={val_losses['pixel']:.6f} "
                f"val_grad={val_losses['gradient']:.6f}",
                flush=True,
            )
        else:
            epochs_without_improvement += 1

            print(
                f"no improvement: "
                f"{epochs_without_improvement}/{patience}",
                flush=True,
            )

        writer.add_scalar("early_stopping/best_val_total", best_val_loss, epoch)
        writer.add_scalar(
            "early_stopping/epochs_without_improvement",
            epochs_without_improvement,
            epoch,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}; "
                f"best_val_total={best_val_loss:.6f}",
                flush=True,
            )
            break

    torch.save(model.state_dict(), last_checkpoint_path)

    print("saved:", last_checkpoint_path, flush=True)
    print("saved best:", best_checkpoint_path, flush=True)
    print("best_epoch:", best_epoch, flush=True)
    print("best_val_loss:", best_val_loss, flush=True)

    writer.close()

    return best_checkpoint_path


if __name__ == "__main__":
    main()