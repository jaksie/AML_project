#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import build_model
from preprocessing import load_dataset, make_loaders
from utils import load_config, set_seed


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def validate(model, loader, loss_fn, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            losses.append(loss.item())

    return float(np.mean(losses))


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

    print("experiment:", name, flush=True)
    print("device:", device, flush=True)
    print("epochs:", cfg["training"]["epochs"], flush=True)
    print("train batches:", len(train_loader), flush=True)
    print("val batches:", len(val_loader), flush=True)

    model = build_model(
        input_shape=cfg["model"]["input_shape"],
        base_filters=cfg["model"]["base_filters"],
    ).to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
    )

    best_val_loss = float("inf")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch}/{cfg['training']['epochs']} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}",
            flush=True,
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), model_dir / "best_model.pt")

            print(
                f"saved best_model.pt "
                f"epoch={epoch} "
                f"val_loss={val_loss:.6f}",
                flush=True,
            )

    torch.save(model.state_dict(), model_dir / "last_model.pt")
    print("saved:", model_dir / "last_model.pt", flush=True)
    print("best_val_loss:", best_val_loss, flush=True)

    writer.close()


if __name__ == "__main__":
    main()