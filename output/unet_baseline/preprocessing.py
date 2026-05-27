import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils import nhwc_to_nchw
# (N, 64, 64, 1) -> (N, 1, 64, 64)


def load_dataset(path):
    d = np.load(path, allow_pickle=True)

    # X (normalized)
    # Y (normalized)
    # train_mask (boolean)
    # val_mask (boolean)
    # test_mask (boolean)

    # X_raw (original, in Kelvin)
    # Y_raw (original, in Kelvin)
    # mean
    # std

    # baseline_mae = mean(abs(X_raw_test - Y_raw_test))
    # unet_mae     = mean(abs(pred_raw - Y_raw_test))

    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)

    train_mask = d["train_mask"].astype(bool)
    val_mask = d["val_mask"].astype(bool)
    test_mask = d["test_mask"].astype(bool)

    # train: 40824 patches
    # val:    7776 patches
    # test:  11664 patches
    # total: 60264 patches

    data = {
        "X_train": nhwc_to_nchw(X[train_mask]),
        "Y_train": nhwc_to_nchw(Y[train_mask]),
        "X_val": nhwc_to_nchw(X[val_mask]),
        "Y_val": nhwc_to_nchw(Y[val_mask]),
        "X_test": nhwc_to_nchw(X[test_mask]),
        "Y_test": nhwc_to_nchw(Y[test_mask]),

        "X_raw_test": X[test_mask],
        "Y_raw_test": Y[test_mask],

        "mean": float(d["mean"]),
        "std": float(d["std"]),
    }

    return data

def make_loaders(data, batch_size):
    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["Y_train"]), # TensorDataset(X_train, Y_train) creates a dataset of (X_train[i], Y_train[i]) pairs
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(data["X_val"], data["Y_val"]),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(data["X_test"], data["Y_test"]),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader