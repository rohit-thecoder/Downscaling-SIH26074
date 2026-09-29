import json
from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

X_PATH = Path("data/processed/tensors/X.pt")
Y_PATH = Path("data/processed/tensors/Y.pt")

OUTPUT_DIR = Path("data/processed/tensors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATS_PATH = OUTPUT_DIR / "normalization.json"


# ============================================================
# LOAD DATA
# ============================================================

X = torch.load(X_PATH, weights_only=False)
Y = torch.load(Y_PATH, weights_only=False)

print("=" * 70)
print("NORMALIZATION")
print("=" * 70)

print("\nX shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# CHANNEL NAMES
# ============================================================

channels = ["t2m", "u10", "v10"]


# ============================================================
# CHRONOLOGICAL TRAINING DATA
# ============================================================

n_samples = X.shape[0]

train_end = int(n_samples * 0.8)

X_train = X[:train_end]
Y_train = Y[:train_end]

print("\nTraining samples:", train_end)
print("Validation/Test samples:", n_samples - train_end)


# ============================================================
# CALCULATE MEAN / STD
# ============================================================

X_mean = []
X_std = []

Y_mean = []
Y_std = []


for channel in range(X.shape[1]):

    # ========================================================
    # X STATISTICS
    # ========================================================

    x_values = X_train[:, channel]

    # Remove invalid values
    x_valid = x_values[torch.isfinite(x_values)]

    x_mean = x_valid.mean()
    x_std = x_valid.std()


    # ========================================================
    # Y STATISTICS
    # ========================================================

    y_values = Y_train[:, channel]

    # Remove NaN / invalid cells
    y_valid = y_values[torch.isfinite(y_values)]

    y_mean = y_valid.mean()
    y_std = y_valid.std()


    # ========================================================
    # STORE
    # ========================================================

    X_mean.append(x_mean.item())
    X_std.append(x_std.item())

    Y_mean.append(y_mean.item())
    Y_std.append(y_std.item())


    # ========================================================
    # PRINT
    # ========================================================

    print(f"\n{channels[channel]}")

    print(f"X mean = {x_mean.item():.6f}")
    print(f"X std  = {x_std.item():.6f}")

    print(f"Y mean = {y_mean.item():.6f}")
    print(f"Y std  = {y_std.item():.6f}")


# ============================================================
# SAVE NORMALIZATION STATISTICS
# ============================================================

stats = {
    "channels": channels,
    "train_samples": train_end,

    "X": {
        "mean": X_mean,
        "std": X_std
    },

    "Y": {
        "mean": Y_mean,
        "std": Y_std
    }
}


with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=4)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZATION STATISTICS SAVED")
print("=" * 70)

print(STATS_PATH)