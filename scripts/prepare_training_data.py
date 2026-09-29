import json
from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

X_PATH = Path("data/processed/tensors/X.pt")
Y_PATH = Path("data/processed/tensors/Y.pt")
STATS_PATH = Path("data/processed/tensors/normalization.json")

OUTPUT_DIR = Path("data/processed/tensors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

X = torch.load(X_PATH, weights_only=False)
Y = torch.load(Y_PATH, weights_only=False)

with open(STATS_PATH, "r") as f:
    stats = json.load(f)


print("=" * 70)
print("PREPARING TRAIN / VALIDATION / TEST DATA")
print("=" * 70)

print("\nOriginal X:", X.shape)
print("Original Y:", Y.shape)


# ============================================================
# NORMALIZATION PARAMETERS
# ============================================================

X_mean = torch.tensor(
    stats["X"]["mean"],
    dtype=torch.float32
).view(1, 3, 1, 1)

X_std = torch.tensor(
    stats["X"]["std"],
    dtype=torch.float32
).view(1, 3, 1, 1)

Y_mean = torch.tensor(
    stats["Y"]["mean"],
    dtype=torch.float32
).view(1, 3, 1, 1)

Y_std = torch.tensor(
    stats["Y"]["std"],
    dtype=torch.float32
).view(1, 3, 1, 1)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n_samples = X.shape[0]

train_end = 12
val_end = 16

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]


print("\nChronological split:")

print(f"Train:      {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test:       {X_test.shape[0]} samples")


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_X(x):
    return (x - X_mean) / X_std


def normalize_Y(y):
    return (y - Y_mean) / Y_std


# ============================================================
# NORMALIZE
# ============================================================

X_train_norm = normalize_X(X_train)
X_val_norm = normalize_X(X_val)
X_test_norm = normalize_X(X_test)

Y_train_norm = normalize_Y(Y_train)
Y_val_norm = normalize_Y(Y_val)
Y_test_norm = normalize_Y(Y_test)


# ============================================================
# SAVE
# ============================================================

torch.save(
    X_train_norm,
    OUTPUT_DIR / "X_train.pt"
)

torch.save(
    Y_train_norm,
    OUTPUT_DIR / "Y_train.pt"
)

torch.save(
    X_val_norm,
    OUTPUT_DIR / "X_val.pt"
)

torch.save(
    Y_val_norm,
    OUTPUT_DIR / "Y_val.pt"
)

torch.save(
    X_test_norm,
    OUTPUT_DIR / "X_test.pt"
)

torch.save(
    Y_test_norm,
    OUTPUT_DIR / "Y_test.pt"
)


# ============================================================
# VERIFY NaNs
# ============================================================

print("\nNaN verification:")

print("X_train NaN:", torch.isnan(X_train_norm).sum().item())
print("X_val NaN:  ", torch.isnan(X_val_norm).sum().item())
print("X_test NaN: ", torch.isnan(X_test_norm).sum().item())

print("Y_train NaN:", torch.isnan(Y_train_norm).sum().item())
print("Y_val NaN:  ", torch.isnan(Y_val_norm).sum().item())
print("Y_test NaN: ", torch.isnan(Y_test_norm).sum().item())


# ============================================================
# VERIFY NORMALIZED TRAINING DATA
# ============================================================

print("\nTraining normalized statistics:")

for channel, name in enumerate(["t2m", "u10", "v10"]):

    values = X_train_norm[:, channel]

    valid_values = values[torch.isfinite(values)]

    print(
        f"{name}: "
        f"mean={valid_values.mean().item():.4f}, "
        f"std={valid_values.std().item():.4f}"
    )


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

split_info = {
    "total_samples": n_samples,
    "train_samples": train_end,
    "validation_samples": val_end - train_end,
    "test_samples": n_samples - val_end,
    "train_range": "0:12",
    "validation_range": "12:16",
    "test_range": "16:20"
}

with open(
    OUTPUT_DIR / "split_info.json",
    "w"
) as f:
    json.dump(split_info, f, indent=4)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)

print("\nCreated files:")

print("X_train.pt")
print("Y_train.pt")
print("X_val.pt")
print("Y_val.pt")
print("X_test.pt")
print("Y_test.pt")
print("split_info.json")

print("\nLocation:")
print(OUTPUT_DIR)