import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch

from src.models.downscaling_cnn import DownscalingCNN


# ============================================================
# PATHS
# ============================================================

X_PATH = "data/processed/tensors/X_test.pt"
Y_PATH = "data/processed/tensors/Y_test.pt"

STATS_PATH = "data/processed/tensors/normalization.json"

MODEL_PATH = "models/downscaling_cnn.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("CNN TEST EVALUATION")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = torch.load(
    X_PATH,
    weights_only=False
)

Y_test = torch.load(
    Y_PATH,
    weights_only=False
)


print("\nX_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

with open(STATS_PATH, "r") as f:
    stats = json.load(f)


Y_mean = torch.tensor(
    stats["Y"]["mean"],
    dtype=torch.float32
).view(1, 3, 1, 1)

Y_std = torch.tensor(
    stats["Y"]["std"],
    dtype=torch.float32
).view(1, 3, 1, 1)


# ============================================================
# LOAD MODEL
# ============================================================

model = DownscalingCNN(
    in_channels=3,
    out_channels=3
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

X_test = X_test.to(device)

with torch.no_grad():

    prediction_norm = model(
        X_test,
        target_size=(61, 51)
    )


# ============================================================
# DENORMALIZE
# ============================================================

prediction = (
    prediction_norm * Y_std.to(device)
    + Y_mean.to(device)
)

target = (
    Y_test.to(device) * Y_std.to(device)
    + Y_mean.to(device)
)


# ============================================================
# METRICS
# ============================================================

channel_names = [
    "t2m",
    "u10",
    "v10"
]


print("\n" + "=" * 70)
print("TEST METRICS")
print("=" * 70)


results = {}


for channel, name in enumerate(channel_names):

    pred = prediction[:, channel]
    true = target[:, channel]

    # Valid ERA5-Land cells
    mask = torch.isfinite(true)

    pred_valid = pred[mask]
    true_valid = true[mask]


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = torch.mean(
        torch.abs(
            pred_valid - true_valid
        )
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = torch.sqrt(
        torch.mean(
            (pred_valid - true_valid) ** 2
        )
    )


    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    ss_res = torch.sum(
        (true_valid - pred_valid) ** 2
    )

    ss_tot = torch.sum(
        (true_valid - true_valid.mean()) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)


    print(f"\n{name}")

    if name == "t2m":
        unit = "°C"
    else:
        unit = "m/s"

    print(f"MAE  = {mae.item():.6f} {unit}")
    print(f"RMSE = {rmse.item():.6f} {unit}")
    print(f"R²   = {r2.item():.6f}")


    results[name] = {
        "MAE": mae.item(),
        "RMSE": rmse.item(),
        "R2": r2.item()
    }


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path(
    "data/processed/cnn_test_metrics.json"
)

with open(output_path, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nMetrics saved to:")
print(output_path)