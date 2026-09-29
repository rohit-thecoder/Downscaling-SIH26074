import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import xarray as xr
import numpy as np

from src.models.downscaling_cnn import DownscalingCNN


# ============================================================
# PATHS
# ============================================================

X_TEST = "data/processed/tensors/X_test.pt"
Y_TEST = "data/processed/tensors/Y_test.pt"

STATS = "data/processed/tensors/normalization.json"
MODEL = "models/downscaling_cnn.pth"

ERA5 = "data/processed/era5_lowres_clean.nc"
ERA5_LAND = "data/processed/era5_land_highres_clean.nc"


# ============================================================
# LOAD DATA
# ============================================================

X_test = torch.load(X_TEST, weights_only=False)
Y_test = torch.load(Y_TEST, weights_only=False)

with open(STATS) as f:
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
# LOAD CNN
# ============================================================

device = torch.device("cpu")

model = DownscalingCNN(
    in_channels=3,
    out_channels=3
)

model.load_state_dict(
    torch.load(
        MODEL,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ============================================================
# CNN PREDICTION
# ============================================================

with torch.no_grad():

    prediction_norm = model(
        X_test,
        target_size=(61, 51)
    )

prediction = (
    prediction_norm * Y_std
    + Y_mean
)

target = (
    Y_test * Y_std
    + Y_mean
)


# ============================================================
# LOAD ORIGINAL GRIDS
# ============================================================

low = xr.open_dataset(ERA5)
high = xr.open_dataset(ERA5_LAND)

variables = ["t2m", "u10", "v10"]

# Use same 4 timestamps as test set
test_times = high.valid_time.values[-4:]


# ============================================================
# BILINEAR BASELINE
# ============================================================

baseline_predictions = []

for time in test_times:

    low_time = low.sel(valid_time=time)

    interpolated = low_time[
        variables
    ].interp(
        latitude=high.latitude,
        longitude=high.longitude,
        method="linear"
    )

    baseline_predictions.append(
        np.stack([
            interpolated[var].values
            for var in variables
        ])
    )


baseline = torch.tensor(
    np.stack(baseline_predictions),
    dtype=torch.float32
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(prediction, target):

    results = {}

    for channel, name in enumerate(variables):

        pred = prediction[:, channel]
        true = target[:, channel]

        mask = torch.isfinite(true)

        pred_valid = pred[mask]
        true_valid = true[mask]

        mae = torch.mean(
            torch.abs(pred_valid - true_valid)
        )

        rmse = torch.sqrt(
            torch.mean(
                (pred_valid - true_valid) ** 2
            )
        )

        ss_res = torch.sum(
            (true_valid - pred_valid) ** 2
        )

        ss_tot = torch.sum(
            (true_valid - true_valid.mean()) ** 2
        )

        r2 = 1 - ss_res / ss_tot

        results[name] = {
            "MAE": mae.item(),
            "RMSE": rmse.item(),
            "R2": r2.item()
        }

    return results


# ============================================================
# CALCULATE
# ============================================================

cnn_results = calculate_metrics(
    prediction,
    target
)

baseline_results = calculate_metrics(
    baseline,
    target
)


# ============================================================
# PRINT
# ============================================================

print("=" * 75)
print("FAIR CNN vs BILINEAR BASELINE")
print("=" * 75)

for variable in variables:

    print(f"\n{variable}")

    print(
        f"Baseline | "
        f"MAE={baseline_results[variable]['MAE']:.6f} | "
        f"RMSE={baseline_results[variable]['RMSE']:.6f} | "
        f"R²={baseline_results[variable]['R2']:.6f}"
    )

    print(
        f"CNN      | "
        f"MAE={cnn_results[variable]['MAE']:.6f} | "
        f"RMSE={cnn_results[variable]['RMSE']:.6f} | "
        f"R²={cnn_results[variable]['R2']:.6f}"
    )


# ============================================================
# SAVE
# ============================================================

comparison = {
    "baseline": baseline_results,
    "cnn": cnn_results,
    "test_samples": 4
}

output = Path(
    "data/processed/model_comparison.json"
)

with open(output, "w") as f:
    json.dump(comparison, f, indent=4)

print("\n" + "=" * 75)
print("COMPARISON SAVED")
print("=" * 75)

print(output)