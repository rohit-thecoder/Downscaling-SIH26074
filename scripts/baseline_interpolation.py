import xarray as xr
import numpy as np
from pathlib import Path


ERA5_FILE = "data/processed/era5_lowres_clean.nc"
ERA5_LAND_FILE = "data/processed/era5_land_highres_clean.nc"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = ["t2m", "u10", "v10"]


print("=" * 70)
print("BILINEAR INTERPOLATION BASELINE")
print("=" * 70)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

era5 = xr.open_dataset(
    ERA5_FILE,
    engine="netcdf4"
)

era5_land = xr.open_dataset(
    ERA5_LAND_FILE,
    engine="netcdf4"
)


# --------------------------------------------------
# Interpolate ERA5 to ERA5-Land grid
# --------------------------------------------------

print("\nInterpolating ERA5 0.25° → ERA5-Land 0.1°...")

predictions = {}

for variable in VARIABLES:

    print(f"Processing: {variable}")

    source = era5[variable]

    target_lat = era5_land.latitude
    target_lon = era5_land.longitude

    interpolated = source.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )

    predictions[variable] = interpolated


# --------------------------------------------------
# Create prediction dataset
# --------------------------------------------------

baseline = xr.Dataset(predictions)

print("\nBaseline dataset:")
print(baseline)


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE METRICS")
print("=" * 70)


metrics = {}


for variable in VARIABLES:

    prediction = baseline[variable]
    actual = era5_land[variable]

    # Only compare where both are valid
    valid = (
        prediction.notnull()
        & actual.notnull()
    )

    y_pred = prediction.values[valid.values]
    y_true = actual.values[valid.values]

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    metrics[variable] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\n{variable}")

    print(f"MAE :  {mae:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"R²  :  {r2:.6f}")


# --------------------------------------------------
# Save baseline
# --------------------------------------------------

output_file = OUTPUT_DIR / "bilinear_baseline.nc"

baseline.to_netcdf(output_file)

print("\nSaved:")
print(output_file)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

import json

metrics_file = OUTPUT_DIR / "baseline_metrics.json"

with open(metrics_file, "w") as f:
    json.dump(
        {
            variable: {
                metric: float(value)
                for metric, value in values.items()
            }
            for variable, values in metrics.items()
        },
        f,
        indent=4
    )

print("Saved:")
print(metrics_file)


era5.close()
era5_land.close()

print("\n" + "=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)