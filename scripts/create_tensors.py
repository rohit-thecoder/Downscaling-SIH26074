import xarray as xr
import numpy as np
import torch
from pathlib import Path


ERA5_FILE = "data/processed/era5_lowres_clean.nc"
ERA5_LAND_FILE = "data/processed/era5_land_highres_clean.nc"

OUTPUT_DIR = Path("data/processed/tensors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


VARIABLES = [
    "t2m",
    "u10",
    "v10"
]


print("=" * 70)
print("CREATING PYTORCH TENSORS")
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
# Extract variables
# --------------------------------------------------

print("\nExtracting variables...")

X_list = []
Y_list = []


for variable in VARIABLES:

    print(f"Processing {variable}")

    x = era5[variable].values
    y = era5_land[variable].values

    X_list.append(x)
    Y_list.append(y)


# --------------------------------------------------
# Stack channels
# --------------------------------------------------

# Current shapes:
#
# Each variable:
# [time, lat, lon]
#
# After stacking:
#
# [time, channels, lat, lon]

X = np.stack(
    X_list,
    axis=1
)

Y = np.stack(
    Y_list,
    axis=1
)


print("\nNumPy shapes:")

print("X:", X.shape)
print("Y:", Y.shape)


# --------------------------------------------------
# Check NaN
# --------------------------------------------------

print("\nChecking NaN...")

print("X NaN:", np.isnan(X).sum())
print("Y NaN:", np.isnan(Y).sum())


# --------------------------------------------------
# Convert to PyTorch
# --------------------------------------------------

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

Y_tensor = torch.tensor(
    Y,
    dtype=torch.float32
)


print("\nPyTorch shapes:")

print("X:", X_tensor.shape)
print("Y:", Y_tensor.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(
    X_tensor,
    OUTPUT_DIR / "X.pt"
)

torch.save(
    Y_tensor,
    OUTPUT_DIR / "Y.pt"
)


# --------------------------------------------------
# Metadata
# --------------------------------------------------

metadata = {
    "variables": VARIABLES,

    "input_shape": list(X_tensor.shape),

    "target_shape": list(Y_tensor.shape),

    "input_resolution": "0.25 degree",

    "target_resolution": "0.1 degree",

    "input_source": "ERA5",

    "target_source": "ERA5-Land",

    "num_samples": X_tensor.shape[0],

    "num_channels": X_tensor.shape[1],

    "input_height": X_tensor.shape[2],

    "input_width": X_tensor.shape[3],

    "target_height": Y_tensor.shape[2],

    "target_width": Y_tensor.shape[3]
}


import json

with open(
    OUTPUT_DIR / "metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print("\nFiles saved:")

print(OUTPUT_DIR / "X.pt")
print(OUTPUT_DIR / "Y.pt")
print(OUTPUT_DIR / "metadata.json")


era5.close()
era5_land.close()


print("\n" + "=" * 70)
print("TENSOR CREATION COMPLETE")
print("=" * 70)