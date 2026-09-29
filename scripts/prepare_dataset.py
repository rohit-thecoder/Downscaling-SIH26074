import xarray as xr
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ERA5_INSTANT = (
    "data/raw/era5/extracted/"
    "data_stream-oper_stepType-instant.nc"
)

ERA5_LAND = (
    "data/raw/era5_land/extracted/"
    "data_0.nc"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load
# ============================================================

print("=" * 70)
print("PREPARING ML DATASET")
print("=" * 70)

era5 = xr.open_dataset(
    ERA5_INSTANT,
    engine="netcdf4"
)

era5_land = xr.open_dataset(
    ERA5_LAND,
    engine="netcdf4"
)


# ============================================================
# Select variables
# ============================================================

variables = [
    "t2m",
    "u10",
    "v10",
]

era5 = era5[variables]
era5_land = era5_land[variables]


# ============================================================
# Check time alignment
# ============================================================

print("\nChecking time alignment...")

if not np.array_equal(
    era5.valid_time.values,
    era5_land.valid_time.values
):
    raise ValueError(
        "ERA5 and ERA5-Land timestamps do not match."
    )

print("Time alignment: OK")


# ============================================================
# Create common spatial validity mask
# ============================================================

print("\nCreating valid spatial mask...")

# A grid cell is valid only if all selected
# ERA5-Land variables have valid values
land_valid = (
    era5_land["t2m"].notnull()
    & era5_land["u10"].notnull()
    & era5_land["v10"].notnull()
)

# Same spatial mask across all timestamps
valid_mask = land_valid.all(dim="valid_time")

print(
    "Valid spatial cells:",
    int(valid_mask.sum())
)

print(
    "Invalid spatial cells:",
    int((~valid_mask).sum())
)


# ============================================================
# Keep only valid ERA5-Land grid cells
# ============================================================

# Mask ERA5-Land
era5_land_clean = era5_land.where(valid_mask)


# ============================================================
# Convert units for physical interpretation
# ============================================================

print("\nConverting units...")

# Temperature: Kelvin -> Celsius
era5_clean = era5.copy()
era5_land_clean = era5_land_clean.copy()

era5_clean["t2m"] = era5_clean["t2m"] - 273.15
era5_land_clean["t2m"] = era5_land_clean["t2m"] - 273.15

era5_clean["t2m"].attrs["units"] = "degC"
era5_land_clean["t2m"].attrs["units"] = "degC"


# ============================================================
# Save cleaned datasets
# ============================================================

era5_output = OUTPUT_DIR / "era5_lowres_clean.nc"
land_output = OUTPUT_DIR / "era5_land_highres_clean.nc"
mask_output = OUTPUT_DIR / "valid_mask.nc"


era5_clean.to_netcdf(era5_output)
era5_land_clean.to_netcdf(land_output)

valid_mask.to_dataset(name="valid_mask").to_netcdf(
    mask_output
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print("\nERA5:")
print(era5_clean)

print("\nERA5-Land:")
print(era5_land_clean)

print("\nFiles created:")

print(era5_output)
print(land_output)
print(mask_output)


# ============================================================
# Close
# ============================================================

era5.close()
era5_land.close()
era5_clean.close()
era5_land_clean.close()