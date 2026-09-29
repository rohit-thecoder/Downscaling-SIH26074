import xarray as xr
import numpy as np

FILE = "data/raw/era5_land/extracted/data_0.nc"

ds = xr.open_dataset(FILE, engine="netcdf4")

print("=" * 70)
print("ERA5-LAND NaN INVESTIGATION")
print("=" * 70)

for variable in ["t2m", "tp", "u10", "v10"]:

    data = ds[variable]

    print(f"\nVariable: {variable}")

    # NaN count for each latitude
    nan_by_lat = data.isnull().sum(dim=["valid_time", "longitude"])

    # NaN count for each longitude
    nan_by_lon = data.isnull().sum(dim=["valid_time", "latitude"])

    print("\nNaN by latitude:")

    for lat, count in zip(
        ds.latitude.values,
        nan_by_lat.values
    ):
        if count > 0:
            print(f"Latitude {lat:.1f}: {int(count)} NaNs")

    print("\nNaN by longitude:")

    for lon, count in zip(
        ds.longitude.values,
        nan_by_lon.values
    ):
        if count > 0:
            print(f"Longitude {lon:.1f}: {int(count)} NaNs")


# --------------------------------------------------
# Check whether all variables have the same NaN mask
# --------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING COMMON NaN MASK")
print("=" * 70)

mask_t2m = ds["t2m"].isnull()
mask_tp = ds["tp"].isnull()
mask_u10 = ds["u10"].isnull()
mask_v10 = ds["v10"].isnull()

print(
    "t2m == tp:",
    bool((mask_t2m == mask_tp).all())
)

print(
    "t2m == u10:",
    bool((mask_t2m == mask_u10).all())
)

print(
    "t2m == v10:",
    bool((mask_t2m == mask_v10).all())
)


# --------------------------------------------------
# Check if NaNs are constant through time
# --------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING TIME CONSISTENCY")
print("=" * 70)

spatial_nan_mask = ds["t2m"].isnull().any(dim="valid_time")

nan_cells = int(spatial_nan_mask.sum())

print("Spatial grid cells containing NaN:", nan_cells)

# Check whether same spatial mask exists at every timestep
first_mask = ds["t2m"].isel(valid_time=0).isnull()

same_mask_all_times = True

for i in range(1, ds.sizes["valid_time"]):
    current_mask = ds["t2m"].isel(valid_time=i).isnull()

    if not bool((first_mask == current_mask).all()):
        same_mask_all_times = False
        break

print("Same NaN spatial mask at every timestep:",
      same_mask_all_times)


ds.close()

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)