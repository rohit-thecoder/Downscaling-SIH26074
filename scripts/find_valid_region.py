import xarray as xr
import numpy as np


FILE = "data/raw/era5_land/extracted/data_0.nc"

ds = xr.open_dataset(FILE, engine="netcdf4")

# --------------------------------------------------
# Common NaN mask
# --------------------------------------------------

mask = ds["t2m"].isnull().any(dim="valid_time")

print("=" * 70)
print("VALID REGION ANALYSIS")
print("=" * 70)

print("\nTotal latitude:", ds.sizes["latitude"])
print("Total longitude:", ds.sizes["longitude"])

print("Invalid grid cells:", int(mask.sum()))

print(
    "Valid grid cells:",
    int((~mask).sum())
)


# --------------------------------------------------
# Find valid latitude rows
# --------------------------------------------------

print("\n--- Latitude validity ---")

for i, lat in enumerate(ds.latitude.values):

    row = mask.isel(latitude=i).values

    valid_count = np.sum(~row)
    total_count = len(row)

    print(
        f"lat={lat:.1f} | "
        f"valid={valid_count}/{total_count}"
    )


# --------------------------------------------------
# Find valid longitude columns
# --------------------------------------------------

print("\n--- Longitude validity ---")

for j, lon in enumerate(ds.longitude.values):

    column = mask.isel(longitude=j).values

    valid_count = np.sum(~column)
    total_count = len(column)

    if valid_count > 0:

        print(
            f"lon={lon:.1f} | "
            f"valid={valid_count}/{total_count}"
        )


# --------------------------------------------------
# Determine completely valid latitude range
# --------------------------------------------------

fully_valid_latitudes = []

for i, lat in enumerate(ds.latitude.values):

    row = mask.isel(latitude=i).values

    if not np.any(row):
        fully_valid_latitudes.append(float(lat))


print("\n--- Completely valid latitude rows ---")

if fully_valid_latitudes:

    print(
        min(fully_valid_latitudes),
        "to",
        max(fully_valid_latitudes)
    )

else:

    print("No completely valid latitude row.")


# --------------------------------------------------
# Determine completely valid longitude range
# --------------------------------------------------

fully_valid_longitudes = []

for j, lon in enumerate(ds.longitude.values):

    column = mask.isel(longitude=j).values

    if not np.any(column):
        fully_valid_longitudes.append(float(lon))


print("\n--- Completely valid longitude columns ---")

if fully_valid_longitudes:

    print(
        min(fully_valid_longitudes),
        "to",
        max(fully_valid_longitudes)
    )

else:

    print("No completely valid longitude column.")


ds.close()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)