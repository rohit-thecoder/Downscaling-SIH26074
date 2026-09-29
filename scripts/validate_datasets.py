import xarray as xr
import numpy as np


ERA5_INSTANT = "data/raw/era5/extracted/data_stream-oper_stepType-instant.nc"
ERA5_ACCUM = "data/raw/era5/extracted/data_stream-oper_stepType-accum.nc"
ERA5_LAND = "data/raw/era5_land/extracted/data_0.nc"


print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

era5_instant = xr.open_dataset(ERA5_INSTANT, engine="netcdf4")
era5_accum = xr.open_dataset(ERA5_ACCUM, engine="netcdf4")
era5_land = xr.open_dataset(ERA5_LAND, engine="netcdf4")


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n[1] Dataset shapes")

print("ERA5 temperature/wind:")
print(era5_instant.sizes)

print("\nERA5 precipitation:")
print(era5_accum.sizes)

print("\nERA5-Land:")
print(era5_land.sizes)


# --------------------------------------------------
# Check time
# --------------------------------------------------

era5_time = era5_instant.valid_time.values
era5_land_time = era5_land.valid_time.values

print("\n[2] Time alignment")

print("ERA5 first:", era5_time[0])
print("ERA5 last :", era5_time[-1])

print("ERA5-Land first:", era5_land_time[0])
print("ERA5-Land last :", era5_land_time[-1])

print("Same number of timestamps:",
      len(era5_time) == len(era5_land_time))

print("Exact timestamps match:",
      np.array_equal(era5_time, era5_land_time))


# --------------------------------------------------
# Check geographic bounds
# --------------------------------------------------

print("\n[3] Geographic bounds")

print("ERA5 latitude:",
      float(era5_instant.latitude.min()),
      "to",
      float(era5_instant.latitude.max()))

print("ERA5 longitude:",
      float(era5_instant.longitude.min()),
      "to",
      float(era5_instant.longitude.max()))

print("\nERA5-Land latitude:",
      float(era5_land.latitude.min()),
      "to",
      float(era5_land.latitude.max()))

print("ERA5-Land longitude:",
      float(era5_land.longitude.min()),
      "to",
      float(era5_land.longitude.max()))


# --------------------------------------------------
# Check variables
# --------------------------------------------------

print("\n[4] Variables")

print("ERA5 instant:",
      list(era5_instant.data_vars))

print("ERA5 accumulated:",
      list(era5_accum.data_vars))

print("ERA5-Land:",
      list(era5_land.data_vars))


# --------------------------------------------------
# Check NaN
# --------------------------------------------------

print("\n[5] Missing values")

datasets = {
    "ERA5 t2m": era5_instant.t2m,
    "ERA5 u10": era5_instant.u10,
    "ERA5 v10": era5_instant.v10,
    "ERA5 tp": era5_accum.tp,

    "ERA5-Land t2m": era5_land.t2m,
    "ERA5-Land u10": era5_land.u10,
    "ERA5-Land v10": era5_land.v10,
    "ERA5-Land tp": era5_land.tp,
}


for name, data in datasets.items():

    nan_count = int(data.isnull().sum())

    print(f"{name:20s}: NaN = {nan_count}")


# --------------------------------------------------
# Resolution
# --------------------------------------------------

print("\n[6] Resolution")

era5_lat_res = abs(
    float(era5_instant.latitude.values[1])
    - float(era5_instant.latitude.values[0])
)

era5_lon_res = abs(
    float(era5_instant.longitude.values[1])
    - float(era5_instant.longitude.values[0])
)

land_lat_res = abs(
    float(era5_land.latitude.values[1])
    - float(era5_land.latitude.values[0])
)

land_lon_res = abs(
    float(era5_land.longitude.values[1])
    - float(era5_land.longitude.values[0])
)


print(f"ERA5 latitude resolution : {era5_lat_res}°")
print(f"ERA5 longitude resolution: {era5_lon_res}°")

print(f"ERA5-Land latitude resolution : {land_lat_res}°")
print(f"ERA5-Land longitude resolution: {land_lon_res}°")


# --------------------------------------------------
# Close
# --------------------------------------------------

era5_instant.close()
era5_accum.close()
era5_land.close()


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)