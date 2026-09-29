import xarray as xr
from pathlib import Path

DATA_DIR = Path("data/raw/era5/extracted")


def inspect_file(file_path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    ds = xr.open_dataset(file_path, engine="netcdf4")

    print("\n--- Dataset ---")
    print(ds)

    print("\n--- Dimensions ---")
    for name, size in ds.sizes.items():
        print(f"{name}: {size}")

    print("\n--- Coordinates ---")
    for name in ds.coords:
        values = ds[name].values

        print(f"\n{name}")
        print(f"  shape: {values.shape}")

        if values.size <= 10:
            print(f"  values: {values}")
        else:
            print(f"  first: {values.flat[0]}")
            print(f"  last : {values.flat[-1]}")

    print("\n--- Variables ---")
    for name in ds.data_vars:
        var = ds[name]

        print(f"\n{name}")
        print(f"  dimensions: {var.dims}")
        print(f"  shape:      {var.shape}")
        print(f"  units:      {var.attrs.get('units', 'N/A')}")
        print(f"  long_name:  {var.attrs.get('long_name', 'N/A')}")

    ds.close()


for file_path in sorted(DATA_DIR.glob("*.nc")):
    inspect_file(file_path)