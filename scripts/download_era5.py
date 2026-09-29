import cdsapi
from pathlib import Path

OUTPUT_DIR = Path("data/raw/era5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": ["reanalysis"],

    "variable": [
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],

    "year": ["2023"],
    "month": ["06"],
    "day": [
        "01",
        "02",
        "03",
        "04",
        "05",
    ],

    "time": [
        "00:00",
        "06:00",
        "12:00",
        "18:00",
    ],

    "area": [
        27.0,   # North
        85.0,   # West
        21.0,   # South
        90.0,   # East
    ],

    "data_format": "netcdf",
}

target = OUTPUT_DIR / "era5_sample.nc"

client.retrieve(
    dataset,
    request,
    str(target)
)

print(f"Downloaded: {target}")