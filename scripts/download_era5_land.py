import cdsapi
from pathlib import Path


OUTPUT_DIR = Path("data/raw/era5_land")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

dataset = "reanalysis-era5-land"


request = {
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

    "download_format": "zip",
}


target = OUTPUT_DIR / "era5_land_sample.zip"


print("Starting ERA5-Land download...")

client.retrieve(
    dataset,
    request,
    str(target)
)

print(f"\nDownloaded successfully:")
print(target)