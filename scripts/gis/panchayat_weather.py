import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import torch

from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import box
from src.inference.downscaler import WeatherDownscaler


# ============================================================
# PATHS
# ============================================================

PANCHAYAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "gis"
    / "panchayat"
    / "jharkhand_panchayats_clean.geojson"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tensors"
    / "X_test.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "panchayat"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "panchayat_weather_final.csv"
GEOJSON_OUTPUT = OUTPUT_DIR / "panchayat_weather_final.geojson"


# ============================================================
# CNN GRID
# ============================================================

LAT_MIN = 21.0
LAT_MAX = 27.0

LON_MIN = 85.0
LON_MAX = 90.0

GRID_ROWS = 61
GRID_COLS = 51


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_INDEX = 0


print("=" * 70)
print("FINAL PANCHAYAT WEATHER DOWNscaling PIPELINE")
print("=" * 70)


# ============================================================
# 1. LOAD PANCHAYATS
# ============================================================

print("\n[1/6] Loading Panchayat boundaries...")

panchayats = gpd.read_file(PANCHAYAT_FILE)

print("Total Panchayats:", len(panchayats))
print("Original CRS:", panchayats.crs)

# Normalize CRS
panchayats = panchayats.to_crs("EPSG:4326")

print("Normalized CRS:", panchayats.crs)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2/6] Loading weather input...")

X_test = torch.load(
    TEST_FILE,
    weights_only=False
)

print("X_test shape:", X_test.shape)


# ============================================================
# 3. CNN INFERENCE
# ============================================================

print("\n[3/6] Running trained CNN...")

downscaler = WeatherDownscaler()

x_normalized = X_test[SAMPLE_INDEX].unsqueeze(0)

# IMPORTANT:
# X_test is already normalized.
# Convert back to raw values before inference.

x_raw = (
    x_normalized * downscaler.X_std
    + downscaler.X_mean
)

prediction = downscaler.predict(x_raw)

prediction = prediction[0].numpy()

print("CNN output:", prediction.shape)

temperature_grid = prediction[0]
u_wind_grid = prediction[1]
v_wind_grid = prediction[2]


# ============================================================
# 4. CREATE 0.1 DEGREE GRID
# ============================================================

print("\n[4/6] Creating high-resolution weather grid...")

latitudes = np.linspace(
    LAT_MAX,
    LAT_MIN,
    GRID_ROWS
)

longitudes = np.linspace(
    LON_MIN,
    LON_MAX,
    GRID_COLS
)

# scipy interpolation requires increasing coordinates
latitudes_inc = latitudes[::-1]

temperature_inc = temperature_grid[::-1, :]
u_wind_inc = u_wind_grid[::-1, :]
v_wind_inc = v_wind_grid[::-1, :]


temperature_interpolator = RegularGridInterpolator(
    (latitudes_inc, longitudes),
    temperature_inc,
    bounds_error=False,
    fill_value=np.nan
)

u_wind_interpolator = RegularGridInterpolator(
    (latitudes_inc, longitudes),
    u_wind_inc,
    bounds_error=False,
    fill_value=np.nan
)

v_wind_interpolator = RegularGridInterpolator(
    (latitudes_inc, longitudes),
    v_wind_inc,
    bounds_error=False,
    fill_value=np.nan
)


# ============================================================
# 5. CLIP PANCHAYATS TO CNN COVERAGE
# ============================================================

print("\n[5/6] Assigning weather to Panchayat locations...")

weather_bbox = gpd.GeoDataFrame(
    geometry=[
        box(
            LON_MIN,
            LAT_MIN,
            LON_MAX,
            LAT_MAX
        )
    ],
    crs="EPSG:4326"
)

# Keep only Panchayats intersecting CNN coverage
panchayats = gpd.clip(
    panchayats,
    weather_bbox
)

print(
    "Panchayats inside CNN coverage:",
    len(panchayats)
)


# ============================================================
# REPRESENTATIVE POINT
# ============================================================

# representative_point() guarantees the point lies
# inside the polygon.

panchayats["weather_point"] = (
    panchayats.geometry.representative_point()
)

weather_points = panchayats["weather_point"]

coordinates = np.column_stack(
    [
        weather_points.y.values,
        weather_points.x.values
    ]
)


# ============================================================
# INTERPOLATE CNN WEATHER
# ============================================================

panchayats["temperature_c"] = (
    temperature_interpolator(coordinates)
)

panchayats["u_wind_ms"] = (
    u_wind_interpolator(coordinates)
)

panchayats["v_wind_ms"] = (
    v_wind_interpolator(coordinates)
)


# ============================================================
# WIND SPEED + DIRECTION
# ============================================================

panchayats["wind_speed_ms"] = np.sqrt(
    panchayats["u_wind_ms"] ** 2
    + panchayats["v_wind_ms"] ** 2
)

panchayats["wind_direction_deg"] = (
    np.degrees(
        np.arctan2(
            -panchayats["u_wind_ms"],
            -panchayats["v_wind_ms"]
        )
    )
    + 360
) % 360


# ============================================================
# QUALITY FLAG
# ============================================================

panchayats["weather_available"] = (
    panchayats["temperature_c"].notna()
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "gp_code",
    "gp_name",
    "dtname",
    "blkname",
    "temperature_c",
    "u_wind_ms",
    "v_wind_ms",
    "wind_speed_ms",
    "wind_direction_deg",
    "weather_available",
    "geometry"
]

final_gdf = panchayats[
    [
        column
        for column in output_columns
        if column in panchayats.columns
    ]
].copy()


# ============================================================
# SAVE GEOJSON
# ============================================================

final_gdf.to_file(
    GEOJSON_OUTPUT,
    driver="GeoJSON"
)


# ============================================================
# SAVE CSV
# ============================================================

csv_gdf = final_gdf.drop(
    columns=["geometry"]
)

csv_gdf.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

available = int(
    final_gdf["weather_available"].sum()
)

total = len(final_gdf)

print("\n" + "=" * 70)
print("FINAL PANCHAYAT WEATHER MODEL COMPLETE")
print("=" * 70)

print("\nTotal Panchayats:", total)
print("Weather available:", available)
print(
    "Coverage:",
    round(available / total * 100, 2),
    "%"
)

print("\nTemperature:")
print(
    final_gdf["temperature_c"].describe()
)

print("\nWind speed:")
print(
    final_gdf["wind_speed_ms"].describe()
)

print("\nCSV:")
print(CSV_OUTPUT)

print("\nGeoJSON:")
print(GEOJSON_OUTPUT)

print("\nSample:")
print(
    csv_gdf.head(10).to_string(index=False)
)

print("\n" + "=" * 70)