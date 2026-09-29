from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "gis"
    / "panchayat"
    / "jharkhand_panchayats.geojson"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "gis"
    / "panchayat"
    / "jharkhand_panchayats_clean.geojson"
)


print("=" * 60)
print("PREPARING PANCHAYAT GEOMETRIES")
print("=" * 60)


print()
print("Loading dataset...")

gdf = gpd.read_file(
    INPUT_FILE
)


print(
    "Panchayats:",
    len(gdf)
)


print()
print(
    "Invalid geometries before:",
    (~gdf.geometry.is_valid).sum()
)


# --------------------------------------------------
# FIX INVALID GEOMETRIES
# --------------------------------------------------

print()
print("Fixing invalid geometries...")


gdf["geometry"] = (
    gdf.geometry.make_valid()
)


# Remove empty geometries

gdf = gdf[
    gdf.geometry.notna()
]


gdf = gdf[
    ~gdf.geometry.is_empty
]


print(
    "Invalid geometries after:",
    (~gdf.geometry.is_valid).sum()
)


# --------------------------------------------------
# FILTER CNN COVERAGE
# --------------------------------------------------

print()
print(
    "Filtering to CNN geographic coverage..."
)


MIN_LON = 85.0
MAX_LON = 90.0

MIN_LAT = 21.0
MAX_LAT = 27.0


coverage = gpd.GeoSeries(
    [
        gpd.points_from_xy(
            [MIN_LON],
            [MIN_LAT]
        )[0]
    ],
    crs="EPSG:4326"
)


# Bounding box

from shapely.geometry import box


weather_box = box(
    MIN_LON,
    MIN_LAT,
    MAX_LON,
    MAX_LAT
)


weather_area = gpd.GeoDataFrame(
    geometry=[weather_box],
    crs="EPSG:4326"
)


# Keep Panchayats intersecting CNN region

gdf = gdf[
    gdf.geometry.intersects(
        weather_box
    )
].copy()


print()
print(
    "Panchayats intersecting CNN region:",
    len(gdf)
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

gdf.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


print()
print("=" * 60)
print("PREPARATION COMPLETE")
print("=" * 60)

print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)