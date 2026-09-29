from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "gis"
    / "panchayat"
    / "jharkhand_panchayats.geojson"
)


print("=" * 60)
print("PANCHAYAT GIS INSPECTION")
print("=" * 60)


# --------------------------------------------------
# LOAD
# --------------------------------------------------

print()
print("Loading Panchayat GeoJSON...")

gdf = gpd.read_file(FILE)


# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------

print()
print("Total Panchayats:")
print(len(gdf))


print()
print("CRS:")
print(gdf.crs)


print()
print("Bounds:")
print(gdf.total_bounds)


# --------------------------------------------------
# COLUMNS
# --------------------------------------------------

print()
print("Columns:")

for column in gdf.columns:

    print(
        " -",
        column
    )


# --------------------------------------------------
# IMPORTANT ATTRIBUTES
# --------------------------------------------------

print()
print("Sample Panchayat records:")

columns = [
    "gp_code",
    "gp_name",
    "stname",
    "dtname",
    "blkname",
    "geometry"
]


available = [
    column
    for column in columns
    if column in gdf.columns
]


print(
    gdf[available].head(10)
)


# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------

print()
print("Geometry types:")

print(
    gdf.geometry.geom_type.value_counts()
)


# --------------------------------------------------
# INVALID GEOMETRIES
# --------------------------------------------------

invalid = (
    ~gdf.geometry.is_valid
).sum()


print()
print(
    "Invalid geometries:",
    invalid
)


# --------------------------------------------------
# EMPTY GEOMETRIES
# --------------------------------------------------

empty = (
    gdf.geometry.is_empty
).sum()


print(
    "Empty geometries:",
    empty
)


# --------------------------------------------------
# DISTRICTS
# --------------------------------------------------

print()
print(
    "Number of districts:"
)

print(
    gdf["dtname"]
    .nunique()
)


print()
print(
    "Districts:"
)

print(
    gdf["dtname"]
    .dropna()
    .sort_values()
    .unique()
)


# --------------------------------------------------
# BLOCKS
# --------------------------------------------------

print()
print(
    "Number of blocks:"
)

print(
    gdf["blkname"]
    .nunique()
)


print()
print("=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)