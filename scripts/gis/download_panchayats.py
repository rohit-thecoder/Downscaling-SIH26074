from pathlib import Path
import requests
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gis"
    / "panchayat"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


URL = (
    "https://github.com/"
    "yashveeeeeeer/india-geodata/"
    "releases/download/"
    "admin/panchayats/"
    "LGD_panchayats.parquet"
)


DOWNLOAD_PATH = (
    OUTPUT_DIR
    / "LGD_panchayats.parquet"
)


print("=" * 60)
print("PANCHAYAT DATASET DOWNLOAD")
print("=" * 60)

print()
print("Source:")
print(URL)

print()
print("Downloading LGD Panchayat dataset...")
print("This file is large, so this may take some time.")


response = requests.get(
    URL,
    stream=True,
    timeout=60
)

response.raise_for_status()


total = int(
    response.headers.get(
        "content-length",
        0
    )
)

downloaded = 0


with open(
    DOWNLOAD_PATH,
    "wb"
) as file:

    for chunk in response.iter_content(
        chunk_size=1024 * 1024
    ):

        if not chunk:
            continue

        file.write(chunk)

        downloaded += len(chunk)

        if total:

            percent = (
                downloaded / total
            ) * 100

            print(
                f"\rDownloaded: "
                f"{percent:.1f}%",
                end=""
            )


print()
print()
print("Download complete.")

print()
print("Loading Parquet...")

gdf = gpd.read_parquet(
    DOWNLOAD_PATH
)

print(
    "Total Panchayat records:",
    len(gdf)
)

print()
print("Columns:")

for column in gdf.columns:

    print(
        " -",
        column
    )

print()
print("CRS:")

print(
    gdf.crs
)


print()
print("Bounds:")

print(
    gdf.total_bounds
)


# --------------------------------------------------
# FIND JHARKHAND COLUMN
# --------------------------------------------------

state_columns = [
    column
    for column in gdf.columns
    if column.lower() in [
        "stname",
        "state_name",
        "state"
    ]
]


print()
print(
    "Possible state columns:",
    state_columns
)


# --------------------------------------------------
# FILTER JHARKHAND
# --------------------------------------------------

print()
print("Filtering Jharkhand Panchayats...")


STATE_COLUMN = "stname"


print(
    "Using state column:",
    STATE_COLUMN
)


print()
print(
    "Unique state examples:"
)

print(
    gdf[STATE_COLUMN]
    .dropna()
    .astype(str)
    .unique()[:20]
)


jharkhand = gdf[
    gdf[STATE_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
    == "jharkhand"
].copy()


print()
print(
    "Jharkhand Panchayats:",
    len(jharkhand)
)


if len(jharkhand) == 0:

    print()
    print(
        "ERROR: No Jharkhand Panchayats found."
    )

    raise SystemExit(1)


# --------------------------------------------------
# SAVE JHARKHAND DATA
# --------------------------------------------------

output = (
    OUTPUT_DIR
    / "jharkhand_panchayats.geojson"
)


jharkhand.to_file(
    output,
    driver="GeoJSON"
)


print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print()
print(
    "Jharkhand Panchayats:",
    len(jharkhand)
)

print()
print(
    "Saved:"
)

print(output)

print()
print(
    "CRS:",
    jharkhand.crs
)

print()
print(
    "Bounds:"
)

print(
    jharkhand.total_bounds
)