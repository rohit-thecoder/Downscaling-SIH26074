import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import torch

from scipy.interpolate import RegularGridInterpolator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.downscaler import WeatherDownscaler


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SIH 26074 Weather Downscaling API",
    description="AI based weather downscaling from 0.25° to 0.1°",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

downscaler = WeatherDownscaler()


# ============================================================
# GIS
# ============================================================

PANCHAYAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "gis"
    / "panchayat"
    / "jharkhand_panchayats_clean.geojson"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class WeatherInput(BaseModel):

    temperature: float = Field(
        ...,
        description="Coarse-grid 2m temperature in Celsius"
    )

    u_wind: float = Field(
        ...,
        description="10m U wind component in m/s"
    )

    v_wind: float = Field(
        ...,
        description="10m V wind component in m/s"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "CNN",
        "input_resolution": "0.25°",
        "output_resolution": "0.1°",
        "variables": [
            "temperature",
            "u_wind",
            "v_wind"
        ]
    }


# ============================================================
# DYNAMIC PANCHAYAT PREDICTION
# ============================================================

@app.post("/predict-panchayats")
def predict_panchayats(request: WeatherInput):

    # --------------------------------------------------------
    # 1. CREATE COARSE WEATHER GRID
    # --------------------------------------------------------

    x_raw = torch.zeros(
        (1, 3, 25, 21),
        dtype=torch.float32
    )

    x_raw[:, 0, :, :] = request.temperature
    x_raw[:, 1, :, :] = request.u_wind
    x_raw[:, 2, :, :] = request.v_wind


    # --------------------------------------------------------
    # 2. CNN PREDICTION
    # --------------------------------------------------------

    prediction = downscaler.predict(x_raw)

    prediction = prediction[0].numpy()

    temperature_grid = prediction[0]
    u_wind_grid = prediction[1]
    v_wind_grid = prediction[2]


    # --------------------------------------------------------
    # 3. HIGH-RESOLUTION GRID
    # --------------------------------------------------------

    latitudes = np.linspace(
        27.0,
        21.0,
        61
    )

    longitudes = np.linspace(
        85.0,
        90.0,
        51
    )

    # scipy requires increasing latitude
    latitudes_inc = latitudes[::-1]

    temperature_inc = temperature_grid[::-1, :]
    u_wind_inc = u_wind_grid[::-1, :]
    v_wind_inc = v_wind_grid[::-1, :]


    temp_interp = RegularGridInterpolator(
        (latitudes_inc, longitudes),
        temperature_inc,
        bounds_error=False,
        fill_value=np.nan
    )

    u_interp = RegularGridInterpolator(
        (latitudes_inc, longitudes),
        u_wind_inc,
        bounds_error=False,
        fill_value=np.nan
    )

    v_interp = RegularGridInterpolator(
        (latitudes_inc, longitudes),
        v_wind_inc,
        bounds_error=False,
        fill_value=np.nan
    )


    # --------------------------------------------------------
    # 4. LOAD PANCHAYATS
    # --------------------------------------------------------

    gdf = gpd.read_file(
        PANCHAYAT_FILE
    ).to_crs("EPSG:4326")


    # --------------------------------------------------------
    # 5. REPRESENTATIVE POINT
    # --------------------------------------------------------

    representative_points = (
        gdf.geometry.representative_point()
    )

    coordinates = np.column_stack(
        [
            representative_points.y.values,
            representative_points.x.values
        ]
    )


    # --------------------------------------------------------
    # 6. PANCHAYAT WEATHER
    # --------------------------------------------------------

    gdf["temperature_c"] = temp_interp(
        coordinates
    )

    gdf["u_wind_ms"] = u_interp(
        coordinates
    )

    gdf["v_wind_ms"] = v_interp(
        coordinates
    )


    # --------------------------------------------------------
    # 7. WIND
    # --------------------------------------------------------

    gdf["wind_speed_ms"] = np.sqrt(
        gdf["u_wind_ms"] ** 2
        + gdf["v_wind_ms"] ** 2
    )

    gdf["wind_direction_deg"] = (
        np.degrees(
            np.arctan2(
                -gdf["u_wind_ms"],
                -gdf["v_wind_ms"]
            )
        ) + 360
    ) % 360


    # --------------------------------------------------------
    # 8. CLEAN NaN
    # --------------------------------------------------------

    numeric_columns = [
        "temperature_c",
        "u_wind_ms",
        "v_wind_ms",
        "wind_speed_ms",
        "wind_direction_deg"
    ]

    for column in numeric_columns:
        gdf[column] = gdf[column].replace(
            {np.nan: None}
        )


    # --------------------------------------------------------
    # 9. GEOJSON
    # --------------------------------------------------------

    geojson = gdf.__geo_interface__


    # --------------------------------------------------------
    # 10. SUMMARY
    # --------------------------------------------------------

    valid_temperature = gdf[
        "temperature_c"
    ].dropna()

    return {
        "status": "success",

        "input": {
            "temperature": request.temperature,
            "u_wind": request.u_wind,
            "v_wind": request.v_wind
        },

        "input_resolution": "0.25°",

        "output_resolution": "0.1°",

        "panchayat_count": len(gdf),

        "temperature_summary": {
            "min": float(valid_temperature.min()),
            "max": float(valid_temperature.max()),
            "mean": float(valid_temperature.mean())
        },

        "geojson": geojson
    }


# ============================================================
# DIRECT PANCHAYAT DATA
# ============================================================

@app.get("/panchayat/{gp_code}")
def get_panchayat(gp_code: str):

    csv_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "panchayat"
        / "panchayat_weather_final.csv"
    )

    df = pd.read_csv(csv_path)

    result = df[
        df["gp_code"].astype(str)
        == str(gp_code)
    ]

    if result.empty:

        return {
            "error": "Panchayat not found"
        }

    return result.iloc[0].to_dict()