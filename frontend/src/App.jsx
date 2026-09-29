import { useState } from "react";

import {
  MapContainer,
  TileLayer,
  GeoJSON,
} from "react-leaflet";

import {
  Activity,
  Gauge,
  MapPin,
  RefreshCw,
  Thermometer,
  Wind,
  Zap,
} from "lucide-react";

import "leaflet/dist/leaflet.css";
import "./App.css";


const API_URL = "http://127.0.0.1:8000";


// ============================================================
// TEMPERATURE COLORS
// ============================================================

function getTemperatureColor(temp) {

  if (temp == null) return "#64748b";

  if (temp < 23) return "#313695";
  if (temp < 25) return "#4575b4";
  if (temp < 27) return "#74add1";
  if (temp < 29) return "#fdae61";

  return "#d73027";
}


// ============================================================
// MAP STYLE
// ============================================================

function getFeatureStyle(feature) {

  const temperature =
    feature.properties?.temperature_c;

  return {

    fillColor:
      getTemperatureColor(
        temperature
      ),

    weight: 0.5,

    color: "#ffffff",

    fillOpacity: 0.78,
  };
}


// ============================================================
// WEATHER CARD
// ============================================================

function WeatherCard({
  icon,
  label,
  value,
  unit,
}) {

  return (

    <div className="weather-card">

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >

        {icon}

        <span>
          {label}
        </span>

      </div>

      <strong>
        {value} {unit}
      </strong>

    </div>
  );
}


// ============================================================
// MAIN APP
// ============================================================

function App() {

  // ==========================================================
  // USER INPUT
  // ==========================================================

  const [temperature, setTemperature] =
    useState(28);

  const [uWind, setUWind] =
    useState(2);

  const [vWind, setVWind] =
    useState(1);


  // ==========================================================
  // OUTPUT
  // ==========================================================

  const [geoData, setGeoData] =
    useState(null);

  const [selected, setSelected] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState(null);


  // ==========================================================
  // PREDICT
  // ==========================================================

  const generateWeather = async () => {

    setLoading(true);
    setError(null);
    setSelected(null);

    try {

      const response =
        await fetch(
          `${API_URL}/predict-panchayats`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({

              temperature:
                Number(temperature),

              u_wind:
                Number(uWind),

              v_wind:
                Number(vWind),

            }),
          }
        );


      if (!response.ok) {

        throw new Error(
          `API Error: ${response.status}`
        );

      }


      const result =
        await response.json();


      if (
        result.status !==
        "success"
      ) {

        throw new Error(
          "Prediction failed"
        );

      }


      setGeoData(
        result.geojson
      );


    } catch (err) {

      console.error(err);

      setError(
        err.message ||
        "Unable to generate prediction."
      );

    } finally {

      setLoading(false);

    }
  };


  // ==========================================================
  // PANCHAYAT CLICK
  // ==========================================================

  const onEachPanchayat = (
    feature,
    layer
  ) => {

    const p =
      feature.properties || {};


    layer.bindTooltip(
      `
      <strong>
        ${p.gp_name || "Panchayat"}
      </strong>
      <br/>
      Temperature:
      ${
        p.temperature_c != null
          ? Number(
              p.temperature_c
            ).toFixed(1)
          : "--"
      }°C
      `,
      {
        sticky: true,
      }
    );


    layer.on({

      click: () => {

        setSelected(p);

      },


      mouseover: (event) => {

        event.target.setStyle({

          weight: 2,

          color: "#111827",

          fillOpacity: 0.95,

        });

        event.target.bringToFront();

      },


      mouseout: (event) => {

        event.target.setStyle(
          getFeatureStyle(
            feature
          )
        );

      },

    });

  };


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <div className="app">


      {/* ====================================================
          HEADER
      ===================================================== */}

      <header className="main-header">

        <div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >

            <Zap size={25} />

            <h1>
              SIH 26074
            </h1>

          </div>

          <p>
            AI Weather Downscaling
            & Panchayat-Level Weather
          </p>

        </div>


        <div className="resolution-badge">

          <Gauge size={18} />

          0.25° → 0.1°

        </div>

      </header>


      {/* ====================================================
          INPUT SECTION
      ===================================================== */}

      <section className="input-section">

        <div className="input-header">

          <div>

            <h2>
              Weather Input
            </h2>

            <p>
              Enter coarse-resolution
              weather conditions.
            </p>

          </div>

        </div>


        <div className="input-grid">


          {/* TEMPERATURE */}

          <div className="input-card">

            <div className="input-icon">

              <Thermometer
                size={24}
              />

            </div>

            <label>
              Temperature
            </label>

            <input
              type="number"
              step="0.1"
              value={temperature}
              onChange={(e) =>
                setTemperature(
                  e.target.value
                )
              }
            />

            <span>
              °C
            </span>

          </div>


          {/* U WIND */}

          <div className="input-card">

            <div className="input-icon">

              <Wind
                size={24}
              />

            </div>

            <label>
              U-Wind Component
            </label>

            <input
              type="number"
              step="0.1"
              value={uWind}
              onChange={(e) =>
                setUWind(
                  e.target.value
                )
              }
            />

            <span>
              m/s
            </span>

          </div>


          {/* V WIND */}

          <div className="input-card">

            <div className="input-icon">

              <Activity
                size={24}
              />

            </div>

            <label>
              V-Wind Component
            </label>

            <input
              type="number"
              step="0.1"
              value={vWind}
              onChange={(e) =>
                setVWind(
                  e.target.value
                )
              }
            />

            <span>
              m/s
            </span>

          </div>


        </div>


        {/* GENERATE BUTTON */}

        <button
          className="generate-button"
          onClick={
            generateWeather
          }
          disabled={loading}
        >

          <RefreshCw
            size={20}
            className={
              loading
                ? "spin"
                : ""
            }
          />

          {loading
            ? "Running AI Model..."
            : "Generate Panchayat Weather"}

        </button>


        {error && (

          <div className="error-box">

            {error}

          </div>

        )}

      </section>


      {/* ====================================================
          MODEL PIPELINE
      ===================================================== */}

      <section className="pipeline-section">

        <h2>
          AI Downscaling Pipeline
        </h2>

        <div className="pipeline">

          <div>
            <strong>
              User Input
            </strong>

            <span>
              0.25° Weather
            </span>
          </div>

          <b>→</b>

          <div>
            <strong>
              CNN
            </strong>

            <span>
              Spatial Downscaling
            </span>
          </div>

          <b>→</b>

          <div>
            <strong>
              0.1°
            </strong>

            <span>
              High Resolution
            </span>
          </div>

          <b>→</b>

          <div>
            <strong>
              GIS
            </strong>

            <span>
              Panchayat Mapping
            </span>
          </div>

        </div>

      </section>


      {/* ====================================================
          MAP SECTION
      ===================================================== */}

      {geoData && (

        <section className="panchayat-section">

          <div className="section-header">

            <div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                }}
              >

                <MapPin size={24} />

                <h2>
                  Panchayat Weather
                </h2>

              </div>

              <p>
                AI-generated weather
                at Panchayat level
              </p>

            </div>

          </div>


          <div className="map-layout">


            {/* MAP */}

            <div className="map-container">

              <MapContainer
                center={[
                  23.6,
                  85.3,
                ]}
                zoom={7}
                scrollWheelZoom={true}
                style={{
                  height: "650px",
                  width: "100%",
                }}
              >

                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <GeoJSON
                  key={
                    JSON.stringify(
                      geoData
                    ).length
                  }
                  data={geoData}
                  style={
                    getFeatureStyle
                  }
                  onEachFeature={
                    onEachPanchayat
                  }
                />

              </MapContainer>

            </div>


            {/* WEATHER PANEL */}

            <div className="weather-panel">

              {!selected && (

                <div className="empty-panel">

                  <MapPin size={40} />

                  <h3>
                    Select a Panchayat
                  </h3>

                  <p>
                    Click any Panchayat
                    on the map to view
                    its downscaled weather.
                  </p>

                </div>

              )}


              {selected && (

                <>

                  <h2>
                    {
                      selected.gp_name ||
                      "Panchayat"
                    }
                  </h2>

                  <p className="location">

                    {selected.dtname}

                    {" · "}

                    {selected.blkname}

                  </p>


                  <div
                    className="temperature-display"
                  >

                    <Thermometer
                      size={32}
                    />

                    <span>
                      Temperature
                    </span>

                    <strong>

                      {
                        selected.temperature_c !=
                        null
                          ? Number(
                              selected.temperature_c
                            ).toFixed(1)
                          : "--"
                      }

                      °C

                    </strong>

                  </div>


                  <WeatherCard
                    icon={
                      <Wind size={18} />
                    }
                    label="Wind Speed"
                    value={
                      selected.wind_speed_ms !=
                      null
                        ? Number(
                            selected.wind_speed_ms
                          ).toFixed(2)
                        : "--"
                    }
                    unit="m/s"
                  />


                  <WeatherCard
                    icon={
                      <Activity size={18} />
                    }
                    label="Wind Direction"
                    value={
                      selected.wind_direction_deg !=
                      null
                        ? Number(
                            selected.wind_direction_deg
                          ).toFixed(0)
                        : "--"
                    }
                    unit="°"
                  />


                  <WeatherCard
                    icon={
                      <Wind size={18} />
                    }
                    label="U-Wind"
                    value={
                      selected.u_wind_ms !=
                      null
                        ? Number(
                            selected.u_wind_ms
                          ).toFixed(2)
                        : "--"
                    }
                    unit="m/s"
                  />


                  <WeatherCard
                    icon={
                      <Wind size={18} />
                    }
                    label="V-Wind"
                    value={
                      selected.v_wind_ms !=
                      null
                        ? Number(
                            selected.v_wind_ms
                          ).toFixed(2)
                        : "--"
                    }
                    unit="m/s"
                  />

                </>

              )}

            </div>

          </div>


          {/* LEGEND */}

          <div className="temperature-legend">

            <strong>
              Temperature:
            </strong>

            <span>
              🔵 &lt;23°C
            </span>

            <span>
              🟦 23–25°C
            </span>

            <span>
              🟢 25–27°C
            </span>

            <span>
              🟠 27–29°C
            </span>

            <span>
              🔴 &gt;29°C
            </span>

          </div>

        </section>

      )}


      {/* ====================================================
          FOOTER
      ===================================================== */}

      <footer className="footer">

        SIH 26074 · AI-Based
        Weather Downscaling

      </footer>


    </div>
  );
}


export default App;