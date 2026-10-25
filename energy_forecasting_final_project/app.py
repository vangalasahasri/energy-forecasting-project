import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Energy Forecasting System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Forecasting System")
st.write("Cloud-Based Energy Forecasting Using Machine Learning")

# Your CSV and PNG files are inside this folder
DATA_DIR = Path(__file__).parent

st.header("Machine Learning Model Comparison")

comparison_file = DATA_DIR / "three_model_comparison.csv"

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)
    st.dataframe(comparison, use_container_width=True)
else:
    st.warning("Model comparison file not found.")

st.header("Model Evaluation Metrics")

metrics_file = DATA_DIR / "energy_model_metrics.csv"

if metrics_file.exists():
    metrics = pd.read_csv(metrics_file)
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Evaluation metrics file not found.")

st.header("Energy Forecast Graphs")

for filename, heading in [
    ("energy_forecast_graph.png", "Energy Forecast"),
    ("energy_24hour_forecast.png", "24-Hour Energy Forecast"),
]:
    image_path = DATA_DIR / filename
    if image_path.exists():
        st.subheader(heading)
        st.image(str(image_path), use_container_width=True)

st.header("Additional Analysis")

extra_files = [
    "energy_forecast_results.csv",
    "energy_24hour_forecast.csv",
    "weather_model_comparison.csv",
    "weekday_weekend_analysis.csv",
    "five_hour_deviation_forecast.csv",
    "fault_tolerance_simulation.csv",
]

for filename in extra_files:
    file_path = DATA_DIR / filename
    if file_path.exists():
        with st.expander(filename.replace("_", " ").title()):
            st.dataframe(
                pd.read_csv(file_path),
                use_container_width=True
            )
