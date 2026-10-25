import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Energy Forecasting and Optimization System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Forecasting and Optimization System")
st.write(
    "Cloud-Based Energy Consumption Forecasting "
    "and Optimization Using Machine Learning"
)

DATA_DIR = Path(__file__).parent

# --------------------------------------------------
# 1. MACHINE LEARNING MODEL COMPARISON
# --------------------------------------------------

st.header("Machine Learning Model Comparison")

comparison_file = DATA_DIR / "three_model_comparison.csv"

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)
    st.dataframe(comparison, use_container_width=True)
else:
    st.warning("Model comparison file not found.")

# --------------------------------------------------
# 2. MODEL EVALUATION METRICS
# --------------------------------------------------

st.header("Model Evaluation Metrics")

metrics_file = DATA_DIR / "energy_model_metrics.csv"

if metrics_file.exists():
    metrics = pd.read_csv(metrics_file)
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Evaluation metrics file not found.")

# --------------------------------------------------
# 3. FORECAST GRAPHS
# --------------------------------------------------

st.header("Energy Forecast Graphs")

for filename, heading in [
    ("energy_forecast_graph.png", "Energy Forecast"),
    ("energy_24hour_forecast.png", "24-Hour Energy Forecast"),
]:
    image_path = DATA_DIR / filename
    if image_path.exists():
        st.subheader(heading)
        st.image(str(image_path), use_container_width=True)

# --------------------------------------------------
# 4. ENERGY OPTIMIZATION
# --------------------------------------------------

st.header("⚡ Energy Cost Optimization")

st.write(
    "Use your forecasted energy demand to explore how shifting "
    "flexible consumption from expensive hours to cheaper hours "
    "could reduce estimated electricity costs."
)

forecast_file = DATA_DIR / "energy_24hour_forecast.csv"

if forecast_file.exists():

    forecast = pd.read_csv(forecast_file)

    required_columns = ["Forecast_Hour", "Forecast_Load_MW"]

    if not all(col in forecast.columns for col in required_columns):
        st.error(
            "The forecast CSV must contain Forecast_Hour "
            "and Forecast_Load_MW columns."
        )
    else:
        forecast = forecast[required_columns].copy()

        forecast["Forecast_Hour"] = pd.to_datetime(
            forecast["Forecast_Hour"], errors="coerce"
        )
        forecast["Forecast_Load_MW"] = pd.to_numeric(
            forecast["Forecast_Load_MW"], errors="coerce"
        )

        forecast = forecast.dropna(
            subset=["Forecast_Hour", "Forecast_Load_MW"]
        )

        forecast = forecast[
            forecast["Forecast_Load_MW"] >= 0
        ].sort_values("Forecast_Hour").reset_index(drop=True)

        if forecast.empty:
            st.error("No valid forecast data is available.")

        else:
            # One row represents one hourly interval.
            forecast = forecast.drop_duplicates(
                subset=["Forecast_Hour"]
            ).reset_index(drop=True)

            if len(forecast) > 1:
                gaps = forecast["Forecast_Hour"].diff().dropna()
                if not (gaps == pd.Timedelta(hours=1)).all():
                    st.warning(
                        "Your forecast does not contain consecutive "
                        "hourly intervals. Results assume each row "
                        "represents one hour."
                    )

            forecast["Hour"] = forecast["Forecast_Hour"].dt.hour

            # Illustrative time-of-use prices in INR per kWh.
            # Replace these with your actual electricity tariff.
            def default_price(hour):
                if hour in [0, 1, 2, 3, 4, 5]:
                    return 5.0
                elif hour in [18, 19, 20, 21]:
                    return 12.0
                else:
                    return 8.0

            forecast["Price_INR_per_kWh"] = (
                forecast["Hour"].apply(default_price)
            )

            st.subheader("Forecast and Hourly Electricity Prices")

            st.caption(
                "Prices below are illustrative, not live electricity "
                "tariffs. Edit them to match your applicable tariff."
            )

            edited = st.data_editor(
                forecast[
                    [
                        "Forecast_Hour",
                        "Forecast_Load_MW",
                        "Price_INR_per_kWh",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                disabled=["Forecast_Hour", "Forecast_Load_MW"],
                column_config={
                    "Forecast_Hour": st.column_config.DatetimeColumn(
                        "Forecast Hour",
                        format="YYYY-MM-DD HH:mm"
                    ),
                    "Forecast_Load_MW": st.column_config.NumberColumn(
                        "Forecast Load (MW)",
                        format="%.2f"
                    ),
                    "Price_INR_per_kWh": st.column_config.NumberColumn(
                        "Electricity Price (₹/kWh)",
                        min_value=0.0,
                        max_value=100000.0,
                        step=0.5,
                        format="₹%.2f"
                    ),
                },
            )

            st.subheader("Optimization Settings")

            flexible_percent = st.slider(
                "Maximum flexible load (%)",
                min_value=0,
                max_value=30,
                value=10,
                step=5,
                help=(
                    "Assumed share of each hour's forecast load that "
                    "can be shifted. Use zero if no load is flexible."
                ),
            )

            capacity_percent = st.slider(
                "Extra load capacity at receiving hours (%)",
                min_value=0,
                max_value=50,
                value=20,
                step=5,
                help=(
                    "Maximum additional load allowed at a receiving "
                    "hour, as a percentage of its forecast load."
                ),
            )

            if st.button("Optimize Energy Schedule", type="primary"):

                base_load = forecast["Forecast_Load_MW"].to_numpy(
                    dtype=float
                )
                prices = pd.to_numeric(
                    edited["Price_INR_per_kWh"],
                    errors="coerce"
                ).to_numpy(dtype=float)

                if (
                    not np.isfinite(prices).all()
                    or (prices < 0).any()
                ):
                    st.error("Enter valid, non-negative electricity prices.")

                elif flexible_percent == 0 or capacity_percent == 0:
                    st.info(
                        "No load can be shifted with the selected "
                        "settings. Increase the flexible-load and "
                        "receiving-capacity percentages to explore "
                        "a possible schedule."
                    )

                else:
                    optimized_load = base_load.copy()

                    # Maximum amount that may leave each hour.
                    donor_capacity = (
                        base_load * flexible_percent / 100.0
                    )

                    # Maximum extra load that each hour can receive.
                    receiver_capacity = (
                        base_load * capacity_percent / 100.0
                    )

                    # Try the most expensive hours first and move
                    # flexible load only to strictly cheaper hours.
                    donors = np.argsort(-prices)
                    receivers = np.argsort(prices)

                    for donor in donors:
                        for receiver in receivers:

                            if prices[receiver] >= prices[donor]:
                                break

                            if donor == receiver:
                                continue

                            if donor_capacity[donor] <= 1e-9:
                                break

                            if receiver_capacity[receiver] <= 1e-9:
                                continue

                            amount = min(
                                donor_capacity[donor],
                                receiver_capacity[receiver],
                            )

                            optimized_load[donor] -= amount
                            optimized_load[receiver] += amount

                            donor_capacity[donor] -= amount
                            receiver_capacity[receiver] -= amount

                    results = pd.DataFrame({
                        "Forecast_Hour": forecast["Forecast_Hour"],
                        "Electricity_Price_INR_per_kWh": prices,
                        "Original_Load_MW": base_load,
                        "Optimized_Load_MW": optimized_load,
                    })

                    # Each row is assumed to represent one hour.
                    # MW * 1000 kWh/MWh * ₹/kWh = ₹ per hour.
                    results["Original_Cost_INR"] = (
                        results["Original_Load_MW"]
                        * 1000
                        * results["Electricity_Price_INR_per_kWh"]
                    )

                    results["Optimized_Cost_INR"] = (
                        results["Optimized_Load_MW"]
                        * 1000
                        * results["Electricity_Price_INR_per_kWh"]
                    )

                    original_cost = results[
                        "Original_Cost_INR"
                    ].sum()

                    optimized_cost = results[
                        "Optimized_Cost_INR"
                    ].sum()

                    savings = original_cost - optimized_cost

                    savings_percent = (
                        savings / original_cost * 100
                        if original_cost > 0 else 0
                    )

                    total_original_energy = base_load.sum()
                    total_optimized_energy = optimized_load.sum()

                    st.subheader("Optimization Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Original Estimated Cost",
                        f"₹{original_cost:,.2f}"
                    )

                    col2.metric(
                        "Optimized Estimated Cost",
                        f"₹{optimized_cost:,.2f}"
                    )

                    col3.metric(
                        "Estimated Savings",
                        f"₹{savings:,.2f}",
                        delta=f"{savings_percent:.2f}%"
                    )

                    st.write(
                        f"Original energy: "
                        f"{total_original_energy:,.2f} MWh"
                    )

                    st.write(
                        f"Optimized energy: "
                        f"{total_optimized_energy:,.2f} MWh"
                    )

                    st.caption(
                        "This demonstration shifts flexible load "
                        "between hourly periods while preserving total "
                        "forecast energy. It assumes one-hour intervals "
                        "and uses the prices and capacity limits "
                        "entered above. Savings are estimates, not "
                        "guaranteed actual savings. This is a simplified "
                        "greedy scheduling heuristic, not a guarantee "
                        "of a globally optimal solution."
                    )

                    st.subheader("Original vs Optimized Load")

                    chart_data = results.set_index("Forecast_Hour")[
                        ["Original_Load_MW", "Optimized_Load_MW"]
                    ]

                    st.line_chart(chart_data)

                    st.subheader("Cost Comparison by Hour")

                    cost_chart = results.set_index("Forecast_Hour")[
                        ["Original_Cost_INR", "Optimized_Cost_INR"]
                    ]

                    st.bar_chart(cost_chart)

                    st.subheader("Detailed Optimization Schedule")

                    st.dataframe(
                        results.round(3),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Optimization Results CSV",
                        data=csv_data,
                        file_name="energy_optimization_results.csv",
                        mime="text/csv",
                    )

else:
    st.warning(
        "energy_24hour_forecast.csv was not found. "
        "Add it to the application folder to enable optimization."
    )

# --------------------------------------------------
# 5. ADDITIONAL ANALYSIS
# --------------------------------------------------

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

   

