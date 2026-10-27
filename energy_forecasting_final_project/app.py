
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# 1. PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="Energy Forecasting and Optimization System",
    page_icon="⚡",
    layout="wide"
)

DATA_DIR = Path(__file__).parent

st.title("⚡ Energy Forecasting and Optimization System")
st.write(
    "Cloud-Based Energy Consumption Forecasting "
    "and Optimization Using Machine Learning"
)

st.caption(
    "This application displays saved forecasting results, "
    "explores energy cost optimization, and demonstrates "
    "2-out-of-3 model voting."
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def load_csv(filename):
    """Load a CSV file if it exists."""
    path = DATA_DIR / filename

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as error:
            st.error(f"Could not read {filename}: {error}")

    return None


def two_out_of_three_voting(
    sarimax,
    sarimax_svm,
    lstm,
    tolerance=5.0
):
    """
    Compare three model forecasts.
    Two forecasts agree when their symmetric percentage
    difference is within the specified tolerance.
    """

    predictions = {
        "SARIMAX": float(sarimax),
        "SARIMAX + SVM": float(sarimax_svm),
        "LSTM": float(lstm)
    }

    values = list(predictions.values())

    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError(
            "All model predictions must be finite "
            "and non-negative."
        )

    def agree(a, b):
        denominator = (abs(a) + abs(b)) / 2

        if denominator == 0:
            return True

        percentage_difference = (
            abs(a - b) / denominator
        ) * 100

        return percentage_difference <= tolerance

    a, b, c = values

    # If all three agree, use their average.
    if agree(a, b) and agree(a, c) and agree(b, c):
        return {
            "forecast": float(np.mean(values)),
            "status": "All 3 models agree",
            "agreeing_models": list(predictions.keys()),
            "consensus": True
        }

    # Check each possible pair.
    pairs = [
        (a, b, "SARIMAX", "SARIMAX + SVM"),
        (a, c, "SARIMAX", "LSTM"),
        (b, c, "SARIMAX + SVM", "LSTM")
    ]

    agreeing_pairs = [
        pair for pair in pairs
        if agree(pair[0], pair[1])
    ]

    if agreeing_pairs:
        # Select the closest agreeing pair.
        best_pair = min(
            agreeing_pairs,
            key=lambda pair: abs(pair[0] - pair[1])
        )

        x, y, model1, model2 = best_pair

        return {
            "forecast": float((x + y) / 2),
            "status": "2-out-of-3 consensus",
            "agreeing_models": [model1, model2],
            "consensus": True
        }

    # No pair agrees: report median as a fallback,
    # NOT as a successful consensus.
    return {
        "forecast": float(np.median(values)),
        "status": "No consensus - median fallback",
        "agreeing_models": [],
        "consensus": False
    }


# ==================================================
# 3. MACHINE LEARNING MODEL COMPARISON
# ==================================================

st.header("1. Machine Learning Model Comparison")

comparison = load_csv("three_model_comparison.csv")

if comparison is not None:
    st.dataframe(comparison, use_container_width=True)
else:
    st.warning("three_model_comparison.csv was not found.")


# ==================================================
# 4. MODEL EVALUATION METRICS
# ==================================================

st.header("2. Model Evaluation Metrics")

metrics = load_csv("energy_model_metrics.csv")

if metrics is not None:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("energy_model_metrics.csv was not found.")


# ==================================================
# 5. ENERGY FORECAST GRAPHS
# ==================================================

st.header("3. Energy Forecast Graphs")

for filename, heading in [
    ("energy_forecast_graph.png", "Energy Forecast"),
    ("energy_24hour_forecast.png", "24-Hour Energy Forecast")
]:
    image_path = DATA_DIR / filename

    if image_path.exists():
        st.subheader(heading)
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(f"{filename} was not found.")


# ==================================================
# 6. ENERGY COST OPTIMIZATION
# ==================================================

st.header("4. Energy Cost Optimization")

st.write(
    "Explore how flexible energy consumption could be shifted "
    "from expensive hours to cheaper hours to reduce estimated "
    "electricity costs."
)

forecast = load_csv("energy_24hour_forecast.csv")

if forecast is None:
    st.warning(
        "Add energy_24hour_forecast.csv to the application "
        "folder to enable energy optimization."
    )

else:
    required_columns = [
        "Forecast_Hour",
        "Forecast_Load_MW"
    ]

    if not all(column in forecast.columns
               for column in required_columns):

        st.error(
            "The forecast CSV must contain Forecast_Hour "
            "and Forecast_Load_MW columns."
        )

    else:
        forecast = forecast[required_columns].copy()

        forecast["Forecast_Hour"] = pd.to_datetime(
            forecast["Forecast_Hour"],
            errors="coerce"
        )

        forecast["Forecast_Load_MW"] = pd.to_numeric(
            forecast["Forecast_Load_MW"],
            errors="coerce"
        )

        forecast = forecast.dropna(
            subset=["Forecast_Hour", "Forecast_Load_MW"]
        )

        forecast = forecast[
            forecast["Forecast_Load_MW"] >= 0
        ]

        forecast = (
            forecast.drop_duplicates(subset=["Forecast_Hour"])
            .sort_values("Forecast_Hour")
            .reset_index(drop=True)
        )

        if forecast.empty:
            st.error("No valid forecast data is available.")

        else:
            if len(forecast) > 1:
                gaps = forecast["Forecast_Hour"].diff().dropna()

                if not (gaps == pd.Timedelta(hours=1)).all():
                    st.warning(
                        "Forecast rows are not all consecutive "
                        "hourly intervals. Cost calculations assume "
                        "each row represents one hour."
                    )

            forecast["Hour"] = forecast["Forecast_Hour"].dt.hour

            # Illustrative prices; replace with actual tariff.
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

            st.subheader("Forecast and Electricity Prices")

            st.caption(
                "Prices are illustrative INR/kWh values, "
                "not live electricity tariffs. Edit them if needed."
            )

            edited = st.data_editor(
                forecast[
                    [
                        "Forecast_Hour",
                        "Forecast_Load_MW",
                        "Price_INR_per_kWh"
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                disabled=[
                    "Forecast_Hour",
                    "Forecast_Load_MW"
                ],
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
                    )
                }
            )

            st.subheader("Optimization Settings")

            flexible_percent = st.slider(
                "Maximum flexible load (%)",
                min_value=0,
                max_value=30,
                value=10,
                step=5
            )

            capacity_percent = st.slider(
                "Extra receiving capacity (%)",
                min_value=0,
                max_value=50,
                value=20,
                step=5
            )

            if st.button(
                "Optimize Energy Schedule",
                type="primary"
            ):
                base_load = forecast[
                    "Forecast_Load_MW"
                ].to_numpy(dtype=float)

                prices = pd.to_numeric(
                    edited["Price_INR_per_kWh"],
                    errors="coerce"
                ).to_numpy(dtype=float)

                if (
                    not np.isfinite(prices).all()
                    or (prices < 0).any()
                ):
                    st.error(
                        "Enter valid, non-negative electricity prices."
                    )

                elif flexible_percent == 0 or capacity_percent == 0:
                    st.info(
                        "Increase both flexibility and receiving "
                        "capacity to explore load shifting."
                    )

                else:
                    optimized_load = base_load.copy()

                    donor_capacity = (
                        base_load * flexible_percent / 100
                    )

                    receiver_capacity = (
                        base_load * capacity_percent / 100
                    )

                    # Move load from expensive to cheaper hours.
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
                                receiver_capacity[receiver]
                            )

                            optimized_load[donor] -= amount
                            optimized_load[receiver] += amount

                            donor_capacity[donor] -= amount
                            receiver_capacity[receiver] -= amount

                    results = pd.DataFrame({
                        "Forecast_Hour": forecast["Forecast_Hour"],
                        "Electricity_Price_INR_per_kWh": prices,
                        "Original_Load_MW": base_load,
                        "Optimized_Load_MW": optimized_load
                    })

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
                        f"Original energy: {base_load.sum():,.2f} MWh"
                    )

                    st.write(
                        f"Optimized energy: "
                        f"{optimized_load.sum():,.2f} MWh"
                    )

                    st.caption(
                        "This is a simplified scheduling demonstration. "
                        "It assumes one-hour intervals, preserves total "
                        "forecast energy, and uses the entered prices. "
                        "Savings are estimates, not guaranteed results."
                    )

                    st.subheader("Original vs Optimized Load")

                    st.line_chart(
                        results.set_index("Forecast_Hour")[
                            ["Original_Load_MW", "Optimized_Load_MW"]
                        ]
                    )

                    st.subheader("Cost Comparison by Hour")

                    st.bar_chart(
                        results.set_index("Forecast_Hour")[
                            ["Original_Cost_INR", "Optimized_Cost_INR"]
                        ]
                    )

                    st.subheader("Detailed Optimization Schedule")

                    st.dataframe(
                        results.round(3),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Optimization Results CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="energy_optimization_results.csv",
                        mime="text/csv"
                    )


# ==================================================
# 7. ADDITIONAL ANALYSIS
# ==================================================

st.header("5. Additional Analysis")

extra_files = [
    "energy_forecast_results.csv",
    "energy_24hour_forecast.csv",
    "weather_model_comparison.csv",
    "weekday_weekend_analysis.csv",
    "five_hour_deviation_forecast.csv",
    "fault_tolerance_simulation.csv"
]

for filename in extra_files:
    data = load_csv(filename)

    if data is not None:
        with st.expander(filename.replace("_", " ").title()):
            st.dataframe(
                data,
                use_container_width=True
            )


# ==================================================
# 8. 2-OUT-OF-3 ML MODEL VOTING
# ==================================================

st.header("6. 2-out-of-3 Machine Learning Model Voting")

st.write(
    "This section demonstrates consensus voting between "
    "SARIMAX, SARIMAX + SVM, and LSTM forecasts."
)

st.info(
    "Demonstration mode: enter three predictions below. "
    "These values are not automatically loaded from your "
    "trained models."
)

tolerance = st.number_input(
    "Agreement tolerance (%)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.5
)

col1, col2, col3 = st.columns(3)

with col1:
    sarimax_value = st.number_input(
        "SARIMAX prediction (MW)",
        min_value=0.0,
        value=100.0,
        key="sarimax_prediction"
    )

with col2:
    svm_value = st.number_input(
        "SARIMAX + SVM prediction (MW)",
        min_value=0.0,
        value=102.0,
        key="svm_prediction"
    )

with col3:
    lstm_value = st.number_input(
        "LSTM prediction (MW)",
        min_value=0.0,
        value=150.0,
        key="lstm_prediction"
    )

if st.button("Run 2-out-of-3 Voting", type="primary"):
    try:
        result = two_out_of_three_voting(
            sarimax_value,
            svm_value,
            lstm_value,
            tolerance
        )

        st.subheader("Voting Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Resulting Forecast",
            f'{result["forecast"]:.2f} MW'
        )

        col2.metric(
            "Models Agreeing",
            f'{len(result["agreeing_models"])}/3'
        )

        st.write("Status:", result["status"])

        if result["consensus"]:
            st.success(
                "Consensus reached between: "
                + ", ".join(result["agreeing_models"])
            )
        else:
            st.warning(
                "No two models agreed within the selected tolerance. "
                "The median is displayed only as a fallback, "
                "not as a successful consensus."
            )

    except ValueError as error:
        st.error(str(error))


# ==================================================
# END OF APPLICATION
# ==================================================







           

                

               
                
