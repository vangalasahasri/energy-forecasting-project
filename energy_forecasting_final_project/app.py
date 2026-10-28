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

    if agree(a, b) and agree(a, c) and agree(b, c):
        return {
            "forecast": float(np.mean(values)),
            "status": "All 3 models agree",
            "agreeing_models": list(predictions.keys()),
            "consensus": True
        }

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


# =====================================================
# 9. ACTUAL FORECAST CORRECTION IMPLEMENTATION
# =====================================================
# =========================================================
# 9. WORKING FORECAST CORRECTION WITH USER INPUTS
# =========================================================

st.header("7. Corrected Energy Forecast")

st.write(
    "Enter your forecast and historical error values. "
    "The application will calculate each correction and "
    "the final corrected forecast."
)

# ---------------------------------------------------------
# INPUT 1: CURRENT FORECAST VALUES
# ---------------------------------------------------------

st.subheader("1. Enter Forecast Values")

col1, col2 = st.columns(2)

with col1:
    original_forecast = st.number_input(
        "Original predicted energy consumption",
        min_value=0.0,
        value=100.0,
        step=1.0,
        key="correction_original_forecast"
    )

with col2:
    actual_consumption = st.number_input(
        "Actual energy consumption (optional, for evaluation)",
        min_value=0.0,
        value=120.0,
        step=1.0,
        key="correction_actual_consumption"
    )

# ---------------------------------------------------------
# INPUT 2: HISTORICAL FORECAST ERRORS
# ---------------------------------------------------------

st.subheader("2. Enter Previous Forecast Errors")

st.write(
    "Enter previous forecast errors separated by commas. "
    "Calculate each error as Actual Consumption minus "
    "Predicted Consumption."
)

previous_errors_text = st.text_input(
    "Previous errors",
    value="15, 10, 5",
    key="previous_forecast_errors",
    help="Example: 15, 10, 5, -2, 4"
)

weekend_errors_text = st.text_input(
    "Previous weekend errors (optional)",
    value="8, 6, 4",
    key="previous_weekend_errors",
    help="Example: 8, 6, 4"
)

is_weekend = st.checkbox(
    "Apply weekend-specific correction",
    value=False,
    key="apply_weekend_correction"
)


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def parse_errors(text_value):
    """Convert comma-separated error values into a numeric list."""

    if not text_value.strip():
        return []

    values = []

    for item in text_value.split(","):
        item = item.strip()

        if not item:
            continue

        value = float(item)

        if not np.isfinite(value):
            raise ValueError(
                "Please enter only finite numeric error values."
            )

        values.append(value)

    return values


# ---------------------------------------------------------
# CALCULATE CORRECTIONS
# ---------------------------------------------------------

try:
    previous_errors = parse_errors(previous_errors_text)
    weekend_errors = parse_errors(weekend_errors_text)

    if not previous_errors:
        st.warning(
            "Enter at least one previous forecast error "
            "to calculate deviation correction."
        )

    else:
        errors = np.asarray(previous_errors, dtype=float)

        # A. DEVIATION CORRECTION
        # Average of the previous forecast errors.
        deviation_correction = float(np.mean(errors))

        # B. TREND CORRECTION
        # Estimate whether the historical errors are increasing
        # or decreasing. At least two values are required.
        if len(errors) >= 2:
            x = np.arange(len(errors), dtype=float)

            slope, intercept = np.polyfit(x, errors, 1)

            trend_correction = float(slope)
        else:
            trend_correction = 0.0

        # C. WEEKEND CORRECTION
        # Average of previously supplied weekend errors.
        if is_weekend and weekend_errors:
            weekend_correction = float(
                np.mean(weekend_errors)
            )
        else:
            weekend_correction = 0.0

        # D. COMBINE CORRECTIONS
        total_correction = (
            deviation_correction
            + trend_correction
            + weekend_correction
        )

        corrected_forecast = max(
            0.0,
            original_forecast + total_correction
        )

        # Current actual value is used only for evaluation.
        # It is NOT used to correct its own forecast.
        current_forecast_error = (
            actual_consumption - original_forecast
        )

        original_absolute_error = abs(
            actual_consumption - original_forecast
        )

        corrected_absolute_error = abs(
            actual_consumption - corrected_forecast
        )

        if original_absolute_error > 0:
            improvement_percent = (
                (
                    original_absolute_error
                    - corrected_absolute_error
                )
                / original_absolute_error
            ) * 100
        else:
            improvement_percent = 0.0

        # -------------------------------------------------
        # DISPLAY ACTUAL CALCULATED VALUES
        # -------------------------------------------------

        st.divider()
        st.subheader("3. Calculated Correction Values")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Deviation Correction",
            f"{deviation_correction:.4f}"
        )

        col2.metric(
            "Trend Correction",
            f"{trend_correction:.4f}"
        )

        col3.metric(
            "Weekend Correction",
            f"{weekend_correction:.4f}"
        )

        st.subheader("4. Final Forecast Calculation")

        st.write(
            f"**Original forecast:** "
            f"{original_forecast:.4f}"
        )

        st.write(
            f"**Deviation correction:** "
            f"{deviation_correction:.4f}"
        )

        st.write(
            f"**Trend correction:** "
            f"{trend_correction:.4f}"
        )

        st.write(
            f"**Weekend correction:** "
            f"{weekend_correction:.4f}"
        )

        st.write(
            f"**Total correction:** "
            f"{total_correction:.4f}"
        )

        st.success(
            f"Corrected forecast = {corrected_forecast:.4f}"
        )

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------

        st.subheader("5. Forecast Error Evaluation")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Current Forecast Error",
            f"{current_forecast_error:.4f}"
        )

        col2.metric(
            "Original Absolute Error",
            f"{original_absolute_error:.4f}"
        )

        col3.metric(
            "Corrected Absolute Error",
            f"{corrected_absolute_error:.4f}"
        )

        st.metric(
            "Absolute Error Improvement (%)",
            f"{improvement_percent:.2f}%"
        )

        if corrected_absolute_error < original_absolute_error:
            st.success(
                "The corrected forecast is closer to the "
                "entered actual value for this example."
            )
        elif corrected_absolute_error > original_absolute_error:
            st.warning(
                "The correction increased the error for this example."
            )
        else:
            st.info(
                "The absolute error did not change for this example."
            )

        # -------------------------------------------------
        # F. RESULTS TABLE AND CHART
        # -------------------------------------------------

        results = pd.DataFrame({
            "Measurement": [
                "Original Forecast",
                "Deviation Correction",
                "Trend Correction",
                "Weekend Correction",
                "Total Correction",
                "Corrected Forecast",
                "Actual Consumption",
                "Current Forecast Error",
                "Original Absolute Error",
                "Corrected Absolute Error"
            ],
            "Calculated_Value": [
                original_forecast,
                deviation_correction,
                trend_correction,
                weekend_correction,
                total_correction,
                corrected_forecast,
                actual_consumption,
                current_forecast_error,
                original_absolute_error,
                corrected_absolute_error
            ]
        })

        st.subheader("Calculated Results Table")

        st.dataframe(
            results.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Forecast Comparison")

        comparison_chart = pd.DataFrame({
            "Energy Consumption": [
                original_forecast,
                corrected_forecast,
                actual_consumption
            ]
        }, index=[
            "Original Forecast",
            "Corrected Forecast",
            "Actual Consumption"
        ])

        st.bar_chart(comparison_chart)

        # -------------------------------------------------
        # G. DOWNLOAD RESULTS
        # -------------------------------------------------

        st.download_button(
            "Download Correction Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="corrected_energy_forecast.csv",
            mime="text/csv",
            key="download_manual_correction_results"
        )

except ValueError:
    st.error(
        "Invalid input. Enter numeric values separated by commas, "
        "for example: 15, 10, 5."
    )

# =========================================================
# END OF FORECAST CORRECTION SECTION
# =========================================================





