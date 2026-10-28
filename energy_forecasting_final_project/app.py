import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 1. PAGE SETUP
# =====================================================

st.set_page_config(
    page_title="Cloud Energy Forecasting System",
    page_icon="⚡",
    layout="wide"
)

DATA_DIR = Path(__file__).parent

st.title("⚡ Cloud-Based Energy Forecasting and Optimization")
st.write(
    "Energy Consumption Forecasting Using Machine Learning"
)

st.info(
    "This application displays forecasting results, "
    "compares models, demonstrates 2-out-of-3 voting, "
    "corrects forecast deviations, and estimates energy savings."
)


# =====================================================
# 2. LOAD DATA FUNCTION
# =====================================================

def load_csv(filename):
    file_path = DATA_DIR / filename

    if file_path.exists():
        try:
            return pd.read_csv(file_path)
        except Exception as error:
            st.error(f"Error reading {filename}: {error}")

    return None


# =====================================================
# 3. MACHINE LEARNING MODEL COMPARISON
# =====================================================

st.header("1. Machine Learning Model Comparison")

comparison = load_csv("three_model_comparison.csv")

if comparison is not None:
    st.dataframe(comparison, use_container_width=True)
else:
    st.warning("three_model_comparison.csv was not found.")


# =====================================================
# 4. MODEL EVALUATION METRICS
# =====================================================

st.header("2. Model Evaluation Metrics")

metrics = load_csv("energy_model_metrics.csv")

if metrics is not None:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("energy_model_metrics.csv was not found.")


# =====================================================
# 5. ENERGY FORECAST GRAPHS
# =====================================================

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


# =====================================================
# 6. 2-OUT-OF-3 VOTING FUNCTION
# =====================================================

def two_out_of_three_voting(
    sarimax,
    sarimax_svm,
    lstm,
    tolerance=5.0
):
    predictions = {
        "SARIMAX": float(sarimax),
        "SARIMAX + SVM": float(sarimax_svm),
        "LSTM": float(lstm)
    }

    values = list(predictions.values())

    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError(
            "Predictions must be finite and non-negative."
        )

    def agree(a, b):
        denominator = (abs(a) + abs(b)) / 2

        if denominator == 0:
            return True

        difference = abs(a - b) / denominator * 100

        return difference <= tolerance

    a, b, c = values

    if agree(a, b) and agree(a, c) and agree(b, c):
        return {
            "forecast": np.mean(values),
            "status": "All 3 models agree",
            "models": list(predictions.keys())
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
            "forecast": (x + y) / 2,
            "status": "2-out-of-3 consensus",
            "models": [model1, model2]
        }

    return {
        "forecast": float(np.median(values)),
        "status": "No consensus - median fallback",
        "models": []
    }


# =====================================================
# 7. INTERACTIVE 2-OUT-OF-3 VOTING
# =====================================================

st.header("4. 2-out-of-3 Fault-Tolerance Voting")

st.write(
    "Enter the predictions produced by the three forecasting "
    "models. The application checks whether at least two "
    "predictions agree within the selected tolerance."
)

col1, col2, col3 = st.columns(3)

with col1:
    sarimax_value = st.number_input(
        "SARIMAX prediction (MW)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

with col2:
    svm_value = st.number_input(
        "SARIMAX + SVM prediction (MW)",
        min_value=0.0,
        value=102.0,
        step=1.0
    )

with col3:
    lstm_value = st.number_input(
        "LSTM prediction (MW)",
        min_value=0.0,
        value=101.0,
        step=1.0
    )

tolerance = st.slider(
    "Model agreement tolerance (%)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5
)

if st.button("Run 2-out-of-3 Voting"):
    result = two_out_of_three_voting(
        sarimax_value,
        svm_value,
        lstm_value,
        tolerance
    )

    st.subheader("Voting Result")

    st.metric(
        "Final Voting Forecast (MW)",
        f"{result['forecast']:.2f}"
    )

    st.write("Status:", result["status"])

    if result["models"]:
        st.success(
            "Agreeing models: " + ", ".join(result["models"])
        )
    else:
        st.warning(
            "No two models agreed. The median is shown as "
            "a fallback, not as a successful consensus."
        )


# =====================================================
# 8. ENERGY COST OPTIMIZATION
# =====================================================

st.header("5. Energy Cost Optimization")

forecast = load_csv("energy_24hour_forecast.csv")

if forecast is None:
    st.warning(
        "Place energy_24hour_forecast.csv in the same folder "
        "as app.py to enable energy cost optimization."
    )

else:
    required = ["Forecast_Hour", "Forecast_Load_MW"]

    if all(column in forecast.columns for column in required):

        forecast = forecast[required].copy()

        forecast["Forecast_Hour"] = pd.to_datetime(
            forecast["Forecast_Hour"],
            errors="coerce"
        )

        forecast["Forecast_Load_MW"] = pd.to_numeric(
            forecast["Forecast_Load_MW"],
            errors="coerce"
        )

        forecast = forecast.dropna()

        forecast = forecast[
            forecast["Forecast_Load_MW"] >= 0
        ].copy()

        forecast["Hour"] = forecast["Forecast_Hour"].dt.hour

        def default_price(hour):
            if hour in [0, 1, 2, 3, 4, 5]:
                return 5.0
            elif hour in [18, 19, 20, 21]:
                return 12.0
            return 8.0

        forecast["Default_Price"] = forecast["Hour"].apply(
            default_price
        )

        st.write(
            "Prices below are illustrative. Replace them with "
            "your actual electricity tariff."
        )

        forecast["Price_INR_per_kWh"] = forecast["Default_Price"]

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
            disabled=["Forecast_Hour", "Forecast_Load_MW"]
        )

        base_load = edited["Forecast_Load_MW"].to_numpy(
            dtype=float
        )

        prices = edited["Price_INR_per_kWh"].to_numpy(
            dtype=float
        )

        flexible_percent = st.slider(
            "Flexible load available to shift (%)",
            min_value=0,
            max_value=50,
            value=10
        )

        receiving_percent = st.slider(
            "Receiving-hour capacity (%)",
            min_value=0,
            max_value=50,
            value=10
        )

        if st.button("Optimize Energy Consumption"):

            optimized_load = base_load.copy()

            donor_capacity = (
                base_load * flexible_percent / 100
            )

            receiver_capacity = (
                base_load * receiving_percent / 100
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
                "Forecast_Hour": edited["Forecast_Hour"],
                "Price_INR_per_kWh": prices,
                "Original_Load_MW": base_load,
                "Optimized_Load_MW": optimized_load
            })

            results["Original_Cost_INR"] = (
                results["Original_Load_MW"]
                * 1000
                * results["Price_INR_per_kWh"]
            )

            results["Optimized_Cost_INR"] = (
                results["Optimized_Load_MW"]
                * 1000
                * results["Price_INR_per_kWh"]
            )

            original_cost = results["Original_Cost_INR"].sum()
            optimized_cost = results["Optimized_Cost_INR"].sum()

            savings = original_cost - optimized_cost

            savings_percent = (
                savings / original_cost * 100
                if original_cost > 0 else 0
            )

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

            st.subheader("Original vs Optimized Energy Load")

            st.line_chart(
                results.set_index("Forecast_Hour")[
                    ["Original_Load_MW", "Optimized_Load_MW"]
                ]
            )

            st.subheader("Cost Comparison")

            st.bar_chart(
                results.set_index("Forecast_Hour")[
                    ["Original_Cost_INR", "Optimized_Cost_INR"]
                ]
            )

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download Optimization Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="energy_optimization_results.csv",
                mime="text/csv"
            )

            st.caption(
                "This is a simplified load-shifting simulation. "
                "Savings are estimates and depend on the tariff "
                "and the assumptions used."
            )

    else:
        st.error(
            "The forecast CSV must contain Forecast_Hour "
            "and Forecast_Load_MW columns."
        )


# =====================================================
# 9. TREND, DEVIATION AND HOLIDAY CORRECTION
# =====================================================

st.header("6. Trend, Deviation and Holiday Correction")

st.write(
    "This module demonstrates forecast correction using "
    "historical prediction errors, a fitted error trend, "
    "and an optional weekend or holiday adjustment."
)

correction_data = load_csv("energy_forecast_results.csv")

if correction_data is None:
    correction_data = load_csv("energy_24hour_forecast.csv")

if correction_data is None:
    st.warning(
        "No forecast CSV was found. Add energy_forecast_results.csv "
        "or energy_24hour_forecast.csv to enable correction."
    )

else:
    st.write("Available columns:", list(correction_data.columns))

    date_options = [
        column for column in correction_data.columns
        if any(word in column.lower()
               for word in ["date", "time", "hour", "timestamp"])
    ]

    prediction_options = [
        column for column in correction_data.columns
        if any(word in column.lower()
               for word in ["forecast", "predict", "predicted"])
    ]

    actual_options = [
        column for column in correction_data.columns
        if any(word in column.lower()
               for word in ["actual", "observed", "real"])
    ]

    if not prediction_options:
        st.warning(
            "No prediction column was identified. "
            "Check the CSV column names."
        )

    else:
        prediction_column = st.selectbox(
            "Select forecast/prediction column",
            prediction_options
        )

        actual_column = None

        if actual_options:
            actual_column = st.selectbox(
                "Select actual consumption column (if available)",
                ["None"] + actual_options
            )

            if actual_column == "None":
                actual_column = None

        date_column = None

        if date_options:
            date_column = st.selectbox(
                "Select date/time column (if available)",
                ["None"] + date_options
            )

            if date_column == "None":
                date_column = None

        weekend_factor = st.number_input(
            "Weekend/holiday multiplier",
            min_value=0.0,
            max_value=2.0,
            value=0.937,
            step=0.01,
            help=(
                "1.0 means no adjustment. 0.937 reduces "
                "the forecast by approximately 6.3%."
            )
        )

        holiday_dates_text = st.text_input(
            "Holiday dates (optional, comma-separated YYYY-MM-DD)",
            placeholder="2026-12-25, 2027-01-01"
        )

        corrected = correction_data.copy()

        corrected[prediction_column] = pd.to_numeric(
            corrected[prediction_column],
            errors="coerce"
        )

        valid_forecasts = corrected[prediction_column].dropna()

        if valid_forecasts.empty:
            st.error("The selected prediction column has no valid numbers.")

        else:
            # Initial forecast values
            corrected["Original_Forecast"] = (
                corrected[prediction_column]
            )

            # Default corrections
            corrected["Deviation_Correction"] = 0.0
            corrected["Trend_Correction"] = 0.0

            # Historical error correction requires actual values.
            if actual_column is not None:
                corrected[actual_column] = pd.to_numeric(
                    corrected[actual_column],
                    errors="coerce"
                )

                corrected["Historical_Error"] = (
                    corrected[actual_column]
                    - corrected[prediction_column]
                )

                errors = corrected["Historical_Error"].dropna()

                if len(errors) > 0:
                    recent_errors = errors.tail(24)
                    average_error = recent_errors.mean()

                    corrected["Deviation_Correction"] = (
                        average_error
                    )

                    st.metric(
                        "Recent Average Forecast Error",
                        f"{average_error:.3f}"
                    )

                    # Fit a simple linear trend to available errors.
                    if len(errors) >= 2:
                        error_values = errors.to_numpy(dtype=float)
                        x_values = np.arange(len(error_values))

                        slope, intercept = np.polyfit(
                            x_values,
                            error_values,
                            1
                        )

                        trend_x = np.arange(
                            len(corrected),
                            len(corrected) + len(corrected)
                        )

                        trend_values = (
                            slope * trend_x + intercept
                        )

                        corrected["Trend_Correction"] = trend_values

                        st.write(
                            "Estimated error trend per row:",
                            round(float(slope), 6)
                        )

                else:
                    st.info(
                        "No valid actual/predicted pairs were found "
                        "for calculating historical errors."
                    )

            # Apply error correction to the forecast.
            corrected["Corrected_Forecast"] = (
                corrected["Original_Forecast"]
                + corrected["Deviation_Correction"]
                + corrected["Trend_Correction"]
            )

            # Optional weekend and holiday adjustment.
            corrected["Holiday_or_Weekend"] = False

            if date_column is not None:
                corrected[date_column] = pd.to_datetime(
                    corrected[date_column],
                    errors="coerce"
                )

                valid_dates = corrected[date_column].notna()

                corrected.loc[
                    valid_dates,
                    "Holiday_or_Weekend"
                ] = (
                    corrected.loc[valid_dates, date_column].dt.dayofweek
                    >= 5
                ).to_numpy()

                if holiday_dates_text.strip():
                    try:
                        holiday_dates = [
                            pd.to_datetime(item.strip()).date()
                            for item in holiday_dates_text.split(",")
                            if item.strip()
                        ]

                        holiday_mask = (
                            corrected[date_column].dt.date.isin(
                                holiday_dates
                            )
                        )

                        corrected.loc[
                            holiday_mask,
                            "Holiday_or_Weekend"
                        ] = True

                    except Exception:
                        st.warning(
                            "Some holiday dates could not be parsed. "
                            "Use YYYY-MM-DD format."
                        )

                corrected["Corrected_Forecast"] = np.where(
                    corrected["Holiday_or_Weekend"],
                    corrected["Corrected_Forecast"] * weekend_factor,
                    corrected["Corrected_Forecast"]
                )

            st.subheader("Corrected Forecast Results")

            display_columns = [
                column for column in [
                    date_column,
                    "Original_Forecast",
                    "Historical_Error",
                    "Deviation_Correction",
                    "Trend_Correction",
                    "Holiday_or_Weekend",
                    "Corrected_Forecast"
                ]
                if column is not None and column in corrected.columns
            ]

            st.dataframe(
                corrected[display_columns],
                use_container_width=True
            )

            st.subheader("Original vs Corrected Forecast")

            chart_columns = corrected[
                ["Original_Forecast", "Corrected_Forecast"]
            ].copy()

            if date_column is not None:
                chart_columns.index = corrected[date_column]

            st.line_chart(chart_columns)

            st.download_button(
                "Download Corrected Forecast CSV",
                data=corrected.to_csv(index=False).encode("utf-8"),
                file_name="corrected_energy_forecast.csv",
                mime="text/csv"
            )

            st.warning(
                "Important: this is a basic correction demonstration. "
                "Validate correction settings against held-out actual "
                "data before claiming improved accuracy. The default "
                "weekend multiplier is illustrative, not a learned "
                "or scientifically validated value."
            )


# =====================================================
# 10. ADDITIONAL ANALYSIS FILES
# =====================================================

st.header("7. Additional Analysis")

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
            st.dataframe(data, use_container_width=True)


# =====================================================
# 11. END OF APPLICATION
# =====================================================

st.success("Energy Forecasting and Optimization application loaded.")

st.caption(
    "Note: this application displays saved model results and "
    "demonstrates voting, correction, and optimization. It does "
    "not automatically train the models or connect to multiple "
    "cloud providers unless those integrations are implemented separately."
)
      
        
