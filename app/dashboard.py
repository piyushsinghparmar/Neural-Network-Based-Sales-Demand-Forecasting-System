
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from model_loader import load_ann_model
from prediction import predict_sales
from forecasting import build_next_day_features


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sales Demand Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_PATH / "data" / "processed" /
    "sales_processed.csv"
)

REPORT_PATH = (
    PROJECT_PATH / "reports" /
    "model_comparison.csv"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 700;
}

.subtitle {
    color: #888888;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.08);
    padding: 16px;
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    min-width: 250px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">📊 Sales Demand Forecasting System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Neural Network-Based Retail Sales Prediction and Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 4. LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required = ["date", "store", "item", "sales"]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}"
        )

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    return data


try:

    df = load_data()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


# =====================================================
# 5. LOAD ANN
# =====================================================

try:

    model, preprocessor, target_scaler, device = (
        load_ann_model()
    )

except Exception as error:

    st.error(f"ANN loading failed: {error}")
    st.stop()


# =====================================================
# 6. LOAD MODEL COMPARISON
# =====================================================

@st.cache_data
def load_comparison():

    if REPORT_PATH.exists():

        return pd.read_csv(REPORT_PATH)

    return pd.DataFrame()


results = load_comparison()


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "Historical Analysis",
        "Future Forecast",
        "Model Comparison",
        "Inventory Planning"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Neural Network-Based Sales Demand Forecasting System"
)


# =====================================================
# 8. SHARED FILTERS
# =====================================================

st.sidebar.subheader("Forecast Controls")

store_list = sorted(df["store"].unique())

selected_store = st.sidebar.selectbox(
    "Select Store",
    store_list
)

store_data = df[
    df["store"] == selected_store
].copy()

item_list = sorted(
    store_data["item"].unique()
)

selected_item = st.sidebar.selectbox(
    "Select Product",
    item_list
)

product_data = store_data[
    store_data["item"] == selected_item
].sort_values("date").copy()


# =====================================================
# 9. DASHBOARD OVERVIEW
# =====================================================

if page == "Dashboard Overview":

    st.subheader("Business Overview")

    total_sales = df["sales"].sum()

    average_sales = df["sales"].mean()

    total_stores = df["store"].nunique()

    total_products = df["item"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        "Average Daily Sales",
        f"{average_sales:.2f}"
    )

    col3.metric(
        "Total Stores",
        total_stores
    )

    col4.metric(
        "Total Products",
        total_products
    )

    st.divider()

    st.subheader("Selected Store and Product")

    col1, col2, col3 = st.columns(3)

    col1.metric("Store", selected_store)

    col2.metric("Product", selected_item)

    col3.metric(
        "Total Product Sales",
        f"{product_data['sales'].sum():,.0f}"
    )

    st.subheader("Overall Daily Sales Trend")

    daily_sales = (
        df.groupby("date")["sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        daily_sales["date"],
        daily_sales["sales"]
    )

    ax.set_title("Daily Retail Sales")

    ax.set_xlabel("Date")

    ax.set_ylabel("Units Sold")

    ax.grid(alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Top 10 Products")

    top_products = (
        df.groupby("item")["sales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    st.bar_chart(
        top_products.set_index("item")
    )


# =====================================================
# 10. HISTORICAL ANALYSIS
# =====================================================

elif page == "Historical Analysis":

    st.subheader("Historical Sales Analysis")

    st.write(
        f"Store: {selected_store} | Product: {selected_item}"
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        product_data["date"],
        product_data["sales"],
        label="Historical Sales"
    )

    ax.set_title("Historical Sales Trend")

    ax.set_xlabel("Date")

    ax.set_ylabel("Units Sold")

    ax.legend()

    ax.grid(alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    st.subheader("Monthly Sales Analysis")

    monthly_sales = (
        product_data.set_index("date")["sales"]
        .resample("ME")
        .sum()
    )

    st.bar_chart(monthly_sales)

    st.subheader("Historical Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Sales",
        f"{product_data['sales'].mean():.2f}"
    )

    col2.metric(
        "Maximum Sales",
        f"{product_data['sales'].max():.0f}"
    )

    col3.metric(
        "Minimum Sales",
        f"{product_data['sales'].min():.0f}"
    )


# =====================================================
# 11. FUTURE FORECAST
# =====================================================

elif page == "Future Forecast":

    st.subheader("🔮 Future Sales Forecasting")

    st.write(
        "Generate future sales predictions using your trained ANN model."
    )

    st.info(
        "Forecasts are generated recursively. Each predicted day "
        "is added to the history before forecasting the next day."
    )

    forecast_days = st.select_slider(
        "Select Forecast Horizon",
        options=[1, 3, 7, 14, 21, 30],
        value=7
    )

    st.write(
        f"Forecasting {forecast_days} future days for "
        f"Store {selected_store}, Product {selected_item}."
    )

    if st.button(
        "Generate Future Forecast",
        type="primary",
        use_container_width=True
    ):

        try:

            # Keep only raw historical data
            history = product_data[
                ["date", "store", "item", "sales"]
            ].copy()

            forecast_records = []

            progress = st.progress(0)

            status = st.empty()

            for day in range(forecast_days):

                # Generate features for next day
                feature_row, forecast_date = (
                    build_next_day_features(
                        history,
                        selected_store,
                        selected_item
                    )
                )

                # Predict using ANN
                predicted_sales = predict_sales(
                    feature_row,
                    model,
                    preprocessor,
                    target_scaler,
                    device
                )[0]

                predicted_sales = max(
                    float(predicted_sales),
                    0.0
                )

                # Save forecast
                forecast_records.append({
                    "date": forecast_date,
                    "store": selected_store,
                    "item": selected_item,
                    "predicted_sales": predicted_sales
                })

                # Add prediction to history
                new_row = pd.DataFrame([{
                    "date": forecast_date,
                    "store": selected_store,
                    "item": selected_item,
                    "sales": predicted_sales
                }])

                history = pd.concat(
                    [history, new_row],
                    ignore_index=True
                )

                progress.progress(
                    (day + 1) / forecast_days
                )

                status.write(
                    f"Forecasting day {day + 1} of {forecast_days}"
                )

            forecast_df = pd.DataFrame(
                forecast_records
            )

            st.session_state["forecast_df"] = forecast_df

            status.success(
                "Future forecasting completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Forecasting failed: {error}"
            )

    # Display generated forecast
    if "forecast_df" in st.session_state:

        forecast_df = st.session_state["forecast_df"]

        st.divider()

        st.subheader("Forecast Results")

        total_forecast = (
            forecast_df["predicted_sales"].sum()
        )

        average_forecast = (
            forecast_df["predicted_sales"].mean()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Forecast Demand",
            f"{total_forecast:,.2f}"
        )

        col2.metric(
            "Average Daily Demand",
            f"{average_forecast:.2f}"
        )

        col3.metric(
            "Forecast Period",
            f"{len(forecast_df)} days"
        )

        st.subheader("Predicted Sales Trend")

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(
            forecast_df["date"],
            forecast_df["predicted_sales"],
            marker="o",
            label="Predicted Sales"
        )

        ax.set_title("Future Sales Forecast")

        ax.set_xlabel("Date")

        ax.set_ylabel("Predicted Units")

        ax.legend()

        ax.grid(alpha=0.3)

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Forecast Data")

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Forecast CSV",
            data=csv_data,
            file_name="sales_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )


# =====================================================
# 12. MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.subheader("Model Performance Comparison")

    if results.empty:

        st.warning(
            "Model comparison report was not found."
        )

    else:

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        if "MAE" in results.columns:

            best_model = results.loc[
                results["MAE"].idxmin()
            ]

            st.success(
                f"Best validation model: {best_model['Model']} | "
                f"MAE: {best_model['MAE']:.4f}"
            )

            st.subheader("MAE Comparison")

            sorted_results = results.sort_values(
                "MAE"
            )

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.bar(
                sorted_results["Model"].astype(str),
                sorted_results["MAE"]
            )

            ax.set_ylabel("Mean Absolute Error")

            ax.set_title("Model MAE Comparison")

            plt.xticks(rotation=25)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        if "RMSE" in results.columns:

            st.subheader("RMSE Comparison")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.bar(
                results["Model"].astype(str),
                results["RMSE"]
            )

            ax.set_ylabel("Root Mean Squared Error")

            ax.set_title("Model RMSE Comparison")

            plt.xticks(rotation=25)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        if "SMAPE" in results.columns:

            st.subheader("SMAPE Comparison")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.bar(
                results["Model"].astype(str),
                results["SMAPE"]
            )

            ax.set_ylabel("SMAPE (%)")

            ax.set_title("SMAPE Comparison")

            plt.xticks(rotation=25)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# =====================================================
# 13. INVENTORY PLANNING
# =====================================================

elif page == "Inventory Planning":

    st.subheader("📦 Inventory Planning")

    st.write(
        "Estimate stock requirements based on predicted sales demand."
    )

    st.info(
        "Inventory calculations are planning estimates, not "
        "automatically optimized purchase orders."
    )

    forecast_df = st.session_state.get(
        "forecast_df"
    )

    if forecast_df is None:

        st.warning(
            "Generate a future forecast first using the Future Forecast section."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            current_stock = st.number_input(
                "Current Available Stock",
                min_value=0,
                value=100
            )

        with col2:

            safety_stock = st.number_input(
                "Additional Safety Stock",
                min_value=0,
                value=20
            )

        expected_demand = (
            forecast_df["predicted_sales"].sum()
        )

        required_stock = (
            expected_demand + safety_stock
        )

        recommended_order = max(
            0,
            int(np.ceil(required_stock - current_stock))
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Expected Demand",
            f"{expected_demand:.2f} units"
        )

        col2.metric(
            "Required Stock",
            f"{required_stock:.2f} units"
        )

        col3.metric(
            "Recommended Additional Stock",
            f"{recommended_order} units"
        )

        if recommended_order > 0:

            st.warning(
                f"Estimated stock shortage: "
                f"{recommended_order} units."
            )

        else:

            st.success(
                "Current stock is sufficient for the forecast "
                "period, including the specified safety stock."
            )


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "Neural Network-Based Sales Demand Forecasting System"
)

st.caption(
    "Developed using Python, PyTorch, Pandas, NumPy, "
    "Matplotlib and Streamlit."
)
