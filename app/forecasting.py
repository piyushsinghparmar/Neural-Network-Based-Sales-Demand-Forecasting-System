
import numpy as np
import pandas as pd


def build_next_day_features(history, store_id, item_id):

    """
    Generate next-day forecasting features
    for a selected store and product.
    """

    data = history.copy()

    data["date"] = pd.to_datetime(data["date"])

    # Filter selected store and item
    data = data[
        (data["store"] == store_id) &
        (data["item"] == item_id)
    ].copy()

    data = data.sort_values("date").reset_index(drop=True)

    # Minimum history requirement
    if len(data) < 28:
        raise ValueError(
            "At least 28 days of history are required."
        )

    # Check duplicate dates
    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    # Check missing dates
    expected_dates = pd.date_range(
        start=data["date"].iloc[0],
        end=data["date"].iloc[-1],
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(data["date"])

    if not actual_dates.equals(expected_dates):
        raise ValueError(
            "Missing or irregular dates found in sales history."
        )

    # Validate sales
    sales = pd.to_numeric(
        data["sales"],
        errors="coerce"
    )

    if sales.isna().any() or not np.isfinite(sales).all():
        raise ValueError("Invalid sales values found.")

    if (sales < 0).any():
        raise ValueError("Negative sales values found.")

    # Next forecast date
    forecast_date = (
        data["date"].iloc[-1] + pd.Timedelta(days=1)
    )

    features = {}

    # Lag features
    for lag in [1, 7, 14, 28]:

        features[f"lag_{lag}"] = sales.iloc[-lag]

    # Rolling means
    for window in [7, 14, 28]:

        features[f"rolling_mean_{window}"] = (
            sales.iloc[-window:].mean()
        )

    # Rolling standard deviations
    for window in [7, 28]:

        features[f"rolling_std_{window}"] = (
            sales.iloc[-window:].std()
        )

    # Calendar features
    features["year"] = forecast_date.year
    features["month"] = forecast_date.month
    features["day"] = forecast_date.day

    features["day_of_week"] = (
        forecast_date.dayofweek
    )

    features["week_of_year"] = (
        forecast_date.isocalendar().week
    )

    features["quarter"] = forecast_date.quarter

    features["is_weekend"] = int(
        forecast_date.dayofweek >= 5
    )

    # Cyclical features
    features["month_sin"] = np.sin(
        2 * np.pi * features["month"] / 12
    )

    features["month_cos"] = np.cos(
        2 * np.pi * features["month"] / 12
    )

    features["day_sin"] = np.sin(
        2 * np.pi * features["day_of_week"] / 7
    )

    features["day_cos"] = np.cos(
        2 * np.pi * features["day_of_week"] / 7
    )

    # Categorical features
    features["store"] = store_id
    features["item"] = item_id

    # Create feature DataFrame
    feature_df = pd.DataFrame([features])

    # Exact training feature order
    feature_columns = [
        "store",
        "item",
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28"
    ]

    feature_df = feature_df[feature_columns]

    return feature_df, forecast_date
