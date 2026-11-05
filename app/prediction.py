
import numpy as np
import torch


# Categorical features
CATEGORICAL_FEATURES = [
    "store",
    "item"
]


# Numerical features
NUMERICAL_FEATURES = [
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


# Complete feature list
FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES + NUMERICAL_FEATURES
)


def predict_sales(
    row,
    model,
    preprocessor,
    target_scaler,
    device
):

    # Check required columns
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in row.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    # Select features in training order
    X = row[FEATURE_COLUMNS].copy()

    # Apply saved preprocessing
    X_processed = preprocessor.transform(
        X
    ).astype(np.float32)

    # Verify input dimensions
    if X_processed.shape[1] != model.network[0].in_features:
        raise ValueError(
            "Input feature count does not match ANN architecture."
        )

    # Convert to PyTorch tensor
    X_tensor = torch.as_tensor(
        X_processed,
        dtype=torch.float32,
        device=device
    )

    # Generate predictions
    model.eval()

    with torch.inference_mode():

        prediction_scaled = model(
            X_tensor
        ).cpu().numpy()

    # Convert to original sales scale
    prediction = target_scaler.inverse_transform(
        prediction_scaled
    ).flatten()

    # Sales cannot be negative
    prediction = np.maximum(
        prediction,
        0
    )

    return prediction
