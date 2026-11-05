
import torch
import torch.nn as nn
import joblib

from pathlib import Path
import streamlit as st


# ANN Architecture
class SalesANN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Load trained ANN model
@st.cache_resource
def load_ann_model():

    project_path = Path(__file__).resolve().parent.parent

    models_path = project_path / "models"

    # Required model files
    model_file = models_path / "ann_model.pth"
    preprocessor_file = models_path / "ann_preprocessor.joblib"
    scaler_file = models_path / "ann_target_scaler.joblib"

    for file in [model_file, preprocessor_file, scaler_file]:

        if not file.exists():
            raise FileNotFoundError(
                f"Required model file not found: {file}"
            )

    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load checkpoint
    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=True
    )

    # Recreate architecture
    model = SalesANN(
        checkpoint["input_size"]
    ).to(device)

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Load preprocessing objects
    preprocessor = joblib.load(preprocessor_file)

    target_scaler = joblib.load(scaler_file)

    return model, preprocessor, target_scaler, device
