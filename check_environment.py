
import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import sklearn
import torch
import streamlit

print("=" * 40)
print("SALES FORECASTING PROJECT")
print("=" * 40)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("\nAll libraries imported successfully!")
