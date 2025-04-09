import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn

# ----- Load model -----
class BridgeANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load preprocessing pipeline
with open("preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Dummy input to initialize model with correct input size
sample_input = np.zeros((1, len(preprocessor.get_feature_names_out())))
model = BridgeANN(input_size=sample_input.shape[1])
model.load_state_dict(torch.load("pytorch_bridge_model.pt"))
model.eval()

# ----- Streamlit App UI -----
st.title("🚧 Bridge Load Capacity Predictor")
st.write("Enter bridge attributes below to predict its maximum load capacity (in tons).")

# User input
span = st.number_input("Span (ft)", min_value=0.0, max_value=2000.0, value=250.0)
deck = st.number_input("Deck Width (ft)", min_value=0.0, max_value=500.0, value=40.0)
age = st.number_input("Age (years)", min_value=0.0, max_value=150.0, value=20.0)
lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, value=2)
condition = st.selectbox("Condition Rating", [1, 2, 3, 4, 5])
material = st.selectbox("Material", ["Steel", "Concrete", "Composite"])

# Prepare input DataFrame
input_df = pd.DataFrame([[span, deck, age, lanes, material, condition]],
                        columns=['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Material', 'Condition_Rating'])

# Predict button
if st.button("Predict Max Load"):
    input_processed = preprocessor.transform(input_df)
    input_tensor = torch.tensor(input_processed, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    st.success(f"Predicted Maximum Load: **{prediction:.2f} tons**")
