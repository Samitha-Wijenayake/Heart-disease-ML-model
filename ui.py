import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

# Load model
class HeartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

model = HeartModel()
model.load_state_dict(torch.load("heart_model.pth"))
model.eval()

scaler = joblib.load("scaler.pkl")

st.markdown("## Heart Disease Model\n### (Educational Only)")

st.write("⚠️ This tool is for learning ML. It does **not** give medical advice.")

# User Input Form
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex (1 = male, 0 = female)", [0, 1])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1/0)", [0, 1])
restecg = st.number_input("Resting ECG (0–2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1/0)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0–2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0–4)", 0, 4, 0)
thal = st.number_input("Thal (0,1,2,3)", 0, 3, 2)

if st.button("Ask Model"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

    data = scaler.transform(data)
    data = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        output = model(data)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 1:
        st.error("Model says: Higher chance of heart disease (Educational Only)")
    else:
        st.success("Model says: Lower chance of heart disease (Educational Only)")
