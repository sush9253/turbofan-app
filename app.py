import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model
model = joblib.load('engine_failure_model.pkl')

# Load feature names
with open('feature_names.json') as f:
    feature_names = json.load(f)

st.title("✈️ Turbofan Engine Failure Prediction")

st.write("Enter sensor readings to predict failure within 30 cycles.")

st.sidebar.header("Sensor Inputs")

sensor_list = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_14',
    'sensor_17', 'sensor_20', 'sensor_21'
]

# Inputs
inputs = {}
for s in sensor_list:
    inputs[s] = st.sidebar.number_input(s, value=0.0, format="%.4f")

# Predict button
if st.button("Predict"):

    fd = {}

    for s in sensor_list:
        val = inputs[s]
        fd[s] = val

        for w in [5, 10, 20]:
            fd[f"{s}_rollmean_{w}"] = val
            fd[f"{s}_rollstd_{w}"] = 0.0

        fd[f"{s}_roc"] = 0.0

    df = pd.DataFrame([fd])
    df = df.reindex(columns=feature_names, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    THR = 0.35

    st.subheader("Prediction Result")

    if prob >= 0.7:
        st.error(f"CRITICAL ⚠️ | Failure Risk: {prob:.2%}")
    elif prob >= THR:
        st.warning(f"WARNING ⚠️ | Failure Risk: {prob:.2%}")
    else:
        st.success(f"SAFE ✅ | Failure Risk: {prob:.2%}")

    st.progress(min(prob, 1.0))