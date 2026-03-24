# Turbofan Engine Failure Prediction

## Problem
Predict whether an aircraft engine will fail within the next 30 cycles using NASA C-MAPSS sensor data.

---

## Approach
- Removed low-variance sensors and operational settings
- Created rolling features (mean, std, rate of change) with windows of 5, 10, 20
- Compared Logistic Regression, Random Forest, and Gradient Boosting
- Handled class imbalance using class_weight
- Applied threshold tuning to improve recall

---

## Results

Final Model: Gradient Boosting + threshold (0.35)

- Validation Recall: 0.927
- Test Recall: 0.663
- Test Precision: 0.794

Top Features:
- sensor_4_rollmean_5 (LPT temperature)
- sensor_11_rollmean_5 (core speed)

---

## Demo

Streamlit App:
👉 https://turbofan-app-project.streamlit.app/

---

## Notebook

👉 https://colab.research.google.com/drive/1uOawzrnX0CGpSUITYfEsAJDml0rQ1vsd?usp=sharing
---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
