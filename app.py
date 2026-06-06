import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="🏥",
    layout="centered"
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏥 Insurance Claim Approval Prediction")

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

policy = st.selectbox(
    "Policy Type",
    ["Basic", "Premium", "Gold"]
)

claim_amount = st.number_input(
    "Claim Amount",
    min_value=1000,
    value=10000
)

income = st.number_input(
    "Income",
    min_value=10000,
    value=50000
)

medical = st.selectbox(
    "Medical History",
    ["Good", "Average", "Poor"]
)

claim_history = st.slider(
    "Claim History",
    0,
    5,
    1
)

fraud = st.selectbox(
    "Fraud Flag",
    ["No", "Yes"]
)

gender = 1 if gender == "Male" else 0

policy_map = {
    "Basic": 0,
    "Gold": 1,
    "Premium": 2
}

medical_map = {
    "Average": 0,
    "Good": 1,
    "Poor": 2
}

policy = policy_map[policy]
medical = medical_map[medical]
fraud = 1 if fraud == "Yes" else 0

if st.button("Predict"):

    features = np.array([[
        age,
        gender,
        policy,
        claim_amount,
        income,
        medical,
        claim_history,
        fraud
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Claim Approved")
    else:
        st.error("❌ Claim Rejected")
