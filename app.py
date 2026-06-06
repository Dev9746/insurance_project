import streamlit as st
import numpy as np
import joblib

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Insurance Claim Approval Prediction")
st.write("Enter customer details to predict claim approval.")

# ==========================
# INPUTS
# ==========================

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

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
    max_value=500000,
    value=10000
)

income = st.number_input(
    "Income",
    min_value=10000,
    max_value=1000000,
    value=50000
)

medical = st.selectbox(
    "Medical History",
    ["Good", "Average", "Poor"]
)

claim_history = st.slider(
    "Previous Claims",
    0,
    5,
    1
)

fraud = st.selectbox(
    "Fraud Flag",
    ["No", "Yes"]
)

# ==========================
# ENCODING
# ==========================

gender = 1 if gender == "Male" else 0

policy_map = {
    "Basic": 0,
    "Gold": 1,
    "Premium": 2
}
policy = policy_map[policy]

medical_map = {
    "Average": 0,
    "Good": 1,
    "Poor": 2
}
medical = medical_map[medical]

fraud = 1 if fraud == "Yes" else 0

# ==========================
# PREDICTION
# ==========================

if st.button("🚀 Predict Claim Status"):

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

    probability = model.predict_proba(features)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.success("✅ Claim Approved")
    else:
        st.error("❌ Claim Rejected")

    st.write(f"Approval Probability: **{probability:.2%}**")
