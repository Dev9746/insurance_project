import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Insurance Prediction", layout="wide")

st.title("🏥 Insurance Claim Prediction System")

st.sidebar.header("Input Details")

age = st.sidebar.slider("Age", 18, 70)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
policy = st.sidebar.selectbox("Policy Type", ["Basic", "Premium", "Gold"])
claim_amount = st.sidebar.number_input("Claim Amount")
income = st.sidebar.number_input("Income")
medical = st.sidebar.selectbox("Medical History", ["Good", "Average", "Poor"])
claim_history = st.sidebar.slider("Claim History", 0, 5)
fraud = st.sidebar.selectbox("Fraud Flag", [0,1])

# Encoding manually (same as training)
gender = 1 if gender == "Male" else 0
policy = {"Basic":0,"Premium":1,"Gold":2}[policy]
medical = {"Good":0,"Average":1,"Poor":2}[medical]

features = np.array([[age, gender, policy, claim_amount, income, medical, claim_history, fraud]])
features = scaler.transform(features)

if st.button("Predict"):
    result = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if result == 1:
        st.success(f"✅ Claim Approved (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Claim Rejected (Probability: {prob:.2f})")
