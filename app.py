import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Insurance AI Platform", layout="wide")

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(to right, #141e30, #243b55);
color: white;
}
h1, h2, h3, p {
color: white;
text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("🚀 Insurance AI Platform")
st.write("Get instant insurance claim prediction")

st.divider()

# ===============================
# INSURANCE TYPE CARDS
# ===============================
st.subheader("🛡️ Choose Insurance Type")

col1, col2, col3, col4, col5 = st.columns(5)

insurance = "Car"  # default

with col1:
    if st.button("🚗 Car"):
        insurance = "Car"

with col2:
    if st.button("🏍️ Bike"):
        insurance = "Bike"

with col3:
    if st.button("🏥 Health"):
        insurance = "Health"

with col4:
    if st.button("❤️ Life"):
        insurance = "Life"

with col5:
    if st.button("✈️ Travel"):
        insurance = "Travel"

st.divider()

# ===============================
# INPUT FORM
# ===============================
st.subheader("📋 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    claim_amount = st.number_input("Claim Amount", value=20000)
    income = st.number_input("Income", value=50000)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    policy = st.selectbox("Policy Type", ["Basic", "Premium", "Gold"])
    medical = st.selectbox("Medical History", ["Good", "Average", "Poor"])

claim_history = st.slider("Previous Claims", 0, 5, 1)
fraud = st.selectbox("Fraud Flag", [0, 1])

st.divider()

# ===============================
# ENCODING
# ===============================
gender = 1 if gender == "Male" else 0
policy = {"Basic": 0, "Premium": 1, "Gold": 2}[policy]
medical = {"Good": 0, "Average": 1, "Poor": 2}[medical]

insurance_map = {
    "Car": 0,
    "Bike": 1,
    "Health": 2,
    "Life": 3,
    "Travel": 4
}

insurance_val = insurance_map[insurance]

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 View Prediction"):

    try:
        features = np.array([[age, gender, insurance_val, policy, claim_amount, income, medical, claim_history, fraud]])
        features = scaler.transform(features)

        result = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if result == 1:
            st.success(f"✅ Claim Approved (Probability: {prob:.2f})")
        else:
            st.error(f"❌ Claim Rejected (Probability: {prob:.2f})")

        st.progress(float(prob))

    except Exception as e:
        st.error("⚠️ Model mismatch error! Please retrain model.")

st.divider()

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 Built with Machine Learning | Inspired by InsurTech Platforms")
