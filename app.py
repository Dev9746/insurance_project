import streamlit as st
import numpy as np
import joblib
import pandas as pd
import requests

# ===============================
# LOGIN SYSTEM
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login System")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login Successful")
        else:
            st.error("❌ Invalid Credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Insurance System", layout="wide")

# ===============================
# HEADER
# ===============================
st.title("🤖 AI Insurance Claim System")

# ===============================
# INPUT FORM
# ===============================
st.subheader("Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    claim_amount = st.number_input("Claim Amount", value=20000)
    income = st.number_input("Income", value=50000)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    policy = st.selectbox("Policy Type", ["Basic", "Premium", "Gold"])
    medical = st.selectbox("Medical History", ["Good", "Average", "Poor"])

insurance = st.selectbox("Insurance Type", ["Car", "Health", "Life"])
claim_history = st.slider("Previous Claims", 0, 5, 1)
fraud = st.selectbox("Fraud Flag", [0,1])

# ===============================
# ENCODING
# ===============================
gender = 1 if gender == "Male" else 0
policy = {"Basic":0,"Premium":1,"Gold":2}[policy]
medical = {"Good":0,"Average":1,"Poor":2}[medical]

features = np.array([[age, gender, policy, claim_amount, income, medical, claim_history, fraud]])
features_scaled = scaler.transform(features)

# ===============================
# HISTORY STORAGE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# PREDICTION
# ===============================
st.subheader("Prediction")

if st.button("🚀 Predict"):

    result = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    # ===============================
    # INSURANCE TYPE LOGIC
    # ===============================
    if insurance == "Car":
        prob *= 0.95
    elif insurance == "Health":
        prob *= 1.05
    elif insurance == "Life":
        prob *= 1.1

    # ===============================
    # FRAUD DETECTION
    # ===============================
    fraud_risk = "Low"
    if claim_amount > income * 0.8 or fraud == 1:
        fraud_risk = "High"
        st.warning("⚠️ High Fraud Risk Detected!")

    # ===============================
    # RESULT
    # ===============================
    if prob > 0.5:
        st.success("✅ Claim Approved")
    else:
        st.error("❌ Claim Rejected")

    st.metric("Approval Probability", f"{prob*100:.1f}%")

    # ===============================
    # AI EXPLANATION (LITE)
    # ===============================
    st.subheader("🧠 AI Explanation")

    reasons = []

    if income > 50000:
        reasons.append("High income increases approval chance")
    if claim_amount < 30000:
        reasons.append("Lower claim amount is safer")
    if fraud == 0:
        reasons.append("No fraud detected")

    for r in reasons:
        st.write("✔️", r)

    # ===============================
    # SAVE HISTORY
    # ===============================
    st.session_state.history.append({
        "Insurance": insurance,
        "Amount": claim_amount,
        "Result": "Approved" if prob > 0.5 else "Rejected",
        "Fraud Risk": fraud_risk
    })

    # ===============================
    # EMAIL (RESEND)
    # ===============================
    try:
        api_key = st.secrets["RESEND_API_KEY"]

        url = "https://api.resend.com/emails"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "from": "onboarding@resend.dev",
            "to": ["devs72527@gmail.com"],
            "subject": "Insurance Prediction Result",
            "html": f"""
            <h3>Prediction Result</h3>
            <p>Status: {'Approved' if prob>0.5 else 'Rejected'}</p>
            <p>Probability: {prob:.2f}</p>
            <p>Fraud Risk: {fraud_risk}</p>
            """
        }

        requests.post(url, headers=headers, json=data)

    except:
        pass

# ===============================
# HISTORY DASHBOARD
# ===============================
st.subheader("📊 Prediction History")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    st.bar_chart(df["Result"].value_counts())

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 Ultimate AI Insurance System")
