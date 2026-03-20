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
st.set_page_config(page_title="Insurance AI Dashboard", layout="wide") 

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(to right, #141e30, #243b55); 
color: white;
}
h1, h2, h3 {
color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("🚀 Insurance Claim AI Dashboard")
st.write("Advanced ML System for Claim Approval Prediction")

st.divider()

# ===============================
# KPI CARDS
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Users", "5000+")
col2.metric("📊 Accuracy", "90%+")
col3.metric("⚡ Status", "Active")
col4.metric("🤖 Model", "Random Forest")

st.divider()

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("📝 User Details")

age = st.sidebar.slider("Age", 18, 70, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
insurance = st.sidebar.selectbox("Insurance Type", ["Car", "Bike", "Health", "Life", "Travel"])
policy = st.sidebar.selectbox("Policy Type", ["Basic", "Premium", "Gold"])
claim_amount = st.sidebar.number_input("Claim Amount", value=20000)
income = st.sidebar.number_input("Income", value=50000)
medical = st.sidebar.selectbox("Medical History", ["Good", "Average", "Poor"])
claim_history = st.sidebar.slider("Previous Claims", 0, 5, 1)
fraud = st.sidebar.selectbox("Fraud Flag", [0, 1])

# ===============================
# ENCODING
# ===============================
gender = 1 if gender == "Male" else 0
policy = {"Basic":0,"Premium":1,"Gold":2}[policy]
medical = {"Good":0,"Average":1,"Poor":2}[medical]
insurance = {"Car":0,"Bike":1,"Health":2,"Life":3,"Travel":4}[insurance]

features = np.array([[age, gender, insurance, policy, claim_amount, income, medical, claim_history, fraud]])
features = scaler.transform(features)

# ===============================
# SESSION STATE (history)
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# PREDICTION
# ===============================
st.subheader("🔎 Prediction")

if st.button("🚀 Predict Now"):
    result = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    status = "Approved" if result == 1 else "Rejected"

    # Save history
    st.session_state.history.append({
        "Age": age,
        "Insurance": insurance,
        "Amount": claim_amount,
        "Result": status,
        "Probability": round(prob, 2)
    })

    col1, col2 = st.columns(2)

    if result == 1:
        col1.success("✅ Claim Approved")
    else:
        col1.error("❌ Claim Rejected")

    col2.metric("Approval Probability", f"{prob:.2f}")

    st.progress(float(prob))

st.divider()

# ===============================
# ANALYTICS DASHBOARD
# ===============================
st.subheader("📊 Analytics Dashboard")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Prediction Distribution")
        st.bar_chart(df["Result"].value_counts())

    with col2:
        st.write("### Insurance Type Analysis")
        st.bar_chart(df["Insurance"].value_counts())

st.divider()

# ===============================
# HISTORY TABLE
# ===============================
st.subheader("📋 Prediction History")

if len(st.session_state.history) > 0:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.write("No predictions yet")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 Ultimate ML Project | Built for LinkedIn Portfolio")
