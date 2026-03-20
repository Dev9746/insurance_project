import streamlit as st
import numpy as np
import joblib

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Insurance Dashboard", layout="wide")

# ===============================
# CUSTOM STYLE
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(to right, #1e3c72, #2a5298);
color: white;
}
h1, h2, h3 {
color: white;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("💼 Insurance Claim Prediction Dashboard")
st.write("🔍 Predict insurance claim approval using Machine Learning")

st.divider()

# ===============================
# KPI CARDS
# ===============================
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Users", "5000")
col2.metric("📊 Model Accuracy", "90%+")
col3.metric("⚡ System Status", "Active")

st.divider()

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("📝 Enter User Details")

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
# PREDICTION SECTION
# ===============================
st.subheader("🔎 Prediction Result")

if st.button("🚀 Predict Claim Status"):
    result = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    col1, col2 = st.columns(2)

    if result == 1:
        col1.success("✅ Claim Approved")
    else:
        col1.error("❌ Claim Rejected")

    col2.metric("Approval Probability", f"{prob:.2f}")

    st.progress(float(prob))

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 Built with Machine Learning | Professional Project | Ready for LinkedIn")
