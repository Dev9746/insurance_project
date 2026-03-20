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
st.set_page_config(page_title="Insurance AI Platform", layout="wide")

# ===============================
# YELLOW DIGIT STYLE UI
# ===============================
st.markdown("""
<style>

/* Main Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #fff200, #ffd000);
}

/* Text */
h1, h2, h3, p, label {
    color: black !important;
}

/* Buttons */
.stButton>button {
    background-color: black;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #fff8c6;
}

/* Cards spacing */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align:center;'>🚀 Insurance AI Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Get instant claim approval prediction</p>", unsafe_allow_html=True)

st.divider()

# ===============================
# INSURANCE TYPE CARDS
# ===============================
st.subheader("🛡️ Select Insurance Type")

col1, col2, col3, col4, col5 = st.columns(5)

insurance = "Car"

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

        col1, col2 = st.columns(2)

        if result == 1:
            col1.success("✅ Claim Approved")
        else:
            col1.error("❌ Claim Rejected")

        col2.metric("Approval Probability", f"{prob:.2f}")

        st.progress(float(prob))

    except:
        st.error("⚠️ Feature mismatch! Please retrain model.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("<p style='text-align:center;'>🚀 Built with Machine Learning | Digit Style UI</p>", unsafe_allow_html=True)
