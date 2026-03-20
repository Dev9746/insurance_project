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
st.set_page_config(page_title="AI Insurance Claim", layout="wide")

# ===============================
# STYLE (WHITE + BLACK TEXT)
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f5f5;
}

h1, h2, h3, p, label {
    color: black !important;
}

.stButton>button {
    background-color: #ffcc00;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
col1, col2 = st.columns([6,1])

with col1:
    st.markdown("## 🤖 AI Based Insurance Claim")

with col2:
    st.button("Login")

st.divider()

# ===============================
# HERO SECTION (TEXT + IMAGE)
# ===============================
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Smart Insurance Prediction System")
    st.write("AI powered system to predict claim approval instantly")
    st.write("Trusted by intelligent ML models")

with col2:
    st.image("https://images.unsplash.com/photo-1605902711622-cfb43c44367f")

st.divider()

# ===============================
# INSURANCE TYPE
# ===============================
st.subheader("Select Insurance Type")

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
    if st.button("🏢 Commercial"):
        insurance = "Commercial"

with col5:
    if st.button("✈️ Travel"):
        insurance = "Travel"

st.divider()

# ===============================
# FORM
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

claim_history = st.slider("Previous Claims", 0, 5, 1)
fraud = st.selectbox("Fraud Flag", [0,1])

st.divider()

# ===============================
# ENCODING
# ===============================
gender = 1 if gender == "Male" else 0
policy = {"Basic":0,"Premium":1,"Gold":2}[policy]
medical = {"Good":0,"Average":1,"Poor":2}[medical]

insurance_map = {
    "Car": 0,
    "Bike": 1,
    "Health": 2,
    "Life": 3,
    "Travel": 4,
    "Commercial": 0
}

insurance_val = insurance_map.get(insurance, 0)

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
        st.error("⚠️ Model mismatch! Retrain model.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 AI Based Insurance Claim System | Final Year Project")
