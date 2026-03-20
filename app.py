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
st.set_page_config(page_title="Digit Insurance AI", layout="wide")

# ===============================
# FINAL CLEAN UI (BLACK TEXT + WHITE BG)
# ===============================
st.markdown("""
<style>

/* Main Background */
[data-testid="stAppViewContainer"] {
    background-color: #f5f5f5;
}

/* ALL TEXT BLACK */
h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: black !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    color: black;
}

/* Buttons */
.stButton>button {
    background-color: #ffcc00;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}

/* Inputs */
input, textarea {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
col1, col2 = st.columns([6,1])

with col1:
    st.markdown("## digit insurance")

with col2:
    st.button("Login")

st.divider()

# ===============================
# HERO SECTION
# ===============================
st.markdown("### Do the Digit Insurance")
st.write("Trusted by 7 Crore+ Indians")

st.divider()

# ===============================
# INSURANCE TYPE CARDS
# ===============================
st.subheader("Select Insurance")

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
if st.button("View Prices 🚀"):

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

    except Exception as e:
        st.error("⚠️ Model mismatch! Please retrain model.")

st.divider()

# ===============================
# EXTRA SECTION
# ===============================
st.subheader("What Would You Like to Protect Today?")

col1, col2, col3, col4, col5 = st.columns(5)

col1.info("🚗 Car Insurance")
col2.info("🏍️ Bike Insurance")
col3.info("🏥 Health Insurance")
col4.info("🏢 Business Insurance")
col5.info("✈️ Travel Insurance")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("© 2026 Digit AI Insurance Clone")
