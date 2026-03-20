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
st.set_page_config(page_title="Digit Style Insurance", layout="wide")

# ===============================
# CSS (DIGIT STYLE)
# ===============================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #f5f5f5;
}

/* Header */
.header {
    display:flex;
    justify-content: space-between;
    align-items:center;
    padding: 10px 20px;
    background:white;
    border-radius:10px;
}

/* Cards */
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

/* Buttons */
.stButton>button {
    background-color:#ffcc00;
    color:black;
    border-radius:10px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
col1, col2 = st.columns([6,1])

with col1:
    st.markdown("<h2>digit insurance</h2>", unsafe_allow_html=True)

with col2:
    st.button("Login")

st.divider()

# ===============================
# TITLE
# ===============================
st.markdown("<h1 style='text-align:center;'>Do the Digit Insurance</h1>", unsafe_allow_html=True)

st.divider()

# ===============================
# INSURANCE CARDS
# ===============================
st.subheader("Choose Insurance")

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
insurance_map = {"Car":0,"Bike":1,"Health":2,"Life":3,"Travel":4}
insurance_val = insurance_map[insurance]

# ===============================
# PREDICTION
# ===============================
if st.button("View Prices 🚀"):

    try:
        features = np.array([[age, gender, insurance_val, policy, claim_amount, income, medical, claim_history, fraud]])
        features = scaler.transform(features)

        result = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if result == 1:
            st.success(f"✅ Claim Approved ({prob:.2f})")
        else:
            st.error(f"❌ Claim Rejected ({prob:.2f})")

        st.progress(float(prob))

    except:
        st.error("⚠️ Model mismatch! Retrain model.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("© 2026 Insurance AI Platform")
