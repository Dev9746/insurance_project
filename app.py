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
# CLEAN UI
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f5f5;
}

h1, h2, h3, h4, p, label {
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
col1, col2 = st.columns([8,1])

with col1:
    st.markdown("<h2 style='margin-top:10px;'>🤖 AI Based Insurance Claim</h2>", unsafe_allow_html=True)

with col2:
    st.markdown("<div style='margin-top:15px;'>", unsafe_allow_html=True)
    st.button("Login")
    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# HERO SECTION (CENTER IMAGE)
# ===============================
col1, col2 = st.columns([2.5,1])

with col1:
    st.markdown("### Smart Insurance Prediction System")
    st.write("AI powered system to predict claim approval instantly")
    st.write("Professional ML-based insurance approval system")

with col2:
    st.markdown("<div style='margin-top:-50px;'>", unsafe_allow_html=True)  # 👈 move UP
    try:
        st.image("your_image.jpg", width=260)
    except:
        st.image("https://i.imgur.com/8Km9tLL.jpg", width=260)
    st.markdown("</div>", unsafe_allow_html=True)  # spacing
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
st.write("🚀 AI Based Insurance Claim System | Final Year Project")
