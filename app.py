import streamlit as st

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="🏥",
    layout="centered"
)

# ==========================
# TEST APP
# ==========================
st.title("🏥 Insurance Claim Predictor")

st.success("✅ App Running Successfully")

st.write("Agar ye page open ho raha hai to Streamlit deployment sahi hai.")

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

policy = st.selectbox(
    "Policy Type",
    ["Basic", "Premium", "Gold"]
)

claim_amount = st.number_input(
    "Claim Amount",
    min_value=1000,
    value=10000
)

income = st.number_input(
    "Income",
    min_value=10000,
    value=50000
)

medical = st.selectbox(
    "Medical History",
    ["Good", "Average", "Poor"]
)

claim_history = st.slider(
    "Claim History",
    0,
    5,
    1
)

fraud = st.selectbox(
    "Fraud Flag",
    ["No", "Yes"]
)

if st.button("🚀 Test Button"):
    st.success("Button Working Successfully")
