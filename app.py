import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import smtplib

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
# STYLE
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
col1, col2 = st.columns([8,1])

with col1:
    st.markdown("##  AI Based Insurance Claim")

with col2:
    st.button("Login")

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

# ===============================
# ENCODING
# ===============================
gender = 1 if gender == "Male" else 0
policy = {"Basic":0,"Premium":1,"Gold":2}[policy]
medical = {"Good":0,"Average":1,"Poor":2}[medical]

features = np.array([[age, gender, policy, claim_amount, income, medical, claim_history, fraud]])
features_scaled = scaler.transform(features)

# ===============================
# PREDICTION
# ===============================
if st.button(" View Prediction"):

    result = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if result == 1:
        st.success(" Claim Approved")
    else:
        st.error(" Claim Rejected")

    st.metric("Approval Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    # ===============================
    # SHAP EXPLAINABILITY
    # ===============================
    st.subheader(" AI Explanation (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(features_scaled)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.subheader(" Feature Importance")

    importances = model.feature_importances_
    feature_names = ["Age","Gender","Policy","Claim","Income","Medical","History","Fraud"]

    fig2, ax2 = plt.subplots()
    ax2.barh(feature_names, importances)
    st.pyplot(fig2)

# ===============================
# EMAIL SYSTEM
# ===============================
st.divider()
st.subheader(" Contact / Notify")

message = st.text_area("Write message (for notifications only)")

if st.button("Send Email"):
    try:
        sender_email = "devs72527@gmail.com"
        receiver_email = "devs72527@gmail.com"
        password = "YOUR_APP_PASSWORD"  #  replace this

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)

        server.sendmail(sender_email, receiver_email, message)
        server.quit()

        st.success(" Email Sent Successfully")

    except Exception as e:
        st.error(" Email Failed. Check App Password.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write(" AI Insurance System with Explainable AI")
