import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
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
h1, h2, h3, p {
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
    st.markdown("## 🤖 AI Based Insurance Claim")
with col2:
    st.button("Login")

st.divider()

# ===============================
# HERO
# ===============================
col1, col2 = st.columns([2.5,1])

with col1:
    st.markdown("### Smart Insurance Prediction System")
    st.write("AI powered system to predict claim approval instantly")

with col2:
    try:
        st.image("your_image.jpg", width=260)
    except:
        st.image("https://i.imgur.com/8Km9tLL.jpg", width=260)

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
if st.button("🚀 View Prediction"):

    result = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    col1, col2 = st.columns(2)

    if result == 1:
        col1.success("✅ Claim Approved")
    else:
        col1.error("❌ Claim Rejected")

    col2.metric("Approval Probability", f"{prob*100:.1f}%")

    st.progress(float(prob))

    # ===============================
    # SHAP EXPLANATION
    # ===============================
    st.subheader("🔍 Why this prediction?")

    explainer = shap.Explainer(model)
    shap_values = explainer(features_scaled)

    feature_names = [
        "Age","Gender","Policy","Claim Amount",
        "Income","Medical","Claim History","Fraud"
    ]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values.values[0]
    })

    st.bar_chart(shap_df.set_index("Feature"))

# ===============================
# CONTACT FORM (EMAIL ALERT)
# ===============================
st.divider()
st.subheader("📩 Contact / Query")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
message = st.text_area("Your Message")

if st.button("Send Message"):

    try:
        sender_email = "your_email@gmail.com"
        password = "your_app_password"

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)

        text = f"""
        Name: {name}
        Email: {email}
        Message: {message}
        """

        server.sendmail(sender_email, sender_email, text)
        server.quit()

        st.success("✅ Message Sent Successfully!")

    except:
        st.error("❌ Email sending failed")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 Ultimate AI Insurance System")
