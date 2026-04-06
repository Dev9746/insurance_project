import streamlit as st
import numpy as np
import joblib
import requests

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
    st.markdown("## 🤖 AI Based Insurance Claim")

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

# SCALE
try:
    features_scaled = scaler.transform(features)
except:
    st.error("❌ Model mismatch! Retrain model.")
    st.stop()

# ===============================
# PREDICTION
# ===============================
st.subheader("Prediction")

prediction_text = ""
prob = 0

if st.button("🚀 View Prediction"):

    result = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if result == 1:
        prediction_text = "✅ Claim Approved"
        st.success(prediction_text)
    else:
        prediction_text = "❌ Claim Rejected"
        st.error(prediction_text)

    st.metric("Approval Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    # Feature importance
    st.subheader("📊 Feature Importance")

    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    names = ["Age","Gender","Policy","Claim","Income","Medical","History","Fraud"]

    fig, ax = plt.subplots()
    ax.barh(names, importances)
    st.pyplot(fig)

# ===============================
# EMAIL SYSTEM (RESEND API)
# ===============================
st.divider()
st.subheader("📩 Contact / Notify")

message = st.text_area("Write message")

if st.button("Send Email"):

    if message.strip() == "":
        st.warning("⚠️ Please enter message")
    else:
        try:
            api_key = st.secrets["RESEND_API_KEY"]

            url = "https://api.resend.com/emails"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "from": "onboarding@resend.dev",
                "to": ["devs72527@gmail.com"],
                "subject": "Insurance App Message",
                "html": f"""
                <h3>New Message from Insurance App</h3>
                <p><b>Message:</b> {message}</p>
                <p><b>Prediction:</b> {prediction_text}</p>
                <p><b>Probability:</b> {prob:.2f}</p>
                """
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                st.success("✅ Email Sent Successfully 🚀")
            else:
                st.error("❌ Failed to send email")

        except Exception as e:
            st.error(f"Error: {e}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.write("🚀 AI Insurance System with Explainable AI")
