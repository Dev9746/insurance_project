import streamlit as st
import joblib
import traceback

st.set_page_config(page_title="Model Test")

st.title("Model Test")

try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ scaler.pkl loaded")
except Exception:
    st.error("❌ scaler.pkl failed")
    st.code(traceback.format_exc())

try:
    model = joblib.load("model.pkl")
    st.success("✅ model.pkl loaded")
except Exception:
    st.error("❌ model.pkl failed")
    st.code(traceback.format_exc())
