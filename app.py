import streamlit as st
import sklearn
import joblib

st.write("sklearn:", sklearn.__version__)

model = joblib.load("model.pkl")
st.success("model loaded")

scaler = joblib.load("scaler.pkl")
st.success("scaler loaded")
