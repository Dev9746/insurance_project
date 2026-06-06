import streamlit as st

st.set_page_config(page_title="Debug Test")

st.title("Debug Test")

import sklearn
st.write("sklearn version:", sklearn.__version__)

st.success("App Loaded")
