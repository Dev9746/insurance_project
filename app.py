import streamlit as st

st.set_page_config(page_title="Test")

st.title("Dependency Test")

try:
    import joblib
    st.success("joblib imported")
except Exception as e:
    st.error(f"joblib error: {e}")

try:
    import sklearn
    st.success(f"sklearn imported: {sklearn.__version__}")
except Exception as e:
    st.error(f"sklearn error: {e}")

try:
    import numpy as np
    st.success(f"numpy imported: {np.__version__}")
except Exception as e:
    st.error(f"numpy error: {e}")
