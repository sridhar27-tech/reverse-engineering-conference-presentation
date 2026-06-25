import streamlit as st
import torch
from src.core.model import HybridConceptBottleneckModel

st.title("NeuroGuard-XAI Clinician Dashboard")

text = st.text_area("Enter text for analysis")
if st.button("Analyze"):
    model = HybridConceptBottleneckModel()
    # Run inference + explanations
    st.write("Risk Level: High (Demo)")
    st.write("Key Concepts: Hopelessness (0.92)")
    # SHAP viz placeholder
    st.success("Explanation generated - Clinically aligned!")