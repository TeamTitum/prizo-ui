import streamlit as st
from agent import generate_quotation
#Author: Suhail Jamaldeen

# Initialize session state for history if not already set
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

st.header("🛎️ Prizo AI Agent")
st.write("Ask about hotel rooms, availability, pricing, amenities, or meals.")
