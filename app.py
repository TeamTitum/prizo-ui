# app.py – Arabiers AI Agent UI (Streamlit)
# Author: Suhail Jamaldeen
# Updated: 2025-11-07

import streamlit as st
import os
import time
from scripts.browser_console import console_log
from agent import generate_quotation
from scripts.create_logo import ensure_logo

# Ensure logo file exists (creates placeholder if missing) and set page title/favicon
try:
    ensure_logo()
except Exception:
    # don't fail if logo creation fails
    pass

# Must set page config before other Streamlit calls that affect the page
try:
    st.set_page_config(page_title="Arabiers AI Agent", page_icon="assets/favicon.ico")
except Exception:
    # ignore if running in an environment that disallows changing page config
    pass

# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# ──────────────────────────────────────────────────────────────
# Load CSS
# ──────────────────────────────────────────────────────────────
css_path = "assets/styles.css"
try:
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
console_log("App started")
cols = st.columns([1, 2, 1])
with cols[1]:
    try:
        st.image("assets/arabiers.png", width=200)
    except:
        pass
    st.markdown("<div class='arabiers-header'>Arabiers AI Agent</div>", unsafe_allow_html=True)

st.markdown("Ask about **Sri Lanka tourism and hotels** Eg. Find the 2025 contract rate for Cinnamon Bentota for 2 adults, half board.")

# ──────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g., Find the 2025 contract rate for Cinnamon Bentota for 2 adults, half board",
    key="question_input"
)

# ──────────────────────────────────────────────────────────────
# Submit
# ──────────────────────────────────────────────────────────────
if st.button("Ask Arabiers AI", type="primary"):
    if question.strip():
        with st.spinner("Arabiers AI is searching documents and calculating..."):
            try:
                response = generate_quotation(question.strip())
                st.session_state.last_response = response
                st.session_state.question_history.append({
                    "question": question,
                    "response": response,
                    "timestamp": time.strftime("%H:%M")
                })
            except Exception as e:
                st.error(f"Agent error: {e}")
    else:
        st.warning("Please enter a question.")

# ──────────────────────────────────────────────────────────────
# Display Response
# ──────────────────────────────────────────────────────────────
if st.session_state.last_response:
    st.markdown("### Arabiers AI Response")
    st.markdown(st.session_state.last_response, unsafe_allow_html=True)

    # PDF Download
    pdf_path = "quote.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Quotation",
                data=f,
                file_name=f"Arabiers_AI_Quote_{time.strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="secondary"
            )

    # Auto-scroll
    st.markdown(
        "<script>window.parent.document.querySelector('.main').scrollTop = document.body.scrollHeight;</script>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────
if st.button("Clear History"):
    st.session_state.question_history = []
    st.session_state.last_response = ""
    st.rerun()

if st.session_state.question_history:
    st.markdown("### Question History")
    for i, entry in enumerate(reversed(st.session_state.question_history)):
        with st.expander(f"Q{i+1}: {entry['question'][:60]}... • {entry['timestamp']}"):
            st.markdown(f"**Q**: {entry['question']}")
            st.markdown(f"**A**: {entry['response']}")